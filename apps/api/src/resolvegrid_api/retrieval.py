"""Lexical (Postgres full-text) search, vector (pgvector) search, and
Reciprocal Rank Fusion (RRF) combining the two ranked lists (Phase 7 Task
4), plus authorization-aware filtering and a deterministic sufficiency
check (Phase 7 Task 5).

Dependency-direction note (mirrors `knowledge_store.py`'s module docstring
for the same reasoning): this module lives in `apps/api` and owns all DB
access -- it imports `resolvegrid_api.models.knowledge` and takes a
SQLAlchemy `Session`, exactly like `knowledge_store.py`. `services/retrieval`
stays a pure library with zero SQLAlchemy/DB dependency; nothing in this
module is imported from there.

Authz note (Task 5): `vector_search`/`lexical_search` both take an
`AuthzFilter` (`resolvegrid_api.retrieval_authz.build_authz_filter`) as a
**required** keyword-only parameter -- not optional, and not applied
post-hoc to the Python result list. The filter is folded directly into
each query's `WHERE` clause via a `Chunk` -> `DocumentVersion` -> `Document`
join, so an unauthorized chunk is excluded at the database level and never
materializes as a row in the result set at all. This is a deliberate
breaking change to Task 4's signatures (see `test_retrieval.py`).
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from resolvegrid_api.retrieval_authz import AuthzFilter

# RRF's fusion constant. `k=60` is used here, matching the value from the
# original RRF paper (Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion
# Outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009)
# and the value most commonly used as a sane default in IR systems that
# implement RRF (e.g. Elasticsearch's `rank.rrf` retriever, OpenSearch's
# hybrid search). No corpus-specific tuning signal exists yet for this
# codebase (Phase 7's evaluation baseline, Task 9, hasn't run), so there is
# no better-justified value to use instead -- `k=60` is a well-documented,
# defensible default rather than an arbitrary guess.
DEFAULT_RRF_K = 60


def _vector_literal(vector: list[float]) -> str:
    """Render a Python float list as the string pgvector's `::vector` cast
    expects (`"[0.1,0.2,...]"`), for use as a bind parameter in raw SQL.

    Verified directly against the live `resolvegrid-postgres` container:
    binding a plain Python list is not accepted by psycopg for a raw
    `text()` query (pgvector's adapter registration only applies to
    `Vector`-typed ORM columns, not ad-hoc `text()` SQL) -- casting a
    bracketed, comma-separated string literal via `(:param)::vector` is
    the form that actually works.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def vector_search(
    session: Session, query_vector: list[float], *, limit: int, authz_filter: AuthzFilter
) -> list[tuple[int, int]]:
    """Return up to `limit` chunk ids nearest to `query_vector` by pgvector
    cosine distance (the `<=>` operator -- confirmed against
    `apps/api/tests/test_knowledge_store.py`'s existing cosine-distance
    test), nearest-first, restricted to chunks `authz_filter` authorizes.

    Returns `(chunk_id, rank_position)` pairs, `rank_position` starting at
    1 for the nearest match. One embedding per chunk is assumed (Phase 7's
    current ingestion scope, per `knowledge_store.py`); if a chunk has
    multiple embedding rows (e.g. re-embedded with a newer model version),
    each row is a distinct candidate and the chunk id may appear more than
    once at different ranks -- that reconciliation is out of scope here.

    `authz_filter` is required, not optional (Phase 7 Task 5): it joins
    `embedding` -> `chunk` -> `document_version` -> `document` (the only
    path to `Document.access_scope_tags`, per the prior schema review) and
    filters in the `WHERE` clause -- an unauthorized chunk is excluded by
    the query itself and never reaches Python, rather than being filtered
    out of the returned list after the fact.
    """
    rows = session.execute(
        text(
            "SELECT embedding.chunk_id "
            "FROM embedding "
            "JOIN chunk ON chunk.id = embedding.chunk_id "
            "JOIN document_version ON document_version.id = chunk.document_version_id "
            "JOIN document ON document.id = document_version.document_id "
            "WHERE (:unrestricted "
            "OR document.access_scope_tags = ARRAY[]::varchar[] "
            "OR document.access_scope_tags && (:allowed_tags)::varchar[]) "
            "ORDER BY embedding.vector <=> (:query_vector)::vector ASC "
            "LIMIT :limit"
        ),
        {
            "query_vector": _vector_literal(query_vector),
            "limit": limit,
            "unrestricted": authz_filter.unrestricted,
            "allowed_tags": sorted(authz_filter.allowed_tags),
        },
    ).all()
    return [(row.chunk_id, position) for position, row in enumerate(rows, start=1)]


def lexical_search(
    session: Session, query_text: str, *, limit: int, authz_filter: AuthzFilter
) -> list[tuple[int, int]]:
    """Return up to `limit` chunk ids matching `query_text` by Postgres
    full-text search, ranked by `ts_rank_cd` descending (highest-rank-first),
    restricted to chunks `authz_filter` authorizes.

    Uses `chunk.search_vector` (migration 0011's stored generated
    `tsvector` column, GIN-indexed) rather than computing
    `to_tsvector` inline per query -- see that migration's docstring for
    the reasoning. `websearch_to_tsquery('english', query_text)` parses
    `query_text` (confirmed against the live DB: it tolerates natural free
    -text queries, unlike `to_tsquery`'s operator syntax, and is Postgres's
    current-recommended function for user-typed search strings -- unlike
    the older `plainto_tsquery`, it also understands quoted phrases and
    `-exclude` terms).

    Only chunks that actually match (`search_vector @@ query`) AND that
    `authz_filter` authorizes are returned -- there is no zero-rank
    fallback -- so this can return fewer than `limit` results, including
    zero, if nothing matches or nothing is authorized.

    Returns `(chunk_id, rank_position)` pairs, `rank_position` starting at
    1 for the highest-ranked match.

    `authz_filter` is required, not optional (Phase 7 Task 5): see
    `vector_search`'s docstring above -- the same join-and-filter-in-SQL
    reasoning applies here.
    """
    rows = session.execute(
        text(
            "SELECT chunk.id AS chunk_id, "
            "ts_rank_cd(chunk.search_vector, websearch_to_tsquery('english', :query_text)) AS rank "
            "FROM chunk "
            "JOIN document_version ON document_version.id = chunk.document_version_id "
            "JOIN document ON document.id = document_version.document_id "
            "WHERE chunk.search_vector @@ websearch_to_tsquery('english', :query_text) "
            "AND (:unrestricted "
            "OR document.access_scope_tags = ARRAY[]::varchar[] "
            "OR document.access_scope_tags && (:allowed_tags)::varchar[]) "
            "ORDER BY rank DESC "
            "LIMIT :limit"
        ),
        {
            "query_text": query_text,
            "limit": limit,
            "unrestricted": authz_filter.unrestricted,
            "allowed_tags": sorted(authz_filter.allowed_tags),
        },
    ).all()
    return [(row.chunk_id, position) for position, row in enumerate(rows, start=1)]


def fuse_rrf(
    vector_results: list[tuple[int, int]],
    lexical_results: list[tuple[int, int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine a vector-search ranked list and a
    lexical-search ranked list into one fused ranking.

    Each input is a list of `(chunk_id, rank_position)` pairs (1-based
    rank, as returned by `vector_search`/`lexical_search` above -- the
    rank position is taken as given, not re-derived from list order, so
    callers may pass a subset/reordering if they have one).

    Formula (the standard RRF formula, Cormack et al. 2009):

        score(chunk) = sum over each input list containing chunk of
                       1 / (k + rank_in_that_list)

    A chunk absent from a list contributes 0 for that list (not a
    penalty term) -- a chunk in only one list still gets a score, just a
    smaller one than an equivalently-ranked chunk that appears in both.

    Returns `(chunk_id, fused_score)` pairs ordered by fused score
    descending. Ties are broken by `chunk_id` ascending, purely for
    deterministic/reproducible output ordering (not a ranking signal).
    """
    scores: dict[int, float] = {}
    for chunk_id, rank in vector_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in lexical_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


# Sufficiency threshold: the score a chunk gets from ranking #1 in exactly
# *one* of the two ranked lists (vector or lexical), with the default RRF
# constant -- `1 / (DEFAULT_RRF_K + 1)`. This is deliberately not an
# arbitrary small number: RRF scores aren't a normalized 0-1 similarity, so
# "min top-k score" only means something when pinned to a concrete point on
# the RRF scale. Requiring at least this much says "the best result must be
# at least as good as being the single best match in one signal" -- a
# result that only ever placed moderately (e.g. rank 5+) in both lists
# scores below this and is correctly treated as weak/no-match, whereas a
# true top-1 hit in either signal (a very common, and good, outcome) always
# clears it. A fused top result scoring *below* this line means neither
# signal was confident about anything -- exactly the "insufficient
# evidence, abstain/clarify" case per the plan doc.
DEFAULT_MIN_SCORE_THRESHOLD = 1.0 / (DEFAULT_RRF_K + 1)

# Require at least one fused result to say anything at all. A future
# caller with a stricter evidentiary bar (e.g. "need corroboration from
# 2+ chunks before answering") can pass a higher `min_results` -- this
# default only rules out the trivial empty-result case.
DEFAULT_MIN_RESULTS = 1


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of `assess_sufficiency`: enough for a caller to decide
    abstain (`sufficient=False`) vs. proceed to compose an answer
    (`sufficient=True`), plus enough detail to log/explain the decision.
    """

    sufficient: bool
    reason: str
    result_count: int
    top_score: float | None


def assess_sufficiency(
    fused_results: list[tuple[int, float]],
    *,
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    min_results: int = DEFAULT_MIN_RESULTS,
) -> SufficiencyResult:
    """Deterministic sufficiency check over `fuse_rrf`'s output -- no model
    call (per the plan doc: "deterministic checks first (min top-k score,
    required-field coverage)... v1 just routes to abstain/clarify").

    Two checks, both must pass for `sufficient=True`:
      1. Result-count coverage: `len(fused_results) >= min_results`.
      2. Min top-k score: the best (`fused_results[0]`) score is
         `>= min_score_threshold` (boundary case is sufficient, not just
         strictly greater -- a result scoring *exactly* the threshold has
         met the bar, not missed it by definition).

    `fused_results` is assumed already sorted descending by score (as
    `fuse_rrf` returns it) -- this function does not re-sort; it only
    inspects `fused_results[0]` for the top score.

    "required-field coverage" (the plan doc's other named deterministic
    check) is not implemented here: `fuse_rrf`'s output is bare
    `(chunk_id, score)` pairs with no field/metadata payload attached, so
    there is nothing to check field-presence on at this layer yet -- that
    would apply once a caller (Task 7's graph node) attaches chunk
    text/citation metadata to each result and wants to additionally
    require e.g. "every result must carry a resolvable citation". Left as
    a documented gap, not silently skipped.
    """
    if len(fused_results) < min_results:
        return SufficiencyResult(
            sufficient=False,
            reason=f"result_count {len(fused_results)} below min_results {min_results}",
            result_count=len(fused_results),
            top_score=None,
        )

    top_score = fused_results[0][1]
    if top_score < min_score_threshold:
        return SufficiencyResult(
            sufficient=False,
            reason=f"top_score {top_score!r} below min_score_threshold {min_score_threshold!r}",
            result_count=len(fused_results),
            top_score=top_score,
        )

    return SufficiencyResult(
        sufficient=True,
        reason="min_results and min_score_threshold both satisfied",
        result_count=len(fused_results),
        top_score=top_score,
    )
