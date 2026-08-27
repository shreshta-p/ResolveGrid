"""Lexical (Postgres full-text) search, vector (pgvector) search, and
Reciprocal Rank Fusion (RRF) combining the two ranked lists (Phase 7 Task 4).

Dependency-direction note (mirrors `knowledge_store.py`'s module docstring
for the same reasoning): this module lives in `apps/api` and owns all DB
access -- it imports `resolvegrid_api.models.knowledge` and takes a
SQLAlchemy `Session`, exactly like `knowledge_store.py`. `services/retrieval`
stays a pure library with zero SQLAlchemy/DB dependency; nothing in this
module is imported from there.

Scope note: `vector_search` here is a minimal "top-N nearest by cosine
distance" query -- just enough to be a real input to `fuse_rrf`. It does
not implement authz filtering (Task 5) or get wired into the agent graph
(Task 7); both are later, separately-scoped tasks.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

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
    session: Session, query_vector: list[float], *, limit: int
) -> list[tuple[int, int]]:
    """Return up to `limit` chunk ids nearest to `query_vector` by pgvector
    cosine distance (the `<=>` operator -- confirmed against
    `apps/api/tests/test_knowledge_store.py`'s existing cosine-distance
    test), nearest-first.

    Returns `(chunk_id, rank_position)` pairs, `rank_position` starting at
    1 for the nearest match. One embedding per chunk is assumed (Phase 7's
    current ingestion scope, per `knowledge_store.py`); if a chunk has
    multiple embedding rows (e.g. re-embedded with a newer model version),
    each row is a distinct candidate and the chunk id may appear more than
    once at different ranks -- that reconciliation is out of scope here.
    """
    rows = session.execute(
        text(
            "SELECT chunk_id "
            "FROM embedding "
            "ORDER BY vector <=> (:query_vector)::vector ASC "
            "LIMIT :limit"
        ),
        {"query_vector": _vector_literal(query_vector), "limit": limit},
    ).all()
    return [(row.chunk_id, position) for position, row in enumerate(rows, start=1)]


def lexical_search(
    session: Session, query_text: str, *, limit: int
) -> list[tuple[int, int]]:
    """Return up to `limit` chunk ids matching `query_text` by Postgres
    full-text search, ranked by `ts_rank_cd` descending (highest-rank-first).

    Uses `chunk.search_vector` (migration 0011's stored generated
    `tsvector` column, GIN-indexed) rather than computing
    `to_tsvector` inline per query -- see that migration's docstring for
    the reasoning. `websearch_to_tsquery('english', query_text)` parses
    `query_text` (confirmed against the live DB: it tolerates natural free
    -text queries, unlike `to_tsquery`'s operator syntax, and is Postgres's
    current-recommended function for user-typed search strings -- unlike
    the older `plainto_tsquery`, it also understands quoted phrases and
    `-exclude` terms).

    Only chunks that actually match (`search_vector @@ query`) are
    returned -- there is no zero-rank fallback -- so this can return fewer
    than `limit` results, including zero, if nothing matches.

    Returns `(chunk_id, rank_position)` pairs, `rank_position` starting at
    1 for the highest-ranked match.
    """
    rows = session.execute(
        text(
            "SELECT id AS chunk_id, "
            "ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) AS rank "
            "FROM chunk "
            "WHERE search_vector @@ websearch_to_tsquery('english', :query_text) "
            "ORDER BY rank DESC "
            "LIMIT :limit"
        ),
        {"query_text": query_text, "limit": limit},
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
