"""Retrieval evaluation harness (Phase 7 Task 9): loads the hand-labeled
golden query set (`eval/golden/phase7_retrieval_v1.jsonl`), runs each case
through the real `vector_search` + `lexical_search` + `fuse_rrf` pipeline
(`resolvegrid_api.retrieval`) under the case's specified `AuthzFilter`, and
computes standard IR metrics (recall@k, precision@k, MRR, nDCG@k) against
the hand-labeled relevant set.

Chunk-identity resolution (why (title, ordinal), not `Chunk.id`)
------------------------------------------------------------------
`Chunk.id` is an autoincrement PK that is not stable across re-ingestion
runs (a fresh DB, or a wiped/reseeded one, assigns different ids). The
golden set instead hand-labels relevance by `(Document.title,
Chunk.ordinal)` pairs -- both are stable across re-ingestion of the same
seed corpus content (`title` is `ingest_document`'s natural key;
`ordinal` is deterministic given the same `parser_version`/
`chunking_version`, per `resolvegrid_retrieval.chunker`'s docstring).
`resolve_relevant_chunk_ids` below resolves those pairs to real
`Chunk.id` values at eval-run time, against whatever corpus is actually
loaded in the session passed in.

Ingestion is NOT this module's responsibility (decided and documented,
per this task's instructions)
------------------------------------------------------------------------
`run_eval`/`evaluate_case` below assume the seed corpus is *already*
ingested into the session/DB they are given -- they only read. This
mirrors `resolvegrid_api.retrieval`'s own layering (it takes a `Session`
and never ingests anything itself) and keeps this module usable against
either a fresh transactional `db_session` that just ran
`run_seed_corpus_ingestion` (the test's approach, see
`test_eval_retrieval.py` -- no DB residue, since that fixture rolls back)
or a long-lived dev DB that was seeded once out of band. `main()` below
is the one place that *does* ingest, for convenient manual/CLI runs: it
ingests and evaluates inside one session via `flush()` (not `commit()`),
then unconditionally rolls back at the end, so a manual run leaves zero
residue in the shared dev DB either.

k and search-limit choice
------------------------------------------------------------------------
`DEFAULT_K = 5`. The seed corpus is small (8 documents, 36 chunks total
per `dump_chunks` run against the real chunker as of this task), and
every golden case's hand-labeled answer lives in exactly one chunk (each
corpus document's sections are short enough that the chunker emits one
chunk per heading section, never splitting a section across chunks) --
so a single relevant chunk needs to land inside the top-k for recall@k to
be meaningful at all. k=5 mirrors a realistic citation UI (surfacing a
small handful of sources, not dumping the whole authorized corpus) while
staying restrictive enough to be discriminating: some authz-scoped
subsets here have as few as 4-9 authorized chunks total, so a much larger
k (e.g. 10+) would trivially recall almost everything and stop
distinguishing a working retriever from a broken one.

`DEFAULT_SEARCH_LIMIT = 10`: `vector_search`/`lexical_search` are each
called with `limit=10` (twice k) before fusion, so `fuse_rrf` has enough
of each signal's ranked list to produce a meaningful fused top-5 --
capping the per-signal search at exactly k=5 would let a chunk that's
merely mediocre on one signal (but excellent on the other) fall out of
that signal's list before fusion even has a chance to combine them.

Metric formulas (implemented directly, no new dependency -- see
`_recall_at_k`/`_precision_at_k`/`_reciprocal_rank`/`_ndcg_at_k` below for
each, and `apps/api/tests/test_eval_retrieval.py` for hand-computed unit
tests against small fixed examples)
------------------------------------------------------------------------
- recall@k    = |relevant chunks in top-k| / |relevant chunks|
- precision@k = |relevant chunks in top-k| / k   (the standard IR
  definition -- divides by k itself, not by however many results were
  actually returned, so a query that returns fewer than k results is
  correctly penalized rather than getting an inflated score from a
  smaller denominator)
- MRR         = 1 / (rank of the first relevant chunk in the full fused
  list, i.e. up to `search_limit`, not capped to k) -- 0.0 if no relevant
  chunk appears anywhere in the fused list at all
- nDCG@k      = DCG@k / IDCG@k, with binary relevance (rel=1 if the
  chunk at that rank is in the labeled relevant set, else 0), DCG@k =
  sum_{i=1..k} rel_i / log2(i+1), and IDCG@k computed from the ideal
  ranking (min(|relevant|, k) relevant chunks placed first)

A case with an EMPTY hand-labeled relevant set (a "no good match" case,
or an authz-zero-result case) has no well-defined recall/precision/MRR/
nDCG -- those are `None` for that case and excluded from the averages
computed over "answerable" cases, per this task's explicit requirement
that a correct "no good match" must not be penalized as if it were a
retrieval failure. Those cases are still checked, just via different
signals: `must_not_appear` (a hard leakage check -- listed chunks must
never appear in either raw search result list, at any rank, not just be
absent from the fused top-k) and `assess_sufficiency`'s verdict (recorded
for information, not asserted as pass/fail here, since a case can
legitimately have `relevant=[]` while other, non-labeled content still
scores just high enough to pass the sufficiency bar -- e.g. the "parental
leave" case, which has real lexical overlap with genuinely different
content).

Distractor handling (the deliberately-superseded VPN v1/v2 pair)
------------------------------------------------------------------------
A case may list `distractor` pairs: plausible-but-wrong chunks (almost
always the superseded VPN v1 policy) that a naive lexical/vector match
could easily rank above the correct answer. `evaluate_case` records
`distractor_beats_best_relevant`: True if any distractor chunk's fused
rank is better (numerically lower) than the best-ranked relevant chunk's
fused rank. This is reported per-case and aggregated, not silently
folded into recall/precision (a distractor ranking #2 behind a correct
#1 still means recall@k=1.0 -- the distractor check is a distinct,
additional signal this task's plan explicitly calls out).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion
from resolvegrid_api.retrieval import assess_sufficiency, fuse_rrf, lexical_search, vector_search
from resolvegrid_api.retrieval_authz import AuthzFilter
from resolvegrid_retrieval.embedder import DEFAULT_EMBEDDING_MODEL, embed_texts

_DEFAULT_GOLDEN_PATH = (
    Path(__file__).resolve().parents[4] / "eval" / "golden" / "phase7_retrieval_v1.jsonl"
)

DEFAULT_K = 5
DEFAULT_SEARCH_LIMIT = 10


# ---------------------------------------------------------------------------
# Golden case loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenCase:
    """One hand-written golden query, loaded from the JSONL golden set.

    `relevant`/`distractor`/`must_not_appear` are `(Document.title,
    Chunk.ordinal)` pairs -- see module docstring for why, and
    `resolve_relevant_chunk_ids` for how they're resolved to real
    `Chunk.id`s at eval-run time.
    """

    query: str
    authz: AuthzFilter
    relevant: frozenset[tuple[str, int]]
    distractor: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    must_not_appear: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    note: str = ""


def _authz_from_dict(raw: dict) -> AuthzFilter:
    return AuthzFilter(
        unrestricted=bool(raw.get("unrestricted", False)),
        allowed_tags=frozenset(raw.get("allowed_tags", [])),
    )


def _pairs_from_json(raw: list) -> frozenset[tuple[str, int]]:
    return frozenset((title, ordinal) for title, ordinal in raw)


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    """Parse `eval/golden/phase7_retrieval_v1.jsonl` (or `path`, if given)
    into `GoldenCase` objects, one per non-blank line.
    """
    golden_path = path or _DEFAULT_GOLDEN_PATH
    cases: list[GoldenCase] = []
    with golden_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(
                GoldenCase(
                    query=raw["query"],
                    authz=_authz_from_dict(raw["authz"]),
                    relevant=_pairs_from_json(raw["relevant"]),
                    distractor=_pairs_from_json(raw.get("distractor", [])),
                    must_not_appear=_pairs_from_json(raw.get("must_not_appear", [])),
                    note=raw.get("note", ""),
                )
            )
    return cases


def resolve_relevant_chunk_ids(
    session: Session, pairs: frozenset[tuple[str, int]]
) -> frozenset[int]:
    """Resolve a set of `(Document.title, Chunk.ordinal)` pairs to real
    `Chunk.id` values against whatever corpus is currently loaded in
    `session`.

    Raises `ValueError` (rather than silently dropping or arbitrarily
    picking one) if any pair doesn't resolve to exactly one chunk --
    either zero matches (the golden set's hand-labeling is stale against
    the current corpus/chunker output) or more than one (the same title
    has multiple `DocumentVersion`s with a chunk at that ordinal, e.g.
    after a re-ingestion of changed content -- `.all()` + an explicit
    count check, not `.first()`, so this can't silently pick an arbitrary
    version's chunk instead of raising).
    """
    resolved: set[int] = set()
    for title, ordinal in pairs:
        chunk_ids = session.scalars(
            select(Chunk.id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Document.title == title, Chunk.ordinal == ordinal)
        ).all()
        if len(chunk_ids) != 1:
            raise ValueError(
                f"golden set references (title={title!r}, ordinal={ordinal}) "
                f"which resolves to {len(chunk_ids)} chunks in the current "
                "corpus, not exactly 1 -- stale hand-labeling (re-check "
                "against a fresh chunk_markdown() dump of eval/corpus/*.md), "
                "a corpus/chunker change since the golden set was written, "
                "or multiple DocumentVersions of the same title (re-ingestion "
                "history) making the (title, ordinal) key ambiguous."
            )
        resolved.add(chunk_ids[0])
    return frozenset(resolved)


# ---------------------------------------------------------------------------
# Metric formulas (pure functions, no DB) -- see module docstring for each
# ---------------------------------------------------------------------------


def _recall_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
    if not relevant_ids:
        return None
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def _precision_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
    if not relevant_ids:
        return None
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / k


def _reciprocal_rank(ranked_ids: list[int], relevant_ids: frozenset[int]) -> float | None:
    if not relevant_ids:
        return None
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def _ndcg_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
    if not relevant_ids:
        return None
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    query: str
    note: str
    relevant_count: int
    ranked_chunk_ids: list[int]
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    sufficient: bool
    top_score: float | None
    distractor_beats_best_relevant: bool | None
    leaked_chunk_ids: frozenset[int]  # must_not_appear chunks that leaked through


def evaluate_case(
    session: Session,
    case: GoldenCase,
    *,
    k: int = DEFAULT_K,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> CaseResult:
    relevant_ids = resolve_relevant_chunk_ids(session, case.relevant)
    distractor_ids = resolve_relevant_chunk_ids(session, case.distractor)
    must_not_appear_ids = resolve_relevant_chunk_ids(session, case.must_not_appear)

    query_vector = embed_texts([case.query], model=embedding_model)[0]
    vector_results = vector_search(
        session, query_vector, limit=search_limit, authz_filter=case.authz
    )
    lexical_results = lexical_search(
        session, case.query, limit=search_limit, authz_filter=case.authz
    )
    fused = fuse_rrf(vector_results, lexical_results)
    ranked_chunk_ids = [chunk_id for chunk_id, _ in fused]

    sufficiency = assess_sufficiency(fused)

    distractor_beats_best_relevant: bool | None = None
    if distractor_ids and relevant_ids:
        rank_of = {chunk_id: position for position, chunk_id in enumerate(ranked_chunk_ids)}
        best_relevant_rank = min(
            (rank_of[c] for c in relevant_ids if c in rank_of), default=None
        )
        best_distractor_rank = min(
            (rank_of[c] for c in distractor_ids if c in rank_of), default=None
        )
        if best_relevant_rank is None:
            # Relevant chunk didn't even appear -- trivially, any present
            # distractor "beats" it.
            distractor_beats_best_relevant = best_distractor_rank is not None
        else:
            distractor_beats_best_relevant = (
                best_distractor_rank is not None and best_distractor_rank < best_relevant_rank
            )

    # Leakage check for must_not_appear: these chunks must be absent from
    # the RAW vector_search/lexical_search result sets entirely (not just
    # the fused top-k) -- authz filtering happens in the SQL query itself
    # (retrieval.py), so this is checking the same property
    # test_adversarial_authz_filter_blocks_cross_scope_leakage checks, on
    # this task's own golden queries.
    raw_ids = {c for c, _ in vector_results} | {c for c, _ in lexical_results}
    leaked_chunk_ids = frozenset(must_not_appear_ids & raw_ids)

    return CaseResult(
        query=case.query,
        note=case.note,
        relevant_count=len(relevant_ids),
        ranked_chunk_ids=ranked_chunk_ids,
        recall_at_k=_recall_at_k(ranked_chunk_ids, relevant_ids, k),
        precision_at_k=_precision_at_k(ranked_chunk_ids, relevant_ids, k),
        reciprocal_rank=_reciprocal_rank(ranked_chunk_ids, relevant_ids),
        ndcg_at_k=_ndcg_at_k(ranked_chunk_ids, relevant_ids, k),
        sufficient=sufficiency.sufficient,
        top_score=sufficiency.top_score,
        distractor_beats_best_relevant=distractor_beats_best_relevant,
        leaked_chunk_ids=leaked_chunk_ids,
    )


# ---------------------------------------------------------------------------
# Full golden-set run + aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSummary:
    k: int
    search_limit: int
    case_results: list[CaseResult]

    @property
    def answerable_case_results(self) -> list[CaseResult]:
        """Cases with a non-empty hand-labeled relevant set -- the only
        ones recall@k/precision@k/MRR/nDCG@k are defined for."""
        return [c for c in self.case_results if c.relevant_count > 0]

    @property
    def mean_recall_at_k(self) -> float:
        cases = self.answerable_case_results
        return sum(c.recall_at_k for c in cases) / len(cases)

    @property
    def mean_precision_at_k(self) -> float:
        cases = self.answerable_case_results
        return sum(c.precision_at_k for c in cases) / len(cases)

    @property
    def mean_reciprocal_rank(self) -> float:
        cases = self.answerable_case_results
        return sum(c.reciprocal_rank for c in cases) / len(cases)

    @property
    def mean_ndcg_at_k(self) -> float:
        cases = self.answerable_case_results
        return sum(c.ndcg_at_k for c in cases) / len(cases)

    @property
    def any_leakage(self) -> bool:
        return any(c.leaked_chunk_ids for c in self.case_results)

    @property
    def any_distractor_beats_relevant(self) -> bool:
        return any(c.distractor_beats_best_relevant for c in self.case_results)


def run_eval(
    session: Session,
    cases: list[GoldenCase] | None = None,
    *,
    k: int = DEFAULT_K,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> EvalSummary:
    """Run every golden case (loaded from the default path if `cases` is
    not given) against `session` and return the aggregated `EvalSummary`.

    Assumes the seed corpus is already ingested into `session` -- see
    module docstring's "Ingestion is NOT this module's responsibility"
    section.
    """
    resolved_cases = cases if cases is not None else load_golden_cases()
    results = [
        evaluate_case(session, case, k=k, search_limit=search_limit, embedding_model=embedding_model)
        for case in resolved_cases
    ]
    return EvalSummary(k=k, search_limit=search_limit, case_results=results)


def format_report(summary: EvalSummary) -> str:
    """Human-readable report of an `EvalSummary`, used by `main()` and
    handy for pasting into `docs/EXPERIMENT_REGISTRY.md`.
    """
    lines = [
        f"Golden set: {len(summary.case_results)} cases "
        f"({len(summary.answerable_case_results)} answerable, "
        f"{len(summary.case_results) - len(summary.answerable_case_results)} no-good-match), "
        f"k={summary.k}, search_limit={summary.search_limit}",
        f"recall@{summary.k}:    {summary.mean_recall_at_k:.4f}",
        f"precision@{summary.k}: {summary.mean_precision_at_k:.4f}",
        f"MRR:          {summary.mean_reciprocal_rank:.4f}",
        f"nDCG@{summary.k}:      {summary.mean_ndcg_at_k:.4f}",
        f"leakage detected (must_not_appear violated): {summary.any_leakage}",
        f"distractor outranked correct answer in any case: {summary.any_distractor_beats_relevant}",
        "",
        "Per-case detail:",
    ]
    for c in summary.case_results:
        lines.append(
            f"  - {c.query!r}: recall={c.recall_at_k} precision={c.precision_at_k} "
            f"rr={c.reciprocal_rank} ndcg={c.ndcg_at_k} sufficient={c.sufficient} "
            f"top_score={c.top_score} distractor_beats_relevant={c.distractor_beats_best_relevant} "
            f"leaked={sorted(c.leaked_chunk_ids)}"
        )
    return "\n".join(lines)


def main() -> None:
    """Standalone CLI entry point for a manual/reproducible run against the
    live dev DB: `uv run --package resolvegrid-api python -m
    resolvegrid_api.eval_retrieval`.

    Ingests the seed corpus and runs the full golden set inside one
    session, `flush()`-ing (not committing) so `run_eval`'s queries see
    the ingested rows within the same transaction, then unconditionally
    `rollback()`s at the end -- simpler than `test_ingestion_worker.py`'s
    delete-what-we-created cleanup (which is only needed there because
    that test goes through a genuinely separate, real-Arq-committing
    session it cannot roll back). A manual run here leaves zero residue
    in the shared dev DB, matching `db_session`'s fixture-level
    convention applied at the script level.
    """
    from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        try:
            run_seed_corpus_ingestion(session)
            session.flush()

            summary = run_eval(session)
            print(format_report(summary))
        finally:
            session.rollback()
    engine.dispose()


if __name__ == "__main__":
    main()
