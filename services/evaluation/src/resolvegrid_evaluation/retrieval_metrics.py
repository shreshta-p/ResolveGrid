"""Retrieval IR-metric graders (Phase 10 Task 3): public recall@k/
precision@k/MRR/nDCG@k formulas, plus a per-case retrieval grader,
consolidated out of `apps/api/src/resolvegrid_api/eval_retrieval.py`'s
previously-private `_recall_at_k`/`_precision_at_k`/`_reciprocal_rank`/
`_ndcg_at_k` functions and that module's `evaluate_case`'s inline
leakage-check / distractor-beats-relevant-check logic.

This is a pure refactor-for-reuse, not a behavior change (Phase 10 Task 3):
every formula below is byte-for-byte the same computation the `_`-prefixed
originals performed, just renamed and made public. `eval_retrieval.py` now
imports these instead of keeping its own copies -- see that module's
docstring for how it re-exports the old private names so
`apps/api/tests/test_eval_retrieval.py` keeps passing unmodified.

Dependency direction (must read before adding any import here)
------------------------------------------------------------------------
Same rule as `schema.py`/`graders.py` (see their module docstrings): this
package must NEVER import `resolvegrid_api`, SQLAlchemy, or anything
DB-shaped. All of `apps/api/src/resolvegrid_api/eval_retrieval.py`'s
functions here operate purely on already-resolved `Chunk.id` ints (plain
`int`, `list[int]`, `frozenset[int]`) -- never on `(Document.title,
Chunk.ordinal)` pairs or a real `Session`. Resolving a golden case's
hand-labeled `(title, ordinal)` pairs to real `Chunk.id`s requires a DB
query (`resolvegrid_api.eval_retrieval.resolve_relevant_chunk_ids`), which
stays `apps/api`'s job; callers there resolve first, then pass the
resolved id sets in here via `RetrievalCase`.

Metric formulas (see `apps/api/tests/test_eval_retrieval.py` and this
package's own `tests/test_retrieval_metrics.py` for hand-computed unit
tests against small fixed examples)
------------------------------------------------------------------------
- recall@k    = |relevant chunks in top-k| / |relevant chunks|
- precision@k = |relevant chunks in top-k| / k   (the standard IR
  definition -- divides by k itself, not by however many results were
  actually returned, so a query that returns fewer than k results is
  correctly penalized rather than getting an inflated score from a
  smaller denominator)
- MRR         = 1 / (rank of the first relevant chunk in the full ranked
  list, not capped to k) -- 0.0 if no relevant chunk appears anywhere in
  the ranked list at all
- nDCG@k      = DCG@k / IDCG@k, with binary relevance (rel=1 if the chunk
  at that rank is in the labeled relevant set, else 0), DCG@k =
  sum_{i=1..k} rel_i / log2(i+1), and IDCG@k computed from the ideal
  ranking (min(|relevant|, k) relevant chunks placed first)

A case with an EMPTY relevant set (a "no good match" case, or an
authz-zero-result case) has no well-defined recall/precision/MRR/nDCG --
those are `None` for that case, per `eval_retrieval.py`'s original
convention (excluded from the averages computed over "answerable" cases,
so a correct "no good match" is never penalized as a retrieval failure).

Distractor / leakage checks (ported from `evaluate_case`'s per-case
computation, faithfully -- see that function's original inline logic)
------------------------------------------------------------------------
`grade_retrieval_case` records:
- `leaked_chunk_ids`: `case.must_not_appear_ids` chunks that appear
  anywhere in `case.raw_retrieved_ids` (the raw, pre-fusion vector/lexical
  search result ids) -- a hard authz-leakage check, unaffected by
  ranking/reranking/dedup (all of those operate only on chunks that
  already passed the authz-filtered SQL query).
- `distractor_beats_best_relevant`/`distractor_margin`: only computed when
  both `case.distractor_ids` and `case.relevant_ids` are non-empty. `True`
  if any distractor chunk's rank in `ranked_chunk_ids` is better
  (numerically lower) than the best-ranked relevant chunk's rank (or, if
  no relevant chunk appears at all, `True` iff any distractor chunk
  appears). `distractor_margin` is the signed rank gap
  (`best_distractor_rank - best_relevant_rank`, 0-indexed positions) when
  both a distractor and a relevant chunk appear somewhere in
  `ranked_chunk_ids`, else `None`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "RetrievalCase",
    "RetrievalGradeResult",
    "grade_retrieval_case",
]


def recall_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
    if not relevant_ids:
        return None
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
    if not relevant_ids:
        return None
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: list[int], relevant_ids: frozenset[int]) -> float | None:
    if not relevant_ids:
        return None
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: list[int], relevant_ids: frozenset[int], k: int) -> float | None:
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


@dataclass(frozen=True)
class RetrievalCase:
    """The already-resolved (real `Chunk.id`) inputs `grade_retrieval_case`
    needs for one retrieval-dimension case.

    Mirrors `resolvegrid_api.eval_retrieval.GoldenCase`'s
    `relevant`/`distractor`/`must_not_appear` fields, except these are
    resolved `Chunk.id` sets, not `(title, ordinal)` pairs -- see module
    docstring's dependency-direction note for why the resolution itself
    stays in `apps/api`.
    """

    relevant_ids: frozenset[int]
    distractor_ids: frozenset[int] = field(default_factory=frozenset)
    must_not_appear_ids: frozenset[int] = field(default_factory=frozenset)
    # The raw, pre-fusion union of vector_search/lexical_search result ids
    # -- what the leakage check runs against (see module docstring).
    raw_retrieved_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RetrievalGradeResult:
    """Per-case retrieval grade: the four IR metrics plus the leakage and
    distractor-beats-relevant checks, ported faithfully from
    `eval_retrieval.py`'s `evaluate_case` (see module docstring).
    """

    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    leaked_chunk_ids: frozenset[int]
    distractor_beats_best_relevant: bool | None
    # Signed rank gap between the best distractor and best relevant chunk
    # (0-indexed positions in ranked_chunk_ids). `None` unless both a
    # distractor and a relevant chunk appear somewhere in ranked_chunk_ids.
    distractor_margin: int | None = None


def grade_retrieval_case(
    case: RetrievalCase, ranked_chunk_ids: list[int], *, k: int
) -> RetrievalGradeResult:
    """Grade one retrieval case's already-ranked chunk-id list against its
    hand-labeled relevant/distractor/must_not_appear id sets.

    Ports `eval_retrieval.py`'s `evaluate_case` per-case computation
    verbatim (recall/precision/MRR/nDCG via the formulas above, the
    must_not_appear leakage check, and the distractor-beats-relevant /
    distractor_margin check) into a standalone, independently testable,
    pure function -- see module docstring for each check's exact
    semantics.
    """
    relevant_ids = case.relevant_ids
    distractor_ids = case.distractor_ids

    distractor_beats_best_relevant: bool | None = None
    distractor_margin: int | None = None
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
        if best_relevant_rank is not None and best_distractor_rank is not None:
            distractor_margin = best_distractor_rank - best_relevant_rank

    leaked_chunk_ids = frozenset(case.must_not_appear_ids & case.raw_retrieved_ids)

    return RetrievalGradeResult(
        recall_at_k=recall_at_k(ranked_chunk_ids, relevant_ids, k),
        precision_at_k=precision_at_k(ranked_chunk_ids, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(ranked_chunk_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(ranked_chunk_ids, relevant_ids, k),
        leaked_chunk_ids=leaked_chunk_ids,
        distractor_beats_best_relevant=distractor_beats_best_relevant,
        distractor_margin=distractor_margin,
    )
