"""Tests for `resolvegrid_evaluation.retrieval_metrics` (Phase 10 Task 3).

The hand-computed metric-formula tests below are ported verbatim from
`apps/api/tests/test_eval_retrieval.py`'s pure metric-formula unit tests
(same inputs, same hand-computed expected values) -- only the import
source and function names changed (`_recall_at_k` -> `recall_at_k`, etc.,
per this task's public-API rename). See that file's own copies (still
passing there, unmodified, against the same underlying logic re-exported
under the old private names) for the pre-refactor baseline this consolidation
must not have changed.

`test_grade_retrieval_case_*` below are new: they exercise
`grade_retrieval_case`'s leakage-detection and distractor-beats-relevant
detection, ported out of `eval_retrieval.py`'s `evaluate_case` per-case
computation (see retrieval_metrics.py's module docstring).
"""

import math

import pytest

from resolvegrid_evaluation.retrieval_metrics import (
    RetrievalCase,
    grade_retrieval_case,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Pure metric-formula unit tests (no DB) -- hand-computed expected values,
# ported verbatim from apps/api/tests/test_eval_retrieval.py.
# ---------------------------------------------------------------------------


def test_recall_at_k_hand_computed():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({20, 40, 99})  # 99 never appears in ranked
    # 2 of 3 relevant chunks (20, 40) are within top-5.
    assert recall_at_k(ranked, relevant, k=5) == 2 / 3


def test_recall_at_k_respects_k_cutoff():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({50})
    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert recall_at_k(ranked, relevant, k=5) == 1.0


def test_recall_at_k_empty_relevant_is_none():
    assert recall_at_k([1, 2, 3], frozenset(), k=5) is None


def test_precision_at_k_hand_computed():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({20, 40})
    # 2 hits in top-5, divided by k=5 (not by len(ranked)).
    assert precision_at_k(ranked, relevant, k=5) == 2 / 5


def test_precision_at_k_divides_by_k_not_by_result_count():
    # Only 2 results returned at all, both relevant, but k=5 -- precision
    # must still be 2/5, not 2/2, per the standard IR definition.
    ranked = [10, 20]
    relevant = frozenset({10, 20})
    assert precision_at_k(ranked, relevant, k=5) == 2 / 5


def test_reciprocal_rank_hand_computed():
    ranked = [10, 20, 30]
    relevant = frozenset({30})
    assert reciprocal_rank(ranked, relevant) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    ranked = [10, 20, 30]
    relevant = frozenset({99})
    assert reciprocal_rank(ranked, relevant) == 0.0


def test_reciprocal_rank_empty_relevant_is_none():
    assert reciprocal_rank([1, 2, 3], frozenset()) is None


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = [10, 20, 30]
    relevant = frozenset({10, 20})
    # Both relevant chunks placed first -- this IS the ideal ranking.
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_hand_computed():
    ranked = [10, 20, 30]
    relevant = frozenset({20})
    # DCG@3 = 1/log2(2+1) (chunk 20 is at rank 2) = 1/log2(3)
    # IDCG@3 = 1/log2(1+1) = 1/log2(2) = 1.0 (ideal: the 1 relevant chunk at rank 1)
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected)


def test_ndcg_at_k_empty_relevant_is_none():
    assert ndcg_at_k([1, 2, 3], frozenset(), k=5) is None


# ---------------------------------------------------------------------------
# grade_retrieval_case: leakage detection
# ---------------------------------------------------------------------------


def test_grade_retrieval_case_detects_leakage():
    case = RetrievalCase(
        relevant_ids=frozenset({1}),
        must_not_appear_ids=frozenset({99}),
        raw_retrieved_ids=frozenset({1, 99, 5}),
    )
    grade = grade_retrieval_case(case, ranked_chunk_ids=[1, 5], k=5)
    assert grade.leaked_chunk_ids == frozenset({99})


def test_grade_retrieval_case_no_leakage_when_must_not_appear_absent_from_raw():
    case = RetrievalCase(
        relevant_ids=frozenset({1}),
        must_not_appear_ids=frozenset({99}),
        raw_retrieved_ids=frozenset({1, 5}),
    )
    grade = grade_retrieval_case(case, ranked_chunk_ids=[1, 5], k=5)
    assert grade.leaked_chunk_ids == frozenset()


def test_grade_retrieval_case_leakage_check_ignores_ranked_top_k():
    # A must_not_appear chunk that leaked into the raw retrieved set but
    # didn't make the ranked top-k must still be flagged -- the leakage
    # check runs against raw_retrieved_ids, not the ranked/truncated list.
    case = RetrievalCase(
        relevant_ids=frozenset({1}),
        must_not_appear_ids=frozenset({99}),
        raw_retrieved_ids=frozenset({1, 99}),
    )
    grade = grade_retrieval_case(case, ranked_chunk_ids=[1], k=1)
    assert grade.leaked_chunk_ids == frozenset({99})


# ---------------------------------------------------------------------------
# grade_retrieval_case: distractor-beats-relevant detection
# ---------------------------------------------------------------------------


def test_grade_retrieval_case_distractor_beats_relevant_when_ranked_first():
    case = RetrievalCase(relevant_ids=frozenset({1}), distractor_ids=frozenset({2}))
    grade = grade_retrieval_case(case, ranked_chunk_ids=[2, 1], k=5)
    assert grade.distractor_beats_best_relevant is True
    assert grade.distractor_margin == -1  # distractor at rank 0, relevant at rank 1


def test_grade_retrieval_case_distractor_does_not_beat_relevant_when_ranked_after():
    case = RetrievalCase(relevant_ids=frozenset({1}), distractor_ids=frozenset({2}))
    grade = grade_retrieval_case(case, ranked_chunk_ids=[1, 2], k=5)
    assert grade.distractor_beats_best_relevant is False
    assert grade.distractor_margin == 1  # distractor at rank 1, relevant at rank 0


def test_grade_retrieval_case_distractor_beats_relevant_when_relevant_absent():
    # Relevant chunk never appears at all -- any present distractor
    # trivially "beats" it.
    case = RetrievalCase(relevant_ids=frozenset({1}), distractor_ids=frozenset({2}))
    grade = grade_retrieval_case(case, ranked_chunk_ids=[2, 3], k=5)
    assert grade.distractor_beats_best_relevant is True
    assert grade.distractor_margin is None  # relevant never appeared


def test_grade_retrieval_case_distractor_check_skipped_when_no_distractor_or_relevant():
    case = RetrievalCase(relevant_ids=frozenset())
    grade = grade_retrieval_case(case, ranked_chunk_ids=[1, 2, 3], k=5)
    assert grade.distractor_beats_best_relevant is None
    assert grade.distractor_margin is None


# ---------------------------------------------------------------------------
# grade_retrieval_case: metrics are computed via the same public formulas
# ---------------------------------------------------------------------------


def test_grade_retrieval_case_computes_all_four_metrics():
    case = RetrievalCase(relevant_ids=frozenset({20, 40}))
    ranked = [10, 20, 30, 40, 50]
    grade = grade_retrieval_case(case, ranked, k=5)
    assert grade.recall_at_k == recall_at_k(ranked, case.relevant_ids, 5)
    assert grade.precision_at_k == precision_at_k(ranked, case.relevant_ids, 5)
    assert grade.reciprocal_rank == reciprocal_rank(ranked, case.relevant_ids)
    assert grade.ndcg_at_k == ndcg_at_k(ranked, case.relevant_ids, 5)
