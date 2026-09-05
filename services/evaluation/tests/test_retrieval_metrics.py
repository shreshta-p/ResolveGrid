"""Tests for `resolvegrid_evaluation.retrieval_metrics` (Phase 10 Task 3).

The pure recall@k/precision@k/MRR/nDCG@k formula hand-computed unit tests
live in `apps/api/tests/test_eval_retrieval.py` (imported there directly
from this module) -- kept in one place, not duplicated here under a
second set of names, since a formula change that only got updated in one
copy wouldn't fail loudly (same underlying function object, different
test file). See that file's own copies for the hand-computed
recall_at_k/precision_at_k/reciprocal_rank/ndcg_at_k assertions.

`test_grade_retrieval_case_*` below are this module's own tests: they
exercise `grade_retrieval_case`'s leakage-detection and
distractor-beats-relevant detection, ported out of `eval_retrieval.py`'s
`evaluate_case` per-case computation (see retrieval_metrics.py's module
docstring), plus a check that `grade_retrieval_case` computes its four
metrics via this module's own public formula functions (not a
reimplementation of them).
"""

from resolvegrid_evaluation.retrieval_metrics import (
    RetrievalCase,
    grade_retrieval_case,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

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
