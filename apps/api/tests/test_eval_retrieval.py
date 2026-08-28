"""Tests for `resolvegrid_api.eval_retrieval` (Phase 7 Task 9: retrieval
evaluation baseline).

`test_golden_set_metrics_clear_baseline_regression_guard` is a real,
no-mocking integration test: it ingests the actual seed corpus (real
chunker, real Ollama embedder, real DB writes -- matching
`test_ingestion_worker.py`'s established Phase 7 precedent) via the
transactional `db_session` fixture, then runs the full golden set from
`eval/golden/phase7_retrieval_v1.jsonl` against it and asserts the
aggregated metrics clear a regression-guard bar. `db_session` rolls back
at the end of the test, so this leaves zero residue in the dev DB (see
`conftest.py`'s fixture docstring) -- no manual cleanup needed, unlike
`test_ingestion_worker.py`'s real-Arq test, which goes through a
genuinely separate committing session.

Threshold provenance (read before changing the numbers below): these are
NOT invented targets. They were measured by running this exact golden
set once (via `python -m resolvegrid_api.eval_retrieval`) against a
freshly-ingested real corpus, at commit `1f43bce`, with
`parser_version=markdown-v1`, `chunking_version=heading-aware-v1`,
`embedding_model=nomic-embed-text`, `embedding_version=v1`:

    recall@5 = 1.0000, precision@5 = 0.2000, MRR = 0.9524, nDCG@5 = 0.9643

(full numbers, including the per-case breakdown, are recorded in
`docs/EXPERIMENT_REGISTRY.md`). Each assertion below is set a bit *below*
that observed number -- enough headroom to absorb normal embedding-call
nondeterminism (Ollama's `nomic-embed-text` is not bitwise-deterministic
across calls) and to not be a tautology, while still catching a real
regression:

- `precision@5`'s ceiling is mechanically 0.2000 on this golden set (every
  answerable case has exactly one hand-labeled relevant chunk, so a
  found-it case always contributes exactly 1/5 to the mean) -- the 0.15
  floor tolerates roughly two cases losing their hit entirely before
  failing, not "some absolute quality bar."
- `recall@5`'s 0.85 floor tolerates about two of the 14 answerable cases
  losing their relevant chunk out of the top-5 before failing.
- `MRR`'s 0.80 and `nDCG@5`'s 0.80 floors tolerate a case or two dropping
  from a rank-1 hit to a lower rank (as already genuinely happens for the
  "Access Scope" case, which currently ranks its correct chunk #3, not
  #1) without failing on that alone, while still catching a real
  ranking-quality regression across multiple cases.
"""

import math

import pytest

from resolvegrid_api.eval_retrieval import (
    DEFAULT_K,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    _reciprocal_rank,
    load_golden_cases,
    run_eval,
)
from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion

# See module docstring: measured once against the real ingested corpus
# (see docs/EXPERIMENT_REGISTRY.md for the exact recorded run: recall@5=
# 1.0000, precision@5=0.2000, MRR=0.9524, nDCG@5=0.9643), then set a bit
# below each observed value as a real regression guard.
_MIN_MEAN_RECALL_AT_K = 0.85
_MIN_MEAN_PRECISION_AT_K = 0.15
_MIN_MEAN_RECIPROCAL_RANK = 0.80
_MIN_MEAN_NDCG_AT_K = 0.80


def test_golden_file_has_a_realistic_number_of_cases_and_expected_shape():
    cases = load_golden_cases()
    assert 12 <= len(cases) <= 20

    answerable = [c for c in cases if c.relevant]
    no_match = [c for c in cases if not c.relevant]
    assert answerable, "golden set must include at least one answerable case"
    assert no_match, "golden set must include at least one no-good-match case"

    distractor_cases = [c for c in cases if c.distractor]
    assert distractor_cases, "golden set must include at least one VPN v1/v2 distractor case"

    authz_scoped_cases = [c for c in cases if not c.authz.unrestricted]
    assert authz_scoped_cases, "golden set must exercise authz-scoped (non-unrestricted) cases"

    leakage_cases = [c for c in cases if c.must_not_appear]
    assert leakage_cases, "golden set must include at least one adversarial authz leakage case"


def test_golden_set_metrics_clear_baseline_regression_guard(db_session):
    run_seed_corpus_ingestion(db_session)
    db_session.flush()

    summary = run_eval(db_session)

    assert summary.k == DEFAULT_K
    assert len(summary.case_results) == len(load_golden_cases())

    # No must_not_appear chunk may ever leak through the authz filter, for
    # any case -- a hard, zero-tolerance check, not a threshold.
    assert summary.any_leakage is False, [
        (c.query, sorted(c.leaked_chunk_ids)) for c in summary.case_results if c.leaked_chunk_ids
    ]

    # The deliberately-superseded VPN v1 policy must never outrank the
    # correct, current v2 answer in any distractor case -- also a hard
    # check, not a threshold: this is exactly the kind of mistake a real
    # KB user would notice and complain about, and the whole point of
    # labeling these cases is to catch it precisely, not just "on average."
    assert summary.any_distractor_beats_relevant is False, [
        (c.query, c.ranked_chunk_ids)
        for c in summary.case_results
        if c.distractor_beats_best_relevant
    ]

    assert summary.mean_recall_at_k >= _MIN_MEAN_RECALL_AT_K, summary.mean_recall_at_k
    assert summary.mean_precision_at_k >= _MIN_MEAN_PRECISION_AT_K, summary.mean_precision_at_k
    assert summary.mean_reciprocal_rank >= _MIN_MEAN_RECIPROCAL_RANK, summary.mean_reciprocal_rank
    assert summary.mean_ndcg_at_k >= _MIN_MEAN_NDCG_AT_K, summary.mean_ndcg_at_k


# ---------------------------------------------------------------------------
# Pure metric-formula unit tests (no DB) -- hand-computed expected values,
# per this task's requirement that the formulas themselves are verified
# precisely, not just plausible-looking on the real corpus.
# ---------------------------------------------------------------------------


def test_recall_at_k_hand_computed():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({20, 40, 99})  # 99 never appears in ranked
    # 2 of 3 relevant chunks (20, 40) are within top-5.
    assert _recall_at_k(ranked, relevant, k=5) == 2 / 3


def test_recall_at_k_respects_k_cutoff():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({50})
    assert _recall_at_k(ranked, relevant, k=2) == 0.0
    assert _recall_at_k(ranked, relevant, k=5) == 1.0


def test_recall_at_k_empty_relevant_is_none():
    assert _recall_at_k([1, 2, 3], frozenset(), k=5) is None


def test_precision_at_k_hand_computed():
    ranked = [10, 20, 30, 40, 50]
    relevant = frozenset({20, 40})
    # 2 hits in top-5, divided by k=5 (not by len(ranked)).
    assert _precision_at_k(ranked, relevant, k=5) == 2 / 5


def test_precision_at_k_divides_by_k_not_by_result_count():
    # Only 2 results returned at all, both relevant, but k=5 -- precision
    # must still be 2/5, not 2/2, per the standard IR definition.
    ranked = [10, 20]
    relevant = frozenset({10, 20})
    assert _precision_at_k(ranked, relevant, k=5) == 2 / 5


def test_reciprocal_rank_hand_computed():
    ranked = [10, 20, 30]
    relevant = frozenset({30})
    assert _reciprocal_rank(ranked, relevant) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    ranked = [10, 20, 30]
    relevant = frozenset({99})
    assert _reciprocal_rank(ranked, relevant) == 0.0


def test_reciprocal_rank_empty_relevant_is_none():
    assert _reciprocal_rank([1, 2, 3], frozenset()) is None


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = [10, 20, 30]
    relevant = frozenset({10, 20})
    # Both relevant chunks placed first -- this IS the ideal ranking.
    assert _ndcg_at_k(ranked, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_hand_computed():
    ranked = [10, 20, 30]
    relevant = frozenset({20})
    # DCG@3 = 1/log2(2+1) (chunk 20 is at rank 2) = 1/log2(3)
    # IDCG@3 = 1/log2(1+1) = 1/log2(2) = 1.0 (ideal: the 1 relevant chunk at rank 1)
    expected = (1 / math.log2(3)) / 1.0
    assert _ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected)


def test_ndcg_at_k_empty_relevant_is_none():
    assert _ndcg_at_k([1, 2, 3], frozenset(), k=5) is None
