"""Tests for `resolvegrid_api.eval_judge` (Phase 10 Task 4).

Runs the judge-calibration harness end to end against the REAL running
LiteLLM/Ollama gateway -- no mocking, matching this project's established
norm for real-integration tests (e.g.
`test_ingestion_worker.py`'s `test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis`
against a real Redis; every `apps/api` test in this suite already assumes
the docker-compose dev stack, including Ollama, is up).

Per the plan doc's explicit instruction, this test does NOT hard-gate on
any specific agreement threshold (e.g. `assert agreement >= 0.80`) -- a
real LLM judge call's actual measured agreement could plausibly land
anywhere, and asserting a specific number here would make this test flaky
against real model nondeterminism for no real safety benefit. It asserts
only that the harness runs correctly end to end and produces a real,
valid-range `AgreementReport` per judge dimension. The actual measured
number is reported (printed here, and expected to be transcribed into
`docs/EXPERIMENT_REGISTRY.md` by Task 9's closing documentation work) --
never gated.
"""

from pathlib import Path

from resolvegrid_api.eval_judge import run_real_calibration
from resolvegrid_evaluation.judge import AgreementReport

_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3] / "eval" / "golden" / "judge_calibration_v1.jsonl"
)


def test_calibration_file_exists_with_expected_shape():
    # Fast, non-network sanity check that fails loudly with a clear message
    # if the fixture file ever moves/gets renamed, rather than the real
    # test below failing obscurely inside run_real_calibration.
    assert _CALIBRATION_PATH.exists(), f"calibration file not found at {_CALIBRATION_PATH}"


def test_run_real_calibration_end_to_end_against_real_ollama():
    report = run_real_calibration(_CALIBRATION_PATH)

    assert isinstance(report, AgreementReport)

    # The calibration file (eval/golden/judge_calibration_v1.jsonl) covers
    # all three judge dimensions defined in
    # resolvegrid_evaluation.schema.JudgeDimension -- every one of them
    # must show up with a real, non-empty comparison count.
    expected_dimensions = {"groundedness", "correctness", "abstention_appropriateness"}
    assert set(report.per_dimension.keys()) == expected_dimensions

    print("\n--- Judge calibration agreement (real Ollama, Phase 10 Task 4) ---")
    for dimension in sorted(report.per_dimension.keys()):
        stats = report.per_dimension[dimension]
        assert stats.total_count > 0
        assert stats.agreement_pct is not None
        assert 0.0 <= stats.agreement_pct <= 1.0
        assert 0 <= stats.agreed_count <= stats.total_count
        print(
            f"{dimension}: agreement={stats.agreement_pct:.3f} "
            f"({stats.agreed_count}/{stats.total_count})"
        )

    assert report.overall_agreement_pct is not None
    assert 0.0 <= report.overall_agreement_pct <= 1.0
    print(f"overall: agreement={report.overall_agreement_pct:.3f}")
    print("--------------------------------------------------------------------")
