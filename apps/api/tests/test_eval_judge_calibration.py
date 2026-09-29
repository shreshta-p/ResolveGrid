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

from sqlalchemy import func, select

from resolvegrid_api.eval_judge import run_real_calibration
from resolvegrid_api.models import ModelCall
from resolvegrid_evaluation.judge import AgreementReport
from resolvegrid_telemetry import init_tracing

_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3] / "eval" / "golden" / "judge_calibration_v1.jsonl"
)


def test_calibration_file_exists_with_expected_shape():
    # Fast, non-network sanity check that fails loudly with a clear message
    # if the fixture file ever moves/gets renamed, rather than the real
    # test below failing obscurely inside run_real_calibration.
    assert _CALIBRATION_PATH.exists(), f"calibration file not found at {_CALIBRATION_PATH}"


def test_run_real_calibration_end_to_end_against_real_ollama(raw_db_session):
    # `eval_judge.py`'s `tracer = trace.get_tracer(__name__)` binds to
    # whatever global TracerProvider is registered by the time a span is
    # actually started (see that module's comment) -- if no other test in
    # this pytest session/process has triggered main.py's lifespan
    # (resolvegrid_telemetry.init_tracing) yet, that's a no-op
    # ProxyTracerProvider, and every span's trace_id formats to the
    # reserved all-zero INVALID sentinel. init_tracing() is idempotent
    # (OTel only allows the global provider to be set once per process,
    # per test_model_call_logging.py's identical fix/comment), so calling
    # it here guarantees a REAL, non-zero trace id below regardless of
    # what else has or hasn't run yet in this session.
    init_tracing("test-eval-judge-calibration")

    # Phase 11 Task 3: watermark ModelCall.id BEFORE this run so the
    # assertions below can identify exactly which rows THIS calibration run
    # wrote -- ModelCall is an accumulating log table (see
    # test_ticket_summarize.py/test_chat_api.py's identical convention), no
    # cleanup needed/expected afterward.
    before_max_model_call_id = raw_db_session.scalar(select(func.max(ModelCall.id))) or 0

    report = run_real_calibration(_CALIBRATION_PATH)

    assert isinstance(report, AgreementReport)

    # Core exit criterion: every real judge completion `eval_judge.py`'s
    # `real_complete_fn` closure makes (one per judge dimension per
    # calibration case) now writes a real `ModelCall` row via
    # `model_call_logging.log_completion`, purpose="eval.judge" -- closing
    # the gap the phase's audit found (judge completions were real LLM
    # calls that were never logged at all).
    judge_model_calls = raw_db_session.scalars(
        select(ModelCall)
        .where(ModelCall.id > before_max_model_call_id, ModelCall.purpose == "eval.judge")
        .order_by(ModelCall.id)
    ).all()
    assert len(judge_model_calls) > 0, "expected at least one real ModelCall row for purpose='eval.judge'"
    # Every REAL attempt -- success or error-shaped -- must carry a real,
    # well-formed trace_id: `log_completion` sets this unconditionally
    # before either outcome is known (see model_call_logging.py), so this
    # holds regardless of whether the underlying completion succeeded.
    for call in judge_model_calls:
        assert call.trace_id is not None
        assert len(call.trace_id) == 32
        assert call.trace_id != "0" * 32

    # NOT `assert call.status == "success"` for every row (this test's
    # original assertion, which a real CI run genuinely failed on
    # 2026-09-29 -- see docs/DECISION_LOG.md's entry for that date for the
    # full investigation). `_attempt_judge_call` (services/evaluation's
    # judge.py) is explicitly, deliberately designed to tolerate a real
    # completion failure: it catches ANY exception from `complete_fn`,
    # returns None, and `judge_response` retries once with an identical
    # prompt before giving up -- and `log_completion` (Task 1) writes a
    # real, honest error-shaped `ModelCall` row for that first failed
    # attempt BEFORE re-raising, exactly matching this project's own
    # already-established precedent for a genuine cold-start/timeout
    # failure (see `summarize_ticket`'s identical "first cold attempt
    # times out at 60s, records a real status='error' ModelCall row, not a
    # bug" behavior, documented in docs/PROGRESS.md's Phase 4 row). A
    # transient real failure on ONE attempt, gracefully retried by the
    # judge module's own by-design resilience, is therefore expected,
    # correct behavior under real hardware variance (confirmed root cause:
    # `qwen3:14b` generation on GitHub's CPU-only hosted runners is far
    # slower/more variable than local GPU hardware -- the same
    # already-documented characteristic `docs/DECISION_LOG.md`'s
    # 2026-09-11 entry found for `test_eval_worker.py`), not a code
    # regression -- requiring zero-ever-errors here would make this test
    # flaky against real, honest hardware variance for no real safety
    # benefit, the same reasoning this test's own module docstring already
    # applies to NOT hard-gating on a specific agreement percentage.
    #
    # What's still required, to keep this a REAL, meaningful check and not
    # a rubber stamp: at least one real success (a total wipeout -- every
    # single attempt failing -- would indicate a genuine systemic problem,
    # e.g. a broken gateway wiring, not hardware variance) and a real
    # majority success rate (a systemic bug would fail most/every call,
    # not just an occasional cold-start outlier).
    success_calls = [c for c in judge_model_calls if c.status == "success"]
    error_calls = [c for c in judge_model_calls if c.status != "success"]
    assert len(success_calls) > 0, "expected at least one real successful eval.judge ModelCall row"
    assert len(success_calls) >= len(error_calls), (
        f"expected a real majority of eval.judge completions to succeed, got "
        f"{len(success_calls)} success / {len(error_calls)} error "
        f"(error rows: {[(c.id, c.error_message) for c in error_calls]})"
    )

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
