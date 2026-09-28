"""Real wiring for `resolvegrid_evaluation.judge`'s model-judge grader
(Phase 10 Task 4).

This is the ONLY module in the whole codebase allowed to import both
`resolvegrid_evaluation.judge` and the real LLM gateway
(`resolvegrid_api.llm_gateway`) together -- see `judge.py`'s module
docstring for the dependency-direction rule this enforces
(`services/evaluation` must never import `apps/api`/the real gateway
directly; it only ever takes a `complete_fn`-shaped callable).

Phase 11 Task 3: `real_complete_fn` is now built via
`resolvegrid_api.model_call_logging.make_logging_complete_fn` (the same
shared factory `main.py`'s/`eval_worker.py`'s real chat-graph closures use)
instead of the original bare `lambda prompt: llm_gateway.complete(prompt)
.text` -- every real judge completion this closure makes now also writes a
real `ModelCall` row (`purpose="eval.judge"`), closing the gap the phase's
audit found: judge completions were real LLM calls that were never logged
at all. `make_logging_complete_fn` opens its own short-lived session per
call via `resolvegrid_api.db.session_factory()`, exactly like
`agent_retrieval.py`'s established `retrieve_for_agent` pattern (this
module is built at import time, same as those, so it has no per-request
session to close over either -- see that factory's own docstring for the
full rationale).
"""

from pathlib import Path

from opentelemetry import trace
from resolvegrid_evaluation.judge import (
    AgreementReport,
    CalibrationCase,
    JudgeVerdict,
    calculate_agreement,
    judge_response,
    load_calibration_cases,
    run_calibration,
)
from resolvegrid_evaluation.schema import EvalCase

from resolvegrid_api.db import session_factory
from resolvegrid_api.model_call_logging import make_logging_complete_fn

__all__ = [
    "real_complete_fn",
    "judge_response_real",
    "run_real_calibration",
    "AgreementReport",
    "CalibrationCase",
    "JudgeVerdict",
]

# The global tracer provider is already configured once at app startup by
# main.py's lifespan hook (resolvegrid_telemetry.init_tracing) -- this module
# must NOT call init_tracing again, just bind a tracer to whatever provider is
# globally registered by the time a span is actually started (same pattern as
# routers/chat.py/routers/tickets.py/eval_worker.py).
tracer = trace.get_tracer(__name__)

# The real `CompleteFn` closure: a real completion through the real
# LiteLLM/Ollama-backed gateway, logged as a real `ModelCall` row (see
# module docstring). Built once at import time -- matches every other real
# `CompleteFn` closure in this codebase's "built once, not per-call" shape.
real_complete_fn = make_logging_complete_fn(session_factory, tracer, purpose="eval.judge")


def judge_response_real(case: EvalCase, actual_result: dict) -> list[JudgeVerdict]:
    """Grade `actual_result` against `case.judge_dimensions` using the real
    LLM gateway. Thin convenience wrapper around
    `resolvegrid_evaluation.judge.judge_response` bound to
    `real_complete_fn` -- the real entry point Task 7's batch runner (a
    later task) is expected to call for any case with `judge_dimensions`
    set.
    """
    return judge_response(real_complete_fn, case, actual_result)


def run_real_calibration(path: Path) -> AgreementReport:
    """Run the full judge-calibration harness against the real LLM gateway.

    Loads `eval/golden/judge_calibration_v1.jsonl` (or any file in that
    same shape -- see `load_calibration_cases`'s docstring) via
    `resolvegrid_evaluation.judge.load_calibration_cases`, then delegates
    the actual grading + agreement scoring to that package's
    `run_calibration`, bound to `real_complete_fn`. This is the function
    `apps/api/tests/test_eval_judge_calibration.py` calls end to end
    against the real running Ollama.
    """
    calibration_cases = load_calibration_cases(path)
    return run_calibration(real_complete_fn, calibration_cases)
