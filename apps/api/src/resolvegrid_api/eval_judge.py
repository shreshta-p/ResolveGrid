"""Real wiring for `resolvegrid_evaluation.judge`'s model-judge grader
(Phase 10 Task 4).

This is the ONLY module in the whole codebase allowed to import both
`resolvegrid_evaluation.judge` and the real LLM gateway
(`resolvegrid_api.llm_gateway`) together -- see `judge.py`'s module
docstring for the dependency-direction rule this enforces
(`services/evaluation` must never import `apps/api`/the real gateway
directly; it only ever takes a `complete_fn`-shaped callable).

`real_complete_fn` is built exactly the way `apps/api/src/resolvegrid_api/
main.py`'s app lifespan already builds the chat graph's `CompleteFn`
closure: `complete_fn = lambda prompt: llm_gateway.complete(prompt).text`
(see `main.py`). Reused verbatim here rather than re-derived, since it is
already the established, tested pattern for turning
`llm_gateway.complete`'s richer `CompletionResult` into the narrow
prompt-in/text-out shape every `CompleteFn`-shaped consumer in this
codebase expects.
"""

from pathlib import Path

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

from resolvegrid_api import llm_gateway

__all__ = [
    "real_complete_fn",
    "judge_response_real",
    "run_real_calibration",
    "AgreementReport",
    "CalibrationCase",
    "JudgeVerdict",
]


def real_complete_fn(prompt: str) -> str:
    """The real `CompleteFn` closure: calls the real LiteLLM/Ollama-backed
    gateway and returns just the completion text, matching every other
    `CompleteFn` consumer in this codebase (see module docstring).
    """
    return llm_gateway.complete(prompt).text


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
