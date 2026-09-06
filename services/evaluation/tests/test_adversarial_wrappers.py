"""Unit tests for `resolvegrid_evaluation.adversarial_wrappers` (Phase 10
Task 6). No DB/HTTP -- this module only ever turns an already-computed
plain dict into a `GradeResult`, so these tests just prove that mapping is
correct and deterministic, mirroring `test_graders.py`'s own determinism
-test convention.
"""

from resolvegrid_evaluation.adversarial_wrappers import grade_reused_pytest_assertion
from resolvegrid_evaluation.schema import EvalCase

_CASE = EvalCase(
    case_id="adversarial.simulated_provider_outage.001",
    dataset_version="v1",
    dimension="adversarial",
    provenance="hand_written",
    human_review_status="approved",
)


def test_grade_reused_pytest_assertion_passes_when_actual_result_reports_passed_true():
    result = grade_reused_pytest_assertion(_CASE, {"passed": True, "fallback_occurred": True})
    assert result.passed is True
    assert result.details == {"fallback_occurred": True}


def test_grade_reused_pytest_assertion_fails_when_actual_result_reports_passed_false():
    result = grade_reused_pytest_assertion(_CASE, {"passed": False, "reason": "grant leaked through"})
    assert result.passed is False
    assert result.details == {"reason": "grant leaked through"}


def test_grade_reused_pytest_assertion_treats_missing_passed_key_as_failure():
    # A caller that forgot to set "passed" at all must never be silently
    # treated as a pass -- fail closed.
    result = grade_reused_pytest_assertion(_CASE, {"some_detail": "x"})
    assert result.passed is False


def test_grade_reused_pytest_assertion_is_deterministic():
    actual_result = {"passed": True, "http_status": 200, "serving_model_group": "cloud-fallback"}
    first = grade_reused_pytest_assertion(_CASE, actual_result)
    second = grade_reused_pytest_assertion(_CASE, actual_result)
    assert first == second
