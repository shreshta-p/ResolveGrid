"""Tests for `resolvegrid_evaluation.graders` (Phase 10 Task 2).

Covers, per grader: at least one passing case, at least one failing case,
and (for at least two graders) the "N/A dimension passes trivially"
behavior documented in graders.py's module docstring. Also includes the
explicit grader-determinism test the plan calls out as its own exit
criterion: calling every grader twice with byte-identical inputs must
produce byte-identical `GradeResult` output.
"""

import pytest

from resolvegrid_evaluation.graders import (
    GradeResult,
    grade_approval_compliance,
    grade_citation_correctness,
    grade_forbidden_actions,
    grade_schema_validity,
    grade_tool_argument_accuracy,
)
from resolvegrid_evaluation.schema import EvalCase


def _case(**overrides) -> EvalCase:
    base = dict(
        case_id="test.case.001",
        dataset_version="v2",
        dimension="chat",
        provenance="hand_written",
        human_review_status="approved",
    )
    base.update(overrides)
    return EvalCase(**base)


# --------------------------------------------------------------------
# grade_schema_validity
# --------------------------------------------------------------------


def test_grade_schema_validity_passes_valid_tool_call():
    case = _case(dimension="tool", expected_tool_name="grant_vpn_access")
    actual = {
        "tool_name": "grant_vpn_access",
        "tool_params": {"employee_id": 42, "justification": "new hire"},
    }
    result = grade_schema_validity(case, actual)
    assert result.passed is True


def test_grade_schema_validity_fails_unknown_tool_name():
    case = _case(dimension="tool")
    actual = {"tool_name": "not_a_real_tool", "tool_params": {}}
    result = grade_schema_validity(case, actual)
    assert result.passed is False
    assert result.details["tool_name"] == "not_a_real_tool"


def test_grade_schema_validity_fails_params_missing_required_key():
    case = _case(dimension="tool")
    actual = {
        "tool_name": "grant_vpn_access",
        # missing required "justification"
        "tool_params": {"employee_id": 42},
    }
    result = grade_schema_validity(case, actual)
    assert result.passed is False
    assert result.details["reason"] == "tool_params failed schema validation"


def test_grade_schema_validity_fails_params_extra_key_additional_properties_false():
    case = _case(dimension="tool")
    actual = {
        "tool_name": "lookup_employee_entitlements",
        "tool_params": {"employee_id": 1, "unexpected_extra": "nope"},
    }
    result = grade_schema_validity(case, actual)
    assert result.passed is False


def test_grade_schema_validity_chat_dimension_passes_with_expected_keys_present():
    case = _case(dimension="chat", expected_intent="greeting")
    actual = {"intent": "greeting", "risk_level": "low"}
    result = grade_schema_validity(case, actual)
    assert result.passed is True


def test_grade_schema_validity_chat_dimension_fails_missing_keys():
    case = _case(dimension="chat", expected_intent="greeting")
    actual = {"intent": "greeting"}  # missing risk_level
    result = grade_schema_validity(case, actual)
    assert result.passed is False
    assert "risk_level" in result.details["missing_keys"]


def test_grade_schema_validity_na_for_chat_case_with_no_expected_intent():
    # N/A dimension passes trivially: a chat-dimension case that never set
    # expected_intent has nothing structured to check.
    case = _case(dimension="chat")
    result = grade_schema_validity(case, {})
    assert result.passed is True


def test_grade_schema_validity_na_for_retrieval_dimension():
    # N/A dimension passes trivially: retrieval-dimension cases have no
    # structured shape this grader checks at all.
    case = _case(dimension="retrieval")
    result = grade_schema_validity(case, {"anything": "goes"})
    assert result.passed is True


# --------------------------------------------------------------------
# grade_citation_correctness
# --------------------------------------------------------------------


def test_grade_citation_correctness_passes_when_all_citations_retrieved():
    case = _case(dimension="retrieval")
    actual = {"citations": ["c1", "c2"], "retrieved_chunk_ids": ["c1", "c2", "c3"]}
    result = grade_citation_correctness(case, actual)
    assert result.passed is True


def test_grade_citation_correctness_fails_on_fabricated_citation():
    case = _case(dimension="retrieval")
    actual = {"citations": ["c1", "c99"], "retrieved_chunk_ids": ["c1", "c2", "c3"]}
    result = grade_citation_correctness(case, actual)
    assert result.passed is False
    assert result.details["fabricated_ids"] == ["c99"]


def test_grade_citation_correctness_na_when_no_citations_present():
    # N/A dimension passes trivially: nothing was cited, so nothing can be
    # fabricated.
    case = _case(dimension="chat")
    result = grade_citation_correctness(case, {})
    assert result.passed is True


# --------------------------------------------------------------------
# grade_tool_argument_accuracy
# --------------------------------------------------------------------


def test_grade_tool_argument_accuracy_passes_exact_match():
    case = _case(
        dimension="tool",
        expected_tool_name="grant_vpn_access",
        expected_tool_params={"employee_id": 42, "justification": "new hire"},
    )
    actual = {
        "tool_name": "grant_vpn_access",
        "tool_params": {"employee_id": 42, "justification": "new hire"},
    }
    result = grade_tool_argument_accuracy(case, actual)
    assert result.passed is True


def test_grade_tool_argument_accuracy_fails_wrong_tool_name():
    case = _case(dimension="tool", expected_tool_name="grant_vpn_access")
    actual = {"tool_name": "lookup_employee_entitlements", "tool_params": {}}
    result = grade_tool_argument_accuracy(case, actual)
    assert result.passed is False
    assert result.details["reason"] == "tool_name mismatch"


def test_grade_tool_argument_accuracy_fails_param_value_mismatch():
    case = _case(
        dimension="tool",
        expected_tool_name="grant_vpn_access",
        expected_tool_params={"employee_id": 42, "justification": "new hire"},
    )
    actual = {
        "tool_name": "grant_vpn_access",
        "tool_params": {"employee_id": 999, "justification": "new hire"},
    }
    result = grade_tool_argument_accuracy(case, actual)
    assert result.passed is False
    assert "employee_id" in result.details["mismatched_params"]


def test_grade_tool_argument_accuracy_fails_missing_param():
    case = _case(
        dimension="tool",
        expected_tool_name="grant_vpn_access",
        expected_tool_params={"employee_id": 42, "justification": "new hire"},
    )
    actual = {"tool_name": "grant_vpn_access", "tool_params": {"employee_id": 42}}
    result = grade_tool_argument_accuracy(case, actual)
    assert result.passed is False
    assert result.details["mismatched_params"]["justification"]["missing"] is True


def test_grade_tool_argument_accuracy_ignores_extra_actual_params():
    case = _case(
        dimension="tool",
        expected_tool_name="grant_vpn_access",
        expected_tool_params={"employee_id": 42},
    )
    actual = {
        "tool_name": "grant_vpn_access",
        "tool_params": {"employee_id": 42, "justification": "extra, unchecked"},
    }
    result = grade_tool_argument_accuracy(case, actual)
    assert result.passed is True


def test_grade_tool_argument_accuracy_na_for_chat_case_with_no_expected_tool_name():
    # N/A dimension passes trivially: a chat-dimension case never set
    # expected_tool_name, so this grader doesn't apply.
    case = _case(dimension="chat")
    result = grade_tool_argument_accuracy(case, {"tool_name": "grant_vpn_access"})
    assert result.passed is True


# --------------------------------------------------------------------
# grade_approval_compliance
# --------------------------------------------------------------------


def test_grade_approval_compliance_passes_when_both_match():
    case = _case(
        dimension="approval",
        expected_approval_decision="approved",
        expected_final_status="approved",
    )
    actual = {"approval_decision": "approved", "final_status": "approved"}
    result = grade_approval_compliance(case, actual)
    assert result.passed is True


def test_grade_approval_compliance_fails_when_final_status_mismatches():
    case = _case(
        dimension="approval",
        expected_approval_decision="approved",
        expected_final_status="approved",
    )
    actual = {"approval_decision": "approved", "final_status": "pending"}
    result = grade_approval_compliance(case, actual)
    assert result.passed is False
    assert result.details["checks"]["final_status"]["matched"] is False


def test_grade_approval_compliance_na_for_case_with_neither_field_set():
    # N/A dimension passes trivially: chat-dimension case, no approval
    # fields ever set.
    case = _case(dimension="chat")
    result = grade_approval_compliance(case, {"approval_decision": "rejected"})
    assert result.passed is True


# --------------------------------------------------------------------
# grade_forbidden_actions
# --------------------------------------------------------------------


def test_grade_forbidden_actions_passes_when_none_taken():
    case = _case(dimension="adversarial", forbidden_actions=["grant_vpn_access"])
    actual = {"actions_taken": ["lookup_employee_entitlements"]}
    result = grade_forbidden_actions(case, actual)
    assert result.passed is True


def test_grade_forbidden_actions_fails_zero_tolerance_on_any_match():
    case = _case(dimension="adversarial", forbidden_actions=["grant_vpn_access"])
    actual = {"actions_taken": ["lookup_employee_entitlements", "grant_vpn_access"]}
    result = grade_forbidden_actions(case, actual)
    assert result.passed is False
    assert result.details["violated_actions"] == ["grant_vpn_access"]


def test_grade_forbidden_actions_na_when_unset():
    case = _case(dimension="chat")
    result = grade_forbidden_actions(case, {"actions_taken": ["anything"]})
    assert result.passed is True


# --------------------------------------------------------------------
# Grader-determinism test (explicit Phase 10 exit-criterion item):
# calling each grader twice with byte-identical inputs must produce
# byte-identical GradeResult output, every time.
# --------------------------------------------------------------------


def _assert_deterministic(grader, case: EvalCase, actual_result: dict) -> None:
    first = grader(case, actual_result)
    second = grader(case, actual_result)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    "grader,case,actual_result",
    [
        (
            grade_schema_validity,
            _case(dimension="tool"),
            {
                "tool_name": "grant_vpn_access",
                "tool_params": {"employee_id": 42, "justification": "new hire"},
            },
        ),
        (
            grade_schema_validity,
            _case(dimension="tool"),
            {"tool_name": "unknown_tool", "tool_params": {}},
        ),
        (
            grade_citation_correctness,
            _case(dimension="retrieval"),
            {
                "citations": ["c3", "c1", "c2"],
                "retrieved_chunk_ids": ["c1", "c2"],
            },
        ),
        (
            grade_tool_argument_accuracy,
            _case(
                dimension="tool",
                expected_tool_name="grant_vpn_access",
                expected_tool_params={"employee_id": 42, "justification": "new hire"},
            ),
            {
                "tool_name": "grant_vpn_access",
                "tool_params": {"employee_id": 999},
            },
        ),
        (
            grade_approval_compliance,
            _case(
                dimension="approval",
                expected_approval_decision="approved",
                expected_final_status="approved",
            ),
            {"approval_decision": "rejected", "final_status": "rejected"},
        ),
        (
            grade_forbidden_actions,
            _case(
                dimension="adversarial",
                forbidden_actions=["grant_vpn_access", "delete_ticket", "reset_password"],
            ),
            {
                "actions_taken": [
                    "reset_password",
                    "grant_vpn_access",
                    "lookup_employee_entitlements",
                ]
            },
        ),
    ],
)
def test_grader_determinism(grader, case, actual_result):
    _assert_deterministic(grader, case, actual_result)


def test_grade_result_is_frozen():
    result = GradeResult(passed=True, score=None, details={})
    with pytest.raises(Exception):
        result.passed = False  # type: ignore[misc]
