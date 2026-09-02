"""Tests for `resolvegrid_api.tool_execution` (Phase 9 Task 4).

Pure Python, no DB -- `available_tools_for_principal`/`select_tool`/
`validate_tool_schema` are deliberately I/O-free (see
`resolvegrid_api.tool_execution`'s module docstring). Both
`TOOL_REGISTRY` entries (`lookup_employee_entitlements`, `grant_vpn_access`)
require the "analyst" role and no entitlement as of Phase 9 Task 2 -- this
suite asserts that against the real registry rather than assuming it, since
the registry could change.
"""

import pytest
from resolvegrid_authz import Principal, RoleGrant
from resolvegrid_contracts.tools import TOOL_REGISTRY
from resolvegrid_api.tool_execution import (
    ToolNotAllowedError,
    ToolValidationError,
    available_tools_for_principal,
    select_tool,
    validate_tool_schema,
)

LOOKUP_TOOL = TOOL_REGISTRY["lookup_employee_entitlements"]
GRANT_TOOL = TOOL_REGISTRY["grant_vpn_access"]


def test_registry_tools_both_require_analyst_role_with_no_entitlement():
    # Pins the assumption the rest of this suite relies on -- if Task 2's
    # registry ever changes required_role/required_entitlement for either
    # tool, this test fails loudly instead of the other tests silently
    # testing the wrong thing.
    assert LOOKUP_TOOL.required_role == "analyst"
    assert LOOKUP_TOOL.required_entitlement is None
    assert GRANT_TOOL.required_role == "analyst"
    assert GRANT_TOOL.required_entitlement is None


def test_analyst_sees_both_registered_tools():
    principal = Principal(
        employee_id=1, roles=(RoleGrant(role="analyst", scope="department", scope_id=5),)
    )
    available = available_tools_for_principal(principal)
    names = {tool.name for tool in available}
    assert names == {"lookup_employee_entitlements", "grant_vpn_access"}


def test_principal_with_no_role_grants_sees_no_tools():
    principal = Principal(employee_id=2, roles=())
    available = available_tools_for_principal(principal)
    assert available == []


def test_approver_only_principal_does_not_see_analyst_tools():
    # "approver" != "analyst" -- both registered tools require "analyst",
    # so an approver-only principal (no admin, no analyst grant) must not
    # see either tool.
    principal = Principal(
        employee_id=3, roles=(RoleGrant(role="approver", scope="department", scope_id=9),)
    )
    available = available_tools_for_principal(principal)
    names = {tool.name for tool in available}
    assert "grant_vpn_access" not in names
    assert "lookup_employee_entitlements" not in names
    assert available == []


def test_global_admin_sees_both_tools_regardless_of_specific_role_match():
    # Mirrors authorize()'s "admin dominates" precedent, via
    # principal_has_role()'s same admin-dominates check.
    principal = Principal(employee_id=4, roles=(RoleGrant(role="admin", scope="global"),))
    available = available_tools_for_principal(principal)
    names = {tool.name for tool in available}
    assert names == {"lookup_employee_entitlements", "grant_vpn_access"}


def test_select_tool_returns_matching_contract():
    principal = Principal(
        employee_id=5, roles=(RoleGrant(role="analyst", scope="department", scope_id=5),)
    )
    available = available_tools_for_principal(principal)
    selected = select_tool("lookup_employee_entitlements", available)
    assert selected is LOOKUP_TOOL


def test_select_tool_raises_when_filtered_out_of_available():
    principal = Principal(employee_id=6, roles=())
    available = available_tools_for_principal(principal)
    assert available == []
    with pytest.raises(ToolNotAllowedError):
        select_tool("grant_vpn_access", available)


def test_select_tool_raises_when_tool_name_does_not_exist_at_all():
    principal = Principal(
        employee_id=7, roles=(RoleGrant(role="admin", scope="global"),)
    )
    available = available_tools_for_principal(principal)
    with pytest.raises(ToolNotAllowedError):
        select_tool("delete_entire_database", available)


def test_select_tool_raises_identical_exception_shape_for_missing_vs_filtered():
    # The whole point of ToolNotAllowedError's design: "doesn't exist" and
    # "exists but not permitted" must be indistinguishable to the caller.
    no_role_principal = Principal(employee_id=8, roles=())
    available = available_tools_for_principal(no_role_principal)

    with pytest.raises(ToolNotAllowedError) as filtered_exc_info:
        select_tool("grant_vpn_access", available)  # exists in TOOL_REGISTRY, filtered out

    with pytest.raises(ToolNotAllowedError) as missing_exc_info:
        select_tool("no_such_tool", available)  # never existed in TOOL_REGISTRY

    assert type(filtered_exc_info.value) is type(missing_exc_info.value)
    # Same message template ("tool not allowed: <name>") for both -- the
    # tool name itself is echoed back either way (that's the value the
    # caller supplied), but nothing else differs between the two cases.
    assert str(filtered_exc_info.value) == "tool not allowed: grant_vpn_access"
    assert str(missing_exc_info.value) == "tool not allowed: no_such_tool"


def test_validate_tool_schema_accepts_valid_params():
    result = validate_tool_schema(LOOKUP_TOOL, {"employee_id": 42})
    assert result == {"employee_id": 42}


def test_validate_tool_schema_rejects_missing_required_param():
    with pytest.raises(ToolValidationError):
        validate_tool_schema(GRANT_TOOL, {"employee_id": 42})  # missing "justification"


def test_validate_tool_schema_rejects_wrong_type():
    with pytest.raises(ToolValidationError):
        validate_tool_schema(LOOKUP_TOOL, {"employee_id": "forty-two"})


def test_validate_tool_schema_rejects_unexpected_extra_param():
    with pytest.raises(ToolValidationError):
        validate_tool_schema(
            LOOKUP_TOOL, {"employee_id": 42, "unexpected_extra_field": "surprise"}
        )


def test_validate_tool_schema_accepts_valid_grant_vpn_params():
    params = {"employee_id": 42, "justification": "onboarding: remote contractor"}
    result = validate_tool_schema(GRANT_TOOL, params)
    assert result == params
