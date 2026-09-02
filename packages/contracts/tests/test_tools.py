import pytest

from resolvegrid_contracts.tools import TOOL_REGISTRY, ToolContract


def test_registry_has_exactly_two_entries():
    assert set(TOOL_REGISTRY.keys()) == {"lookup_employee_entitlements", "grant_vpn_access"}


def test_registry_entries_are_tool_contract_instances():
    for tool in TOOL_REGISTRY.values():
        assert isinstance(tool, ToolContract)


def test_lookup_employee_entitlements_is_read_only_and_pre_approved():
    tool = TOOL_REGISTRY["lookup_employee_entitlements"]
    assert tool.mutating is False
    assert tool.requires_approval is False
    assert tool.required_role == "analyst"
    assert tool.required_entitlement is None


def test_grant_vpn_access_is_mutating_and_requires_approval():
    tool = TOOL_REGISTRY["grant_vpn_access"]
    assert tool.mutating is True
    assert tool.requires_approval is True
    assert tool.required_role == "analyst"


def test_params_schema_has_valid_json_schema_object_shape():
    for tool in TOOL_REGISTRY.values():
        schema = tool.params_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        for required_key in schema["required"]:
            assert required_key in schema["properties"]


def test_lookup_employee_entitlements_params_schema_requires_employee_id():
    schema = TOOL_REGISTRY["lookup_employee_entitlements"].params_schema
    assert schema["required"] == ["employee_id"]
    assert schema["properties"]["employee_id"]["type"] == "integer"


def test_grant_vpn_access_params_schema_requires_employee_id_and_justification():
    schema = TOOL_REGISTRY["grant_vpn_access"].params_schema
    assert schema["required"] == ["employee_id", "justification"]
    assert schema["properties"]["employee_id"]["type"] == "integer"
    assert schema["properties"]["justification"]["type"] == "string"


def test_tool_registry_is_immutable():
    with pytest.raises(TypeError):
        TOOL_REGISTRY["lookup_employee_entitlements"] = TOOL_REGISTRY["grant_vpn_access"]  # type: ignore[index]
    with pytest.raises(TypeError):
        del TOOL_REGISTRY["grant_vpn_access"]  # type: ignore[attr-defined]
