import pytest
from pydantic import ValidationError

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


def test_params_schema_rejects_additional_properties():
    # additionalProperties: false so Task 4's pre-execution schema validation
    # rejects unexpected extra keys on a proposed tool call, rather than
    # silently letting them through.
    for tool in TOOL_REGISTRY.values():
        assert tool.params_schema["additionalProperties"] is False


def test_lookup_employee_entitlements_params_schema_requires_employee_id():
    schema = TOOL_REGISTRY["lookup_employee_entitlements"].params_schema
    assert schema["required"] == ["employee_id"]
    assert schema["properties"]["employee_id"]["type"] == "integer"


def test_grant_vpn_access_params_schema_requires_employee_id_and_justification():
    schema = TOOL_REGISTRY["grant_vpn_access"].params_schema
    assert schema["required"] == ["employee_id", "justification"]
    assert schema["properties"]["employee_id"]["type"] == "integer"
    assert schema["properties"]["justification"]["type"] == "string"


def test_required_role_rejects_unknown_role_string():
    # required_role is a closed Literal of the actual role strings
    # packages/authz's policy.py checks against, so a typo (e.g. "anaylst")
    # fails at construction time instead of silently making a tool
    # unreachable via authorize().
    with pytest.raises(ValidationError):
        ToolContract(
            name="x",
            version="1.0.0",
            description="y",
            params_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            required_role="anaylst",  # type: ignore[arg-type]
            mutating=False,
            requires_approval=False,
        )


def test_tool_registry_mapping_shell_is_immutable():
    # MappingProxyType blocks rebinding/removing entries in the dict itself...
    with pytest.raises(TypeError):
        TOOL_REGISTRY["lookup_employee_entitlements"] = TOOL_REGISTRY["grant_vpn_access"]  # type: ignore[index]
    with pytest.raises(TypeError):
        del TOOL_REGISTRY["grant_vpn_access"]  # type: ignore[attr-defined]


def test_tool_contract_instances_are_frozen():
    # ...but that alone doesn't stop someone mutating a security-relevant
    # flag on a contract object already sitting in the registry. ToolContract
    # is `frozen=True` specifically so this raises instead of silently
    # corrupting a shared, already-registered contract.
    with pytest.raises(ValidationError):
        TOOL_REGISTRY["grant_vpn_access"].requires_approval = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TOOL_REGISTRY["lookup_employee_entitlements"].mutating = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TOOL_REGISTRY["grant_vpn_access"].required_role = "admin"  # type: ignore[misc]
