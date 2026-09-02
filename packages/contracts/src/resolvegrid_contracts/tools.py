from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel


class ToolContract(BaseModel):
    """Describes one agent-invocable tool: what it needs to run and who may run it.

    `params_schema` is a JSON Schema dict validated against a proposed tool call's
    params before execution (see Task 4's `validate_tool_schema`). `required_role`
    and `required_entitlement` are consulted by `authorize()` (via Task 4's
    `available_tools_for_principal`) to filter this tool out of a principal's
    available-tools list before it is ever exposed to the model -- never bypass
    that filtering with an ad hoc role-string comparison elsewhere.
    """

    name: str
    version: str
    description: str
    params_schema: dict
    required_role: str
    required_entitlement: str | None = None
    mutating: bool
    requires_approval: bool


TOOL_REGISTRY: Mapping[str, ToolContract] = MappingProxyType(
    {
        "lookup_employee_entitlements": ToolContract(
            name="lookup_employee_entitlements",
            version="1.0.0",
            description="Look up an employee's active (non-revoked) entitlements and access groups.",
            params_schema={
                "type": "object",
                "properties": {"employee_id": {"type": "integer"}},
                "required": ["employee_id"],
            },
            required_role="analyst",
            required_entitlement=None,
            mutating=False,
            requires_approval=False,
        ),
        "grant_vpn_access": ToolContract(
            name="grant_vpn_access",
            version="1.0.0",
            description="Grant an employee VPN access, recording the justification for audit.",
            params_schema={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["employee_id", "justification"],
            },
            required_role="analyst",
            required_entitlement=None,
            mutating=True,
            requires_approval=True,
        ),
    }
)
