from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict

# Mirrors the role strings packages/authz's policy.py actually checks against
# (see e.g. `g.role == "admin"` and `g.role in ("analyst", "approver")`) --
# kept as a closed Literal, like tickets.py's TicketStatus/TicketType/
# TicketPriority, so a typo here fails at construction time instead of
# silently making a tool unreachable via authorize().
ToolRole = Literal["admin", "analyst", "approver"]


class ToolContract(BaseModel):
    """Describes one agent-invocable tool: what it needs to run and who may run it.

    `params_schema` is a JSON Schema dict validated against a proposed tool call's
    params before execution (see Task 4's `validate_tool_schema`). `required_role`
    and `required_entitlement` are consulted by `authorize()` (via Task 4's
    `available_tools_for_principal`) to filter this tool out of a principal's
    available-tools list before it is ever exposed to the model -- never bypass
    that filtering with an ad hoc role-string comparison elsewhere.

    Frozen for the same reason packages/authz's `Decision`/`Principal` are
    frozen dataclasses: `mutating`/`requires_approval`/`required_role` are
    security-relevant flags, and a frozen model turns any accidental
    post-construction mutation of them into a raised error instead of silent
    corruption of a shared registry entry.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    params_schema: dict
    required_role: ToolRole
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
                "additionalProperties": False,
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
                "additionalProperties": False,
            },
            required_role="analyst",
            required_entitlement=None,
            mutating=True,
            requires_approval=True,
        ),
    }
)
