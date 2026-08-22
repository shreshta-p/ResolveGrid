from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleGrant:
    role: str
    scope: str  # "global" | "department" | "team"
    scope_id: int | None = None


@dataclass(frozen=True)
class Principal:
    employee_id: int
    roles: list[RoleGrant] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    filter: dict[str, int] = field(default_factory=dict)


_KNOWN_ACTIONS = {"directory.list_employees", "directory.view_employee"}


def authorize(
    principal: Principal, action: str, department_id: int | None = None
) -> Decision:
    """Authorize a directory action for a principal.

    Returns a Decision whose `filter` narrows what the caller may see:
    - {} means no restriction.
    - {"department_id": N} restricts results to department N.
    - {"employee_id": N} restricts results to a single employee (self only).

    This is the single, centralized policy entry point -- callers (API routes,
    future tool/retrieval nodes) must never re-implement authorization logic
    inline. See docs/adr/0001-modular-monolith-topology.md Decision C.
    """
    if action not in _KNOWN_ACTIONS:
        return Decision(allowed=False, reason=f"unknown action: {action}")

    for grant in principal.roles:
        if grant.role == "admin" and grant.scope == "global":
            return Decision(allowed=True, reason="admin: global access")
        if grant.role in ("analyst", "approver") and grant.scope == "department":
            if department_id is None or department_id == grant.scope_id:
                filter_ = {"department_id": grant.scope_id} if grant.scope_id is not None else {}
                return Decision(allowed=True, reason=f"{grant.role}: department-scoped access", filter=filter_)

    return Decision(
        allowed=True,
        reason="default: no matching role grant, self-view only",
        filter={"employee_id": principal.employee_id},
    )
