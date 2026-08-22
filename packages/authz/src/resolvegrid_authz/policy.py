from dataclasses import dataclass


@dataclass(frozen=True)
class RoleGrant:
    role: str
    scope: str  # "global" | "department" | "team"
    scope_id: int | None = None


@dataclass(frozen=True)
class Principal:
    employee_id: int
    roles: tuple[RoleGrant, ...] = ()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    department_ids: tuple[int, ...] | None = None  # None = no department restriction
    employee_id: int | None = None  # non-None = restrict to this single employee


_KNOWN_ACTIONS = {"directory.list_employees", "directory.view_employee"}


def authorize(principal: Principal, action: str) -> Decision:
    """Authorize a directory action for a principal.

    Returns the principal's allowed access SET:
    - department_ids is None and employee_id is None: unrestricted (admin).
    - department_ids is a non-empty tuple: restricted to those departments.
    - employee_id is set (department_ids is None): restricted to that single
      employee (self-view only).

    Callers are responsible for checking the SPECIFIC resource they're about
    to return against this Decision (e.g. "is this employee's department_id
    in decision.department_ids, or does this employee's id match
    decision.employee_id") -- authorize() computes the allowed set, it has no
    knowledge of any specific resource instance. Every caller must perform
    that membership check, not just the ones that happen to remember to.

    Admin dominates regardless of how many other grants a principal holds or
    what order they were loaded in -- grant order from a database query is
    NOT guaranteed. A department-scoped grant with a missing scope_id is
    ignored (not treated as unrestricted) -- a misconfigured grant must never
    silently escalate to broader access.

    This is the single, centralized policy entry point -- callers (API
    routes, future tool/retrieval nodes) must never re-implement
    authorization logic inline. See
    docs/adr/0001-modular-monolith-topology.md.
    """
    if action not in _KNOWN_ACTIONS:
        return Decision(allowed=False, reason=f"unknown action: {action}")

    if any(g.role == "admin" and g.scope == "global" for g in principal.roles):
        return Decision(allowed=True, reason="admin: global access")

    department_ids = tuple(
        sorted(
            {
                g.scope_id
                for g in principal.roles
                if g.role in ("analyst", "approver")
                and g.scope == "department"
                and g.scope_id is not None
            }
        )
    )
    if department_ids:
        return Decision(allowed=True, reason="department-scoped access", department_ids=department_ids)

    return Decision(
        allowed=True,
        reason="default: no matching role grant, self-view only",
        employee_id=principal.employee_id,
    )
