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


# Actions where a principal with no matching role grant still gets a
# self-scoped Decision (they may see/act on their OWN resources).
_SELF_SCOPED_ACTIONS = {
    "directory.list_employees",
    "directory.view_employee",
    "ticket.list",
    "ticket.view",
    # Phase 7 Task 5 (knowledge retrieval authz filtering): reuses this same
    # admin/department-scoped/self-scoped Decision shape rather than a
    # bespoke retrieval policy -- see
    # apps/api/src/resolvegrid_api/retrieval_authz.py for how the caller
    # interprets a self-scoped Decision (no department grant) for knowledge
    # retrieval specifically, since a Document isn't "owned" by an employee
    # the way a Ticket is.
    "knowledge.retrieve",
}
# Actions that require an actual staff/admin grant -- a principal with no
# matching grant is DENIED outright, never silently downgraded to self-scope
# (e.g. a plain employee must never be able to transition ticket status).
# Phase 9 Task 7b: "approval.list"/"approval.decide" join this set for the
# identical reason -- an ApprovalRequest is never a personal resource an
# ordinary employee should see or decide on (unlike, say, their own ticket),
# so a principal with no admin/department-scoped analyst-or-approver grant
# must be denied outright, exactly matching ticket.transition's precedent,
# not downgraded to a self-scoped Decision that would (incorrectly) let them
# see/decide approvals they merely requested or are the subject of.
_STAFF_ONLY_ACTIONS = {"ticket.transition", "approval.list", "approval.decide"}

_KNOWN_ACTIONS = _SELF_SCOPED_ACTIONS | _STAFF_ONLY_ACTIONS


def _is_global_admin(principal: Principal) -> bool:
    """Does `principal` hold an admin grant scoped globally?

    The single shared definition of "admin dominates" -- both `authorize()`
    and `principal_has_role()` call this instead of each inlining
    `any(g.role == "admin" and g.scope == "global" for g in
    principal.roles)`. If this check ever needs to change (a new admin
    scope type, a revoked-admin flag, etc.), both callers pick up the
    change automatically instead of silently drifting apart.
    """
    return any(g.role == "admin" and g.scope == "global" for g in principal.roles)


def authorize(principal: Principal, action: str) -> Decision:
    """Authorize an action for a principal.

    Returns the principal's allowed access SET:
    - department_ids is None and employee_id is None: unrestricted (admin).
    - department_ids is a non-empty tuple: restricted to resources scoped to
      those departments (employees in that department, or tickets whose
      Queue.department_id is in that set -- callers interpret the tuple
      against whatever resource they're authorizing).
    - employee_id is set (department_ids is None): restricted to that single
      employee's own resources (self-view, or tickets they themselves filed).

    Callers are responsible for checking the SPECIFIC resource they're about
    to return against this Decision -- authorize() computes the allowed set,
    it has no knowledge of any specific resource instance.

    Admin dominates regardless of grant order (grant order from a database
    query is NOT guaranteed). A department-scoped grant with a missing
    scope_id is ignored (not treated as unrestricted) -- a misconfigured
    grant must never silently escalate to broader access. Staff-only actions
    (see _STAFF_ONLY_ACTIONS) are denied outright for principals with no
    admin/department grant, never downgraded to a self-scoped Decision.

    This is the single, centralized policy entry point -- callers (API
    routes, future tool/retrieval nodes) must never re-implement
    authorization logic inline. See
    docs/adr/0001-modular-monolith-topology.md.
    """
    if action not in _KNOWN_ACTIONS:
        return Decision(allowed=False, reason=f"unknown action: {action}")

    if _is_global_admin(principal):
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

    if action in _STAFF_ONLY_ACTIONS:
        return Decision(allowed=False, reason="requires an admin or department-scoped analyst/approver grant")

    return Decision(
        allowed=True,
        reason="default: no matching role grant, self-view only",
        employee_id=principal.employee_id,
    )


def principal_has_role(principal: Principal, role: str) -> bool:
    """Does `principal` hold `role` (in any scope)?

    A general RBAC-style role check, distinct from `authorize()` -- the
    latter is action-per-resource-type (e.g. "can this principal list
    employees") and doesn't map onto "does this principal hold role X" for
    an arbitrary role string (e.g. a tool's `required_role`). This helper
    exists so callers that need that simpler check (see
    `apps/api/src/resolvegrid_api/tool_execution.py`'s
    `available_tools_for_principal`) still go through `packages/authz`
    instead of re-implementing role-string comparisons inline -- this
    package remains the single source of truth for "what can this
    principal do."

    Shares `authorize()`'s admin-dominates precedent via `_is_global_admin`
    -- a global admin grant satisfies any requested role, regardless of
    scope -- rather than inlining an independent copy of that check.
    Otherwise, true iff `principal.roles` contains a grant with `role ==
    role` in ANY scope (global/department/team) -- unlike `authorize()`,
    this check doesn't interpret `scope`/`scope_id` into a Decision, since
    callers of this helper only need a yes/no role check, not a
    resource-filtering scope.
    """
    if _is_global_admin(principal):
        return True
    return any(g.role == role for g in principal.roles)
