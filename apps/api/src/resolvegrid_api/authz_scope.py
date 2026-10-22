"""Shared `Decision` -> SQL-predicate / in-scope-boolean translation (Phase
13 Task 2 code-review follow-up).

`packages/authz/src/resolvegrid_authz/policy.py`'s `authorize()` returns a
`Decision` whose `department_ids`/`employee_id` fields are interpreted
IDENTICALLY everywhere in this codebase a department-scoped resource is
authz-filtered: `routers/tickets.py` (`Ticket`, scoped via
`Queue.department_id` + `Ticket.requester_id`), `routers/directory.py`
(`Employee`, scoped via its own `department_id` + `id`), and
`diagnostic_tools.py` (the same `Ticket` rule, for the ADK diagnostic
specialist's tools). Before this module existed, that identical two-rule
interpretation ("`department_ids` restricts by a department column;
`employee_id` restricts by an owner column") was copy-pasted independently
in all three places -- a real review-flagged risk: a future change to
`Decision` (e.g. a new field) would need updating in three places in
lockstep with nothing enforcing that, and `diagnostic_tools.py`'s copy had
already drifted into existence as a third independent instance of the exact
same logic before this fix.

This module factors out ONLY the entity-agnostic half of that logic -- the
part that's genuinely identical across every caller. Each caller still owns
its own entity-specific JOIN (if one is needed to reach the department
column), since that part is genuinely different per entity (and, for
`Ticket`, conditional -- see `apply_decision_scope`'s docstring) and does
not belong in a shared, entity-agnostic helper.
"""

from __future__ import annotations

from sqlalchemy import Select

from resolvegrid_authz import Decision


def apply_decision_scope(
    query: Select, decision: Decision, *, department_id_column, owner_id_column
) -> Select:
    """Narrow `query` to `decision`'s allowed set via the caller-supplied
    department/owner columns.

    `department_id_column`/`owner_id_column` are plain SQLAlchemy column
    references (e.g. `Queue.department_id`, `Ticket.requester_id`, or
    `Employee.department_id`/`Employee.id`). The caller is responsible for
    any JOIN needed to make `department_id_column` reachable from `query`
    BEFORE calling this -- that join can't move into this shared helper
    because `routers/tickets.py`'s original `_apply_ticket_scope` makes it
    conditional on `decision.department_ids is not None`, to avoid an
    unnecessary JOIN for a self-scoped/admin decision; folding the join in
    here would force it unconditionally for every caller.

    Mirrors `authorize()`'s own `Decision` semantics (see `policy.py`'s
    docstring) exactly: `department_ids is not None` restricts to resources
    whose department column is in that set; `employee_id is not None`
    restricts to resources whose owner column equals it. An unrestricted
    (admin) decision -- both `None` -- applies no filter at all.
    """
    if decision.department_ids is not None:
        query = query.where(department_id_column.in_(decision.department_ids))
    if decision.employee_id is not None:
        query = query.where(owner_id_column == decision.employee_id)
    return query


def resource_in_scope(decision: Decision, *, department_id: int | None, owner_id: int | None) -> bool:
    """Verify one already-fetched resource (its resolved department id and
    owner/self id) is within `decision`'s allowed set.

    The single-resource-level counterpart to `apply_decision_scope`'s
    query-level filter -- needed because a query-level filter alone doesn't
    prove a specific already-fetched row was actually authorized (the exact
    reasoning `routers/directory.py`'s original `_in_scope` docstring gave
    before this extraction).
    """
    if decision.employee_id is not None and decision.employee_id != owner_id:
        return False
    if decision.department_ids is not None and department_id not in decision.department_ids:
        return False
    return True
