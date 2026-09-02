"""Read-only entitlement lookups + the idempotent VPN-access grant adapter
(Phase 9 Task 3).

Plan amendment note (documented here per the plan doc's instruction, and
recorded as a Phase 9 Task 3 decision-log entry in Task 9's documentation
pass): the original plan-mode architecture doc's Section 5 layout named
`services/operational-adapters/` as a new top-level uv workspace package.
Building it that way would require that package to import `apps/api`'s
`Employee`/`EmployeeEntitlement`/`Entitlement`/`AccessGroup` ORM models
directly (Phase 9 Task 1 put them in `resolvegrid_api.models.org`, and
there is no DB-free way to query or insert them) -- exactly the "app
importing a library that imports back into the app" anti-pattern that
`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`'s
module docstring documents and rejects. It is the same tension
`apps/api/src/resolvegrid_api/knowledge_store.py`'s docstring already
resolved once for Phase 7 (and `retrieval.py` again for Phase 7 Task 4/5):
`services/retrieval` stays a pure, DB-free library, and the SQLAlchemy-
touching glue lives in `apps/api`, which already owns the models it needs.
This module follows that same established, twice-precedented resolution:
it is a plain module inside `apps/api` that takes a SQLAlchemy `Session` as
a parameter and owns all DB access, exactly like `knowledge_store.py` and
`retrieval.py` -- no new workspace package, no new `pyproject.toml`, no new
CI job. Nothing outside `apps/api` imports this module.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.models.org import AccessGroup, EmployeeEntitlement, Entitlement

# The well-known access-group/entitlement names this module seeds and
# grants against. "Network Access" is this module's own naming choice for
# the access group (the plan's Task 3 amendment note leaves this to
# implementer judgment) -- chosen to read as a category that could later
# hold sibling entitlements (site-to-site VPN, remote-desktop gateway,
# etc.), distinct from `test_approvals_tools_models.py`'s unrelated
# schema-smoke-test access-group name ("Networking") for the same
# "VPN Access" entitlement name; that test's writes live in their own
# rolled-back transaction, so there is no naming collision at the DB level.
VPN_ACCESS_GROUP_NAME = "Network Access"
VPN_ENTITLEMENT_NAME = "VPN Access"


@dataclass(frozen=True)
class EntitlementSummary:
    """Plain result shape for one active entitlement grant -- mirrors
    `retrieval.py`'s `SufficiencyResult` pattern of returning a small local
    dataclass rather than leaking the ORM row (and its joined `Entitlement`/
    `AccessGroup` relationships) past the query boundary.
    """

    entitlement_id: int
    entitlement_name: str
    access_group_name: str
    granted_at: datetime


def lookup_employee_entitlements(session: Session, employee_id: int) -> list[EntitlementSummary]:
    """Return active (non-revoked) entitlement grants for `employee_id`.

    Read-only: joins `EmployeeEntitlement` -> `Entitlement` -> `AccessGroup`
    and filters to `revoked_at IS NULL` -- a revoked grant is never
    returned, even though its row still exists for audit history. Ordered
    by `granted_at` so repeated calls against unchanged data return results
    in a stable, predictable order.
    """
    rows = session.execute(
        select(
            Entitlement.id,
            Entitlement.name,
            AccessGroup.name,
            EmployeeEntitlement.granted_at,
        )
        .join(Entitlement, EmployeeEntitlement.entitlement_id == Entitlement.id)
        .join(AccessGroup, Entitlement.access_group_id == AccessGroup.id)
        .where(
            EmployeeEntitlement.employee_id == employee_id,
            EmployeeEntitlement.revoked_at.is_(None),
        )
        .order_by(EmployeeEntitlement.granted_at)
    ).all()

    return [
        EntitlementSummary(
            entitlement_id=entitlement_id,
            entitlement_name=entitlement_name,
            access_group_name=access_group_name,
            granted_at=granted_at,
        )
        for entitlement_id, entitlement_name, access_group_name, granted_at in rows
    ]


def ensure_vpn_entitlement_seeded(session: Session) -> Entitlement:
    """Idempotently ensure the well-known `"VPN Access"` `Entitlement`
    (under the `"Network Access"` `AccessGroup`) exists, and return it.

    Query-first, insert-only-if-missing: calling this repeatedly (once per
    `grant_vpn_access` call, potentially many times across separate agent
    runs) must never create a duplicate row. This is what makes
    `grant_vpn_access` safe to call idempotently, and what Task 9's
    restart-mid-approval test depends on being real, not assumed.
    """
    entitlement = session.execute(
        select(Entitlement).where(Entitlement.name == VPN_ENTITLEMENT_NAME)
    ).scalar_one_or_none()
    if entitlement is not None:
        return entitlement

    access_group = session.execute(
        select(AccessGroup).where(AccessGroup.name == VPN_ACCESS_GROUP_NAME)
    ).scalar_one_or_none()
    if access_group is None:
        access_group = AccessGroup(
            name=VPN_ACCESS_GROUP_NAME,
            description="Access to internal network resources via corporate VPN.",
        )
        session.add(access_group)
        session.flush()

    entitlement = Entitlement(
        access_group_id=access_group.id,
        name=VPN_ENTITLEMENT_NAME,
        description="Grants corporate VPN access.",
    )
    session.add(entitlement)
    session.flush()
    return entitlement


def grant_vpn_access(session: Session, employee_id: int, justification: str) -> EmployeeEntitlement:
    """Grant `employee_id` the well-known `"VPN Access"` entitlement, or
    return their existing active grant unchanged if one already exists.

    Idempotent by design: this is the exact re-execution-safety property
    Task 9's restart-mid-approval test depends on being real -- a graph run
    that resumes from a checkpoint after a process restart may call this
    again for the same employee, and it must not create a second
    `EmployeeEntitlement` row. The check is a real query against
    `(employee_id, entitlement_id, revoked_at IS NULL)`, not an assumption
    that the caller only invokes this once.

    `justification` note: `EmployeeEntitlement` (Task 1's schema, see
    `resolvegrid_api.models.org`) has no column to persist a free-text
    justification on the grant row itself. It is accepted here as a
    parameter for interface completeness -- a later task (Task 6) wiring
    this adapter into the approval-execution flow is expected to fold it
    into the `AuditLog` `before_json`/`after_json` diff recorded around the
    grant, rather than storing it on `EmployeeEntitlement`. It is otherwise
    unused by this function; no schema change was made to accommodate it.
    """
    entitlement = ensure_vpn_entitlement_seeded(session)

    existing = session.execute(
        select(EmployeeEntitlement).where(
            EmployeeEntitlement.employee_id == employee_id,
            EmployeeEntitlement.entitlement_id == entitlement.id,
            EmployeeEntitlement.revoked_at.is_(None),
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    grant = EmployeeEntitlement(employee_id=employee_id, entitlement_id=entitlement.id)
    session.add(grant)
    session.flush()
    return grant
