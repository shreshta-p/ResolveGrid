from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.models import RoleAssignment
from resolvegrid_authz import Principal, RoleGrant


def get_principal(
    x_debug_employee_id: int | None = Header(default=None),
    session: Session = Depends(get_db),
) -> Principal:
    """Resolve the requesting principal.

    TEMPORARY: no real authentication exists yet (JWT/session auth is a later
    phase per the approved architecture plan's agent-workflow stage 1). Until
    then, every caller must pass X-Debug-Employee-Id to identify themselves.
    This is NOT a security boundary as written -- anyone can claim any
    employee id -- and MUST be replaced before any non-local deployment.
    Tracked as a documented decision in docs/DECISION_LOG.md.
    """
    if x_debug_employee_id is None:
        raise HTTPException(
            status_code=401,
            detail="X-Debug-Employee-Id header required (temporary auth shim, see docs/DECISION_LOG.md)",
        )
    grants = session.scalars(
        select(RoleAssignment).where(RoleAssignment.employee_id == x_debug_employee_id)
    ).all()
    roles = tuple(RoleGrant(role=g.role, scope=g.scope, scope_id=g.scope_id) for g in grants)
    return Principal(employee_id=x_debug_employee_id, roles=roles)
