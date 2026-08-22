from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.models import Employee


def would_create_cycle(session: Session, employee_id: int, new_manager_id: int) -> bool:
    """Return True if setting employee_id's manager to new_manager_id would create a cycle.

    Walks up new_manager_id's manager chain; if employee_id appears in that
    chain, new_manager_id already reports (directly or transitively) to
    employee_id, so the reassignment would close a loop. Mirrors Project WIN's
    wouldCreateCircle pattern (see docs/adr/0001-modular-monolith-topology.md).
    """
    current_id: int | None = new_manager_id
    visited: set[int] = set()
    while current_id is not None:
        if current_id == employee_id:
            return True
        if current_id in visited:
            # A cycle already exists elsewhere in the data; treat as unsafe.
            return True
        visited.add(current_id)
        current_id = session.scalar(select(Employee.manager_id).where(Employee.id == current_id))
    return False
