from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.authz_scope import apply_decision_scope, resource_in_scope
from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import Employee
from resolvegrid_authz import Decision, Principal, authorize

router = APIRouter(prefix="/directory", tags=["directory"])


def _employee_to_dict(employee: Employee) -> dict:
    return {
        "id": employee.id,
        "display_name": employee.display_name,
        "email": employee.email,
        "title": employee.title,
        "department_id": employee.department_id,
        "manager_id": employee.manager_id,
    }


def _apply_scope(query, decision: Decision):
    """Narrow a query to a Decision's allowed set -- delegates to
    `authz_scope.apply_decision_scope` (Phase 13 Task 2 code-review
    follow-up: this used to be an independent copy of the same two rules
    `routers/tickets.py`'s `_apply_ticket_scope` implements for `Ticket`).
    Unrestricted (admin) decisions (department_ids is None and employee_id
    is None) apply no filter at all."""
    return apply_decision_scope(
        query, decision, department_id_column=Employee.department_id, owner_id_column=Employee.id
    )


def _in_scope(decision: Decision, employee: Employee) -> bool:
    """Verify one already-fetched Employee is within a Decision's allowed
    set -- delegates to `authz_scope.resource_in_scope` (see that module's
    docstring). Required for single-resource endpoints, where a query-level
    filter alone isn't enough to prove the specific row returned was
    actually authorized."""
    return resource_in_scope(decision, department_id=employee.department_id, owner_id=employee.id)


@router.get("/employees")
def list_employees(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict]:
    decision = authorize(principal, "directory.list_employees")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    query = _apply_scope(select(Employee), decision)
    employees = session.scalars(query).all()
    return [_employee_to_dict(e) for e in employees]


@router.get("/employees/{employee_id}")
def get_employee(
    employee_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    decision = authorize(principal, "directory.view_employee")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    employee = session.get(Employee, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    if not _in_scope(decision, employee):
        raise HTTPException(status_code=403, detail="not authorized to view this employee")

    return _employee_to_dict(employee)


@router.get("/employees/{employee_id}/reports")
def list_direct_reports(
    employee_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict]:
    decision = authorize(principal, "directory.list_employees")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    manager = session.get(Employee, employee_id)
    if manager is None:
        raise HTTPException(status_code=404, detail="employee not found")
    if not _in_scope(decision, manager):
        raise HTTPException(status_code=403, detail="not authorized to view this employee's reports")

    query = _apply_scope(select(Employee).where(Employee.manager_id == employee_id), decision)
    reports = session.scalars(query).all()
    return [_employee_to_dict(e) for e in reports]
