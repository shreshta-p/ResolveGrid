"""Deterministic seed generator for Phase 2's org/identity domain.

Generates a small synthetic employee directory: locations, departments,
teams, employees with a valid non-cyclic manager tree, and a handful of role
assignments. Fully deterministic for a given seed and idempotent -- re-running
with the same seed wipes and reproduces byte-identical data. Not real data;
see docs/SECURITY.md. Scales up (to ~1,000 employees, full scenario manifests,
devices/entitlements) in a later phase per the approved architecture plan §8.
"""

import argparse
import os

from faker import Faker
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import Session

from resolvegrid_api.models import AgentRun, AuditLog, Department, Employee, Location, RoleAssignment, Team

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)

_LOCATIONS = [
    ("Austin HQ", "US-Central", "America/Chicago"),
    ("Remote - EU", "EU-West", "Europe/Berlin"),
    ("Remote - APAC", "APAC", "Asia/Singapore"),
]

_DEPARTMENTS = ["Platform Engineering", "IT Support", "Security", "People Ops"]


def generate_org(session: Session, seed: int, num_employees: int = 75) -> None:
    fake = Faker()
    Faker.seed(seed)

    if num_employees < 1:
        raise ValueError("num_employees must be at least 1")

    # Wipe existing rows for idempotency -- except any Employee already
    # referenced by a real AuditLog row (audit_log.actor_id has no cascade;
    # the log is append-only/tamper-evident by design, so such a row can
    # never be deleted -- see apps/api/src/resolvegrid_api/audit.py) OR a
    # real AgentRun row (agent_run.principal_employee_id is a real FK too,
    # added in Phase 6 -- apps/api/tests/test_chat_api.py's fixture commits
    # real /chat activity for a dedicated employee the same way
    # test_tickets_api.py's fixture does for AuditLog). This only happens
    # today via those two test fixtures, which commit real activity under
    # dedicated employees, so also protect the small closure of rows that
    # would otherwise dangle: those employees' department_id/location_id/
    # team_id. This was originally a narrow, targeted fix that assumed no
    # protected employee would ever have manager_id set -- that assumption
    # broke for real once enough real sessions/tests accumulated employees
    # whose manager chains cross into the protected set (observed directly:
    # `ForeignKeyViolation` on `employee_manager_id_fkey`, key (id)=(570)
    # still referenced, during this exact DELETE). Fixed the same way
    # `Team.lead_employee_id` is already handled below: null out any
    # protected employee's `manager_id` before the delete, rather than
    # extending this into a general transitive-closure solver.
    protected_employee_ids = set(
        session.execute(select(AuditLog.actor_id).where(AuditLog.actor_id.isnot(None)).distinct()).scalars()
    ) | set(
        session.execute(
            select(AgentRun.principal_employee_id).where(AgentRun.principal_employee_id.isnot(None)).distinct()
        ).scalars()
    )
    protected_department_ids: set[int] = set()
    protected_location_ids: set[int] = set()
    protected_team_ids: set[int] = set()
    if protected_employee_ids:
        for dept_id, loc_id, team_id in session.execute(
            select(Employee.department_id, Employee.location_id, Employee.team_id).where(
                Employee.id.in_(protected_employee_ids)
            )
        ):
            if dept_id is not None:
                protected_department_ids.add(dept_id)
            if loc_id is not None:
                protected_location_ids.add(loc_id)
            if team_id is not None:
                protected_team_ids.add(team_id)

    # `Team.department_id` is NOT NULL (unlike `manager_id`/`lead_employee_id`
    # above), so a protected team's own department can't be nulled out --
    # it must be added to protected_department_ids directly, one more real
    # hop in the same closure (observed directly: `ForeignKeyViolation` on
    # `team_department_id_fkey` once the manager_id fix above was applied
    # and this hop was still missing). A protected team's department need
    # not match any protected employee's own department_id, since those are
    # independent columns.
    if protected_team_ids:
        protected_department_ids |= set(
            session.execute(select(Team.department_id).where(Team.id.in_(protected_team_ids))).scalars()
        )

    # Team and Employee mutually reference each other (Employee.team_id ->
    # team.id, Team.lead_employee_id -> employee.id), so Team.lead_employee_id
    # must be nulled out before Employee can be safely deleted -- this
    # generator never actually sets lead_employee_id today, so the update is
    # a no-op in practice, but without it this delete order breaks the moment
    # any future code populates a team lead (the same circular FK Task 3 hit
    # on insert).
    session.execute(update(Team).values(lead_employee_id=None))
    if protected_employee_ids:
        session.execute(
            update(Employee).where(Employee.id.in_(protected_employee_ids)).values(manager_id=None)
        )
    session.execute(delete(RoleAssignment))
    session.execute(delete(Employee).where(Employee.id.notin_(protected_employee_ids)))
    session.execute(delete(Team).where(Team.id.notin_(protected_team_ids)))
    session.execute(delete(Department).where(Department.id.notin_(protected_department_ids)))
    session.execute(delete(Location).where(Location.id.notin_(protected_location_ids)))
    session.flush()

    # `Location.name`/`Department.name` are unique. Earlier this only
    # protected a "handful" of employees (per this function's own docstring
    # history), so a fresh INSERT of these exact canonical names never
    # collided in practice. That assumption is now false for real: this dev
    # DB has accumulated 22 real protected employees (observed directly via
    # `docs/PROGRESS.md`'s Phase 13 Task 6 seed count), spanning enough of
    # the 3 canonical locations/4 departments that a protected row with one
    # of these exact names now always already exists (observed directly:
    # `UniqueViolation` on `location_name_key`, key (name)=(Remote - APAC)).
    # Get-or-create by name, reusing a protected row if one already
    # survived the wipe above, rather than assuming a clean INSERT.
    existing_locations = {
        loc.name: loc
        for loc in session.execute(select(Location).where(Location.id.in_(protected_location_ids))).scalars()
    } if protected_location_ids else {}
    locations = []
    for n, r, tz in _LOCATIONS:
        if n in existing_locations:
            locations.append(existing_locations[n])
        else:
            loc = Location(name=n, region=r, timezone=tz)
            session.add(loc)
            locations.append(loc)
    session.flush()

    existing_departments = {
        dept.name: dept
        for dept in session.execute(select(Department).where(Department.id.in_(protected_department_ids))).scalars()
    } if protected_department_ids else {}
    departments = []
    for n in _DEPARTMENTS:
        if n in existing_departments:
            departments.append(existing_departments[n])
        else:
            dept = Department(name=n)
            session.add(dept)
            departments.append(dept)
    session.flush()

    teams = []
    for dept in departments:
        for i in range(2):
            teams.append(Team(name=f"{dept.name} Team {i + 1}", department_id=dept.id))
    session.add_all(teams)
    session.flush()

    employees: list[Employee] = []
    for i in range(num_employees):
        dept = departments[i % len(departments)]
        team = teams[i % len(teams)]
        location = locations[i % len(locations)]
        employees.append(
            Employee(
                display_name=fake.name(),
                email=fake.unique.email(),
                title=fake.job(),
                employment_status="active",
                hire_date=fake.date_time_this_decade(),
                timezone=location.timezone,
                location_id=location.id,
                department_id=dept.id,
                team_id=team.id,
                manager_id=None,
            )
        )
    session.add_all(employees)
    session.flush()

    # Shallow, non-cyclic manager tree by construction: the first
    # `manager_pool_size` employees are roots (manager_id stays None); every
    # later employee reports to one of those roots, deterministically.
    manager_pool_size = max(1, num_employees // 5)
    for i, employee in enumerate(employees):
        if i < manager_pool_size:
            continue
        employee.manager_id = employees[i % manager_pool_size].id
    session.flush()

    # Role assignments: employee 0 gets global admin; the rest of the manager
    # pool gets department-scoped analyst/approver grants for their own dept.
    role_assignments = [RoleAssignment(employee_id=employees[0].id, role="admin", scope="global")]
    for i in range(1, manager_pool_size):
        employee = employees[i]
        role = "approver" if i % 2 == 0 else "analyst"
        role_assignments.append(
            RoleAssignment(
                employee_id=employee.id,
                role=role,
                scope="department",
                scope_id=employee.department_id,
            )
        )
    session.add_all(role_assignments)
    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ResolveGrid's org/identity tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--employees", type=int, default=75)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        generate_org(session, seed=args.seed, num_employees=args.employees)
    print(f"Seeded {args.employees} employees (seed={args.seed}).")


if __name__ == "__main__":
    main()
