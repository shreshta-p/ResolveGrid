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
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from resolvegrid_api.models import Department, Employee, Location, RoleAssignment, Team

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

    # Wipe existing rows for idempotency, in FK-safe (child-first) order.
    session.execute(delete(RoleAssignment))
    session.execute(delete(Employee))
    session.execute(delete(Team))
    session.execute(delete(Department))
    session.execute(delete(Location))
    session.flush()

    locations = [Location(name=n, region=r, timezone=tz) for n, r, tz in _LOCATIONS]
    session.add_all(locations)
    session.flush()

    departments = [Department(name=n) for n in _DEPARTMENTS]
    session.add_all(departments)
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
