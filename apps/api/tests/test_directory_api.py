import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.main import app
from resolvegrid_api.models import Department, Employee, Location, RoleAssignment

client = TestClient(app)

_SEED_EMAILS = ["admin@example.test", "analyst.a@example.test", "plain.b@example.test"]
_SEED_DEPARTMENT_NAMES = ["Dept A", "Dept B"]
_SEED_LOCATION_NAME = "Directory Test HQ"


@pytest.fixture
def seeded_departments(raw_db_session):
    # Idempotent: delete any leftover rows from a prior test run before
    # reinserting -- these are real, committed rows (raw_db_session is
    # required here since client.get(...) hits the router's own separate
    # DB connection, so seeded data must actually be committed for the HTTP
    # request to see it), and Department.name / Location.name / Employee.email
    # are all unique constraints that would otherwise collide across
    # repeated runs.
    raw_db_session.execute(
        delete(RoleAssignment).where(
            RoleAssignment.employee_id.in_(select(Employee.id).where(Employee.email.in_(_SEED_EMAILS)))
        )
    )
    raw_db_session.execute(delete(Employee).where(Employee.email.in_(_SEED_EMAILS)))
    raw_db_session.execute(delete(Department).where(Department.name.in_(_SEED_DEPARTMENT_NAMES)))
    raw_db_session.execute(delete(Location).where(Location.name == _SEED_LOCATION_NAME))
    raw_db_session.commit()

    loc = Location(name=_SEED_LOCATION_NAME, region="US", timezone="America/Chicago")
    dept_a = Department(name="Dept A")
    dept_b = Department(name="Dept B")
    raw_db_session.add_all([loc, dept_a, dept_b])
    raw_db_session.flush()

    admin = Employee(
        display_name="Admin One", email="admin@example.test", title="Admin",
        hire_date="2024-01-01T00:00:00", timezone=loc.timezone,
        location_id=loc.id, department_id=dept_a.id,
    )
    analyst_a = Employee(
        display_name="Analyst A", email="analyst.a@example.test", title="Analyst",
        hire_date="2024-01-01T00:00:00", timezone=loc.timezone,
        location_id=loc.id, department_id=dept_a.id,
    )
    plain_b = Employee(
        display_name="Plain B", email="plain.b@example.test", title="Engineer",
        hire_date="2024-01-01T00:00:00", timezone=loc.timezone,
        location_id=loc.id, department_id=dept_b.id,
    )
    raw_db_session.add_all([admin, analyst_a, plain_b])
    raw_db_session.flush()

    raw_db_session.add_all([
        RoleAssignment(employee_id=admin.id, role="admin", scope="global"),
        RoleAssignment(employee_id=analyst_a.id, role="analyst", scope="department", scope_id=dept_a.id),
    ])
    raw_db_session.flush()
    raw_db_session.commit()
    return admin, analyst_a, plain_b, dept_a, dept_b


def test_missing_principal_header_is_rejected():
    response = client.get("/directory/employees")
    assert response.status_code == 401


def test_admin_sees_all_employees(seeded_departments):
    admin, analyst_a, plain_b, _, _ = seeded_departments
    response = client.get("/directory/employees", headers={"X-Debug-Employee-Id": str(admin.id)})
    assert response.status_code == 200
    emails = {e["email"] for e in response.json()}
    assert {"admin@example.test", "analyst.a@example.test", "plain.b@example.test"} <= emails


def test_analyst_sees_only_own_department(seeded_departments):
    admin, analyst_a, plain_b, dept_a, dept_b = seeded_departments
    response = client.get("/directory/employees", headers={"X-Debug-Employee-Id": str(analyst_a.id)})
    assert response.status_code == 200
    emails = {e["email"] for e in response.json()}
    assert "analyst.a@example.test" in emails
    assert "admin@example.test" in emails  # same department (dept_a)
    assert "plain.b@example.test" not in emails  # different department


def test_plain_employee_sees_only_self(seeded_departments):
    admin, analyst_a, plain_b, _, _ = seeded_departments
    response = client.get("/directory/employees", headers={"X-Debug-Employee-Id": str(plain_b.id)})
    assert response.status_code == 200
    emails = {e["email"] for e in response.json()}
    assert emails == {"plain.b@example.test"}


def test_get_single_employee_respects_authorization(seeded_departments):
    admin, analyst_a, plain_b, _, _ = seeded_departments
    # plain_b may view themself...
    response = client.get(f"/directory/employees/{plain_b.id}", headers={"X-Debug-Employee-Id": str(plain_b.id)})
    assert response.status_code == 200
    # ...but not admin.
    response = client.get(f"/directory/employees/{admin.id}", headers={"X-Debug-Employee-Id": str(plain_b.id)})
    assert response.status_code == 403
