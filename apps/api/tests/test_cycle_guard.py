from resolvegrid_api.models import Department, Employee, Location
from resolvegrid_api.org.cycle_guard import would_create_cycle


def _make_employee(db_session, location, department, **overrides):
    defaults = dict(
        display_name="Employee",
        email=f"employee-{id(overrides)}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    defaults.update(overrides)
    employee = Employee(**defaults)
    db_session.add(employee)
    db_session.flush()
    return employee


def test_no_cycle_for_a_fresh_assignment(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a@example.test")
    b = _make_employee(db_session, location, department, email="b@example.test")

    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=b.id) is False


def test_detects_direct_cycle(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a2@example.test")
    b = _make_employee(db_session, location, department, email="b2@example.test", manager_id=a.id)
    db_session.flush()

    # a already manages b (transitively, here directly) -- making b manage a would cycle.
    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=b.id) is True


def test_detects_transitive_cycle(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a3@example.test")
    b = _make_employee(db_session, location, department, email="b3@example.test", manager_id=a.id)
    db_session.flush()
    c = _make_employee(db_session, location, department, email="c3@example.test", manager_id=b.id)
    db_session.flush()

    # a -> manages b -> manages c. Making a report to c would close the loop.
    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=c.id) is True


def test_unrelated_employees_do_not_cycle(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a4@example.test")
    b = _make_employee(db_session, location, department, email="b4@example.test")
    c = _make_employee(db_session, location, department, email="c4@example.test", manager_id=b.id)
    db_session.flush()

    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=c.id) is False
