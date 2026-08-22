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


def test_self_management_is_flagged_as_a_cycle(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a5@example.test")

    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=a.id) is True


def test_dangling_new_manager_id_does_not_cycle(db_session):
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a6@example.test")
    nonexistent_manager_id = a.id + 999999

    assert would_create_cycle(db_session, employee_id=a.id, new_manager_id=nonexistent_manager_id) is False


def test_preexisting_cycle_in_data_is_treated_as_unsafe(db_session):
    # Construct a pre-existing cycle directly (bypassing the guard, as if bad
    # data already exists), then confirm the guard's visited-set defense
    # correctly flags it as unsafe rather than looping forever.
    location = Location(name="HQ", region="US", timezone="America/Chicago")
    department = Department(name="Eng")
    db_session.add_all([location, department])
    db_session.flush()

    a = _make_employee(db_session, location, department, email="a7@example.test")
    b = _make_employee(db_session, location, department, email="b7@example.test", manager_id=a.id)
    db_session.flush()
    a.manager_id = b.id  # closes a pre-existing a<->b cycle directly, bypassing the guard
    db_session.flush()

    c = _make_employee(db_session, location, department, email="c7@example.test")
    assert would_create_cycle(db_session, employee_id=c.id, new_manager_id=a.id) is True
