from resolvegrid_api.models import Department, Employee, Location, RoleAssignment, Team


def test_can_create_and_query_org_entities(db_session):
    location = Location(name="Test HQ", region="US-Test", timezone="America/Chicago")
    db_session.add(location)
    db_session.flush()

    department = Department(name="Test Department")
    db_session.add(department)
    db_session.flush()

    team = Team(name="Test Team", department_id=department.id)
    db_session.add(team)
    db_session.flush()

    manager = Employee(
        display_name="Manager One",
        email="manager.one@example.test",
        title="Engineering Manager",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
        team_id=team.id,
    )
    db_session.add(manager)
    db_session.flush()

    report = Employee(
        display_name="Report One",
        email="report.one@example.test",
        title="Engineer",
        hire_date="2024-02-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
        team_id=team.id,
        manager_id=manager.id,
    )
    db_session.add(report)
    db_session.flush()

    role = RoleAssignment(employee_id=manager.id, role="analyst", scope="department", scope_id=department.id)
    db_session.add(role)
    db_session.flush()

    fetched_report = db_session.get(Employee, report.id)
    assert fetched_report is not None
    assert fetched_report.manager_id == manager.id
    assert fetched_report.manager.display_name == "Manager One"
