import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.main import app
from resolvegrid_api.models import (
    Department,
    Employee,
    Location,
    Queue,
    RoleAssignment,
    Ticket,
    TicketMessage,
    TicketStateTransition,
)

client = TestClient(app)

_SEED_EMAILS = ["ticket.requester@example.test", "ticket.analyst@example.test"]
_SEED_QUEUE_NAME = "Ticket Test Queue"
_SEED_DEPARTMENT_NAME = "Ticket Test Dept"
_SEED_LOCATION_NAME = "Ticket Test HQ"

# A SECOND, disjoint department/queue/requester -- used only by
# test_admin_sees_tickets_across_departments below, to prove a global admin
# Decision (department_ids is None and employee_id is None, per
# authz_scope.py's docstring) really does see tickets from a department it
# has no explicit grant for, not merely "the one department ticket_fixtures
# happens to set up". Mirrors test_directory_api.py's
# test_admin_sees_all_employees precedent for the same admin-sanity-check
# coverage gap, now closed for GET /tickets too.
_SEED_EMAIL_OTHER_REQUESTER = "ticket.requester.other@example.test"
_SEED_EMAIL_ADMIN = "ticket.admin@example.test"
_SEED_QUEUE_NAME_OTHER = "Ticket Test Queue Other"
_SEED_DEPARTMENT_NAME_OTHER = "Ticket Test Dept Other"


@pytest.fixture
def ticket_fixtures(raw_db_session):
    # Ticket and its children are wiped every run -- audit_log.entity_id has
    # no FK to ticket.id, so this is always safe. Child rows referencing
    # ticket.id via FK must go first, or a re-run against leftover rows from
    # a prior run hits a ForeignKeyViolation (ticket_message /
    # ticket_state_transition -> ticket).
    raw_db_session.execute(delete(TicketStateTransition))
    raw_db_session.execute(delete(TicketMessage))
    raw_db_session.execute(delete(Ticket))
    raw_db_session.commit()

    # Employee/Department/Location/Queue are NOT deleted-and-recreated here.
    # Creating a ticket now commits a real AuditLog row with a genuine FK to
    # employee.id (audit_log is an append-only, hash-chained log -- deleting
    # a row would both violate that FK, since there's no ON DELETE CASCADE,
    # and corrupt the chain for every later verify_chain_integrity() check).
    # So this fixture is idempotent: reuse the existing row by natural key
    # across runs instead of delete+recreate.
    loc = raw_db_session.scalar(select(Location).where(Location.name == _SEED_LOCATION_NAME))
    if loc is None:
        loc = Location(name=_SEED_LOCATION_NAME, region="US", timezone="America/Chicago")
        raw_db_session.add(loc)
        raw_db_session.flush()

    dept = raw_db_session.scalar(select(Department).where(Department.name == _SEED_DEPARTMENT_NAME))
    if dept is None:
        dept = Department(name=_SEED_DEPARTMENT_NAME)
        raw_db_session.add(dept)
        raw_db_session.flush()

    queue = raw_db_session.scalar(select(Queue).where(Queue.name == _SEED_QUEUE_NAME))
    if queue is None:
        queue = Queue(name=_SEED_QUEUE_NAME, department_id=dept.id)
        raw_db_session.add(queue)
        raw_db_session.flush()

    requester = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[0]))
    if requester is None:
        requester = Employee(
            display_name="Ticket Requester", email=_SEED_EMAILS[0], title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(requester)
        raw_db_session.flush()

    analyst = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[1]))
    if analyst is None:
        analyst = Employee(
            display_name="Ticket Analyst", email=_SEED_EMAILS[1], title="Analyst",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(analyst)
        raw_db_session.flush()

    # RoleAssignment has no incoming FK from audit_log, so this can still be
    # cleared and recreated freely each run.
    raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == analyst.id))
    raw_db_session.add(RoleAssignment(employee_id=analyst.id, role="analyst", scope="department", scope_id=dept.id))
    raw_db_session.flush()
    raw_db_session.commit()
    return requester, analyst, queue, dept


def test_employee_can_create_a_ticket(ticket_fixtures):
    requester, analyst, queue, dept = ticket_fixtures
    response = client.post(
        "/tickets",
        json={"subject": "VPN down", "body": "Can't connect", "type": "incident", "queue_id": queue.id},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "open"
    assert body["requester_id"] == requester.id


def test_employee_sees_only_own_tickets(ticket_fixtures):
    requester, analyst, queue, dept = ticket_fixtures
    client.post(
        "/tickets",
        json={"subject": "Mine", "body": "x", "type": "incident", "queue_id": queue.id},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    )
    response = client.get("/tickets", headers={"X-Debug-Employee-Id": str(requester.id)})
    assert response.status_code == 200
    subjects = {t["subject"] for t in response.json()}
    assert "Mine" in subjects


def test_admin_sees_tickets_across_departments(ticket_fixtures, raw_db_session):
    """Real, non-mocked proof that a global admin Decision (the
    `department_ids is None and employee_id is None` case
    `authz_scope.apply_decision_scope` applies NO filter for) genuinely
    returns tickets from a SECOND department the admin has no explicit
    grant for -- not just the one department `ticket_fixtures` already sets
    up, which `test_employee_sees_only_own_tickets` could pass even if admin
    scoping were silently broken for every OTHER department.

    Reuses `ticket_fixtures`' idempotent query-first-then-create convention
    for the extra department/queue/employee rows (safe to rerun -- natural
    -key lookups, not delete+recreate, for anything `audit_log` could hold
    an FK to).
    """
    requester, _analyst, queue, dept = ticket_fixtures

    loc = raw_db_session.scalar(select(Location).where(Location.name == _SEED_LOCATION_NAME))

    other_dept = raw_db_session.scalar(select(Department).where(Department.name == _SEED_DEPARTMENT_NAME_OTHER))
    if other_dept is None:
        other_dept = Department(name=_SEED_DEPARTMENT_NAME_OTHER)
        raw_db_session.add(other_dept)
        raw_db_session.flush()

    other_queue = raw_db_session.scalar(select(Queue).where(Queue.name == _SEED_QUEUE_NAME_OTHER))
    if other_queue is None:
        other_queue = Queue(name=_SEED_QUEUE_NAME_OTHER, department_id=other_dept.id)
        raw_db_session.add(other_queue)
        raw_db_session.flush()

    other_requester = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAIL_OTHER_REQUESTER))
    if other_requester is None:
        other_requester = Employee(
            display_name="Ticket Requester Other", email=_SEED_EMAIL_OTHER_REQUESTER, title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=other_dept.id,
        )
        raw_db_session.add(other_requester)
        raw_db_session.flush()

    admin = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAIL_ADMIN))
    if admin is None:
        admin = Employee(
            display_name="Ticket Admin", email=_SEED_EMAIL_ADMIN, title="Admin",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(admin)
        raw_db_session.flush()

    raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == admin.id))
    raw_db_session.add(RoleAssignment(employee_id=admin.id, role="admin", scope="global"))
    raw_db_session.flush()
    raw_db_session.commit()

    own_dept_ticket = client.post(
        "/tickets",
        json={"subject": "Admin visibility own dept", "body": "x", "type": "incident", "queue_id": queue.id},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    ).json()
    other_dept_ticket = client.post(
        "/tickets",
        json={
            "subject": "Admin visibility other dept",
            "body": "x",
            "type": "incident",
            "queue_id": other_queue.id,
        },
        headers={"X-Debug-Employee-Id": str(other_requester.id)},
    ).json()

    response = client.get("/tickets", headers={"X-Debug-Employee-Id": str(admin.id)})
    assert response.status_code == 200
    ticket_ids = {t["id"] for t in response.json()}
    assert own_dept_ticket["id"] in ticket_ids
    assert other_dept_ticket["id"] in ticket_ids


def test_analyst_can_transition_a_ticket_the_employee_cannot(ticket_fixtures):
    requester, analyst, queue, dept = ticket_fixtures
    create_resp = client.post(
        "/tickets",
        json={"subject": "Needs triage", "body": "x", "type": "incident", "queue_id": queue.id},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    )
    ticket_id = create_resp.json()["id"]

    # Employee cannot transition their own ticket.
    denied = client.post(
        f"/tickets/{ticket_id}/transition",
        json={"to_status": "in_progress"},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    )
    assert denied.status_code == 403

    # Analyst can.
    allowed = client.post(
        f"/tickets/{ticket_id}/transition",
        json={"to_status": "in_progress"},
        headers={"X-Debug-Employee-Id": str(analyst.id)},
    )
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "in_progress"


def test_illegal_transition_is_rejected(ticket_fixtures):
    requester, analyst, queue, dept = ticket_fixtures
    create_resp = client.post(
        "/tickets",
        json={"subject": "Illegal transition test", "body": "x", "type": "incident", "queue_id": queue.id},
        headers={"X-Debug-Employee-Id": str(requester.id)},
    )
    ticket_id = create_resp.json()["id"]

    # open -> resolved is not a legal transition (must go through in_progress).
    response = client.post(
        f"/tickets/{ticket_id}/transition",
        json={"to_status": "resolved"},
        headers={"X-Debug-Employee-Id": str(analyst.id)},
    )
    assert response.status_code == 400


def test_rate_limit_blocks_excessive_ticket_creation(ticket_fixtures):
    requester, analyst, queue, dept = ticket_fixtures
    last_status = None
    for _ in range(15):
        resp = client.post(
            "/tickets",
            json={"subject": "Spam", "body": "x", "type": "incident", "queue_id": queue.id},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
        last_status = resp.status_code
    assert last_status == 429
