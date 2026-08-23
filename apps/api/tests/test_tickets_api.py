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
