"""Tests for `POST /tickets/{id}/diagnose` (Phase 13 Task 4 -- see
`docs/superpowers/plans/2026-10-21-phase13-adk-diagnostic-specialist.md`'s
"Task 4" section and `docs/adr/0004-adk-diagnostic-specialist-wiring.md`'s
Task 4 addendum).

Matches this phase's "nothing mocked" standing directive exactly like
`test_diagnostic_agent.py` does: the one real, happy-path diagnostic run
below is a genuine, non-mocked end-to-end call through the real running
FastAPI app, against the real LiteLLM proxy + Ollama `local-qwen3` -- there
is no `llm_gateway.complete` to patch the way `test_ticket_summarize.py`
does, since ADK's own `LiteLlm` wrapper never calls that function at all
(see `diagnostic_agent.py`'s module docstring). The authz-rejection tests
(403 for a plain employee, 404 for a nonexistent ticket) are real but cheap
-- both paths return before any real `Agent`/model call is ever built.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.main import app
from resolvegrid_api.models import (
    AgentRun,
    Department,
    Employee,
    EmployeeEntitlement,
    Location,
    Queue,
    RoleAssignment,
    Span,
    Ticket,
    TicketMessage,
    TicketStateTransition,
)
from resolvegrid_api.operational_adapters.entitlements import grant_vpn_access

_SEED_EMAILS = [
    "diagnose.requester@example.test",
    "diagnose.analyst@example.test",
    "diagnose.outsider@example.test",
]
_SEED_QUEUE_NAME = "Diagnose Test Queue"
_SEED_DEPARTMENT_NAME = "Diagnose Test Dept"
_SEED_LOCATION_NAME = "Diagnose Test HQ"


@pytest.fixture(scope="module")
def client():
    # Module-scoped: `with TestClient(app) as test_client:` re-runs main.py's
    # real lifespan (tracing/checkpointer setup) once for this whole file's
    # tests, mirroring test_chat_api.py's own `client` fixture precedent --
    # this file's tests don't need a fresh lifespan per test.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def diagnose_fixtures(raw_db_session):
    # Mirrors test_ticket_summarize.py's summarize_fixtures pattern exactly
    # (get-or-create org data by natural key; ticket/message/transition
    # rows wiped every run since nothing else holds a real FK into them).
    raw_db_session.execute(delete(TicketStateTransition))
    raw_db_session.execute(delete(TicketMessage))
    raw_db_session.execute(delete(Ticket))
    raw_db_session.commit()

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
            display_name="Diagnose Requester", email=_SEED_EMAILS[0], title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(requester)
        raw_db_session.flush()

    analyst = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[1]))
    if analyst is None:
        analyst = Employee(
            display_name="Diagnose Analyst", email=_SEED_EMAILS[1], title="Analyst",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(analyst)
        raw_db_session.flush()

    # A plain employee with no role grant -- `ticket.diagnose` is staff-only
    # (packages/authz/policy.py's _STAFF_ONLY_ACTIONS), so this principal
    # must be denied outright, even though they're the ticket's own
    # requester (unlike `ticket.view`, there is no self-scoped carve-out).
    outsider = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[2]))
    if outsider is None:
        outsider = Employee(
            display_name="Diagnose Outsider", email=_SEED_EMAILS[2], title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(outsider)
        raw_db_session.flush()

    raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == analyst.id))
    raw_db_session.add(RoleAssignment(employee_id=analyst.id, role="analyst", scope="department", scope_id=dept.id))
    raw_db_session.flush()
    raw_db_session.commit()
    return requester, analyst, outsider, queue, dept


@pytest.fixture(autouse=True)
def _cleanup_agent_runs(raw_db_session):
    # Mirrors test_chat_api.py's `_cleanup_agent_runs` fixture exactly (same
    # real FK-constraint reason: AgentRun.principal_employee_id is a real
    # FK into employee.id, and this file's fixture employees are reused,
    # get-or-create, across runs -- leaving residue would break a later
    # full-reseed).
    #
    # Also cleans up any real `EmployeeEntitlement` row the real end-to-end
    # test's `grant_vpn_access(raw_db_session, requester.id, ...)` call
    # creates (a REAL commit, unlike `test_diagnostic_agent.py`'s own
    # equivalent call via the rollback-based `db_session` fixture) -- found
    # the hard way via this codebase's own full-suite run:
    # `test_operational_adapters_entitlements.py::test_grant_vpn_access_
    # creates_separate_rows_for_different_employees` asserts an exact count
    # of `EmployeeEntitlement` rows for the global `vpn_access` entitlement,
    # which a leftover real row from this file would silently inflate.
    yield
    for email in _SEED_EMAILS:
        employee = raw_db_session.scalar(select(Employee).where(Employee.email == email))
        if employee is None:
            continue
        raw_db_session.execute(
            delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee.id)
        )
        run_ids = raw_db_session.scalars(
            select(AgentRun.id).where(AgentRun.principal_employee_id == employee.id)
        ).all()
        if run_ids:
            raw_db_session.execute(delete(Span).where(Span.agent_run_id.in_(run_ids)))
            raw_db_session.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        raw_db_session.commit()


def _create_ticket(api_client, requester_id: int, queue_id: int, subject: str) -> int:
    response = api_client.post(
        "/tickets",
        json={"subject": subject, "body": "Cannot connect to VPN from home", "type": "incident", "queue_id": queue_id},
        headers={"X-Debug-Employee-Id": str(requester_id)},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_diagnose_ticket_denied_for_plain_employee(diagnose_fixtures, client):
    """A plain employee -- even the ticket's own requester -- gets a real
    403, never a self-scoped allowance. Cheap: the authz check runs before
    any real `Agent`/model call.
    """
    requester, analyst, outsider, queue, dept = diagnose_fixtures
    ticket_id = _create_ticket(client, requester.id, queue.id, "VPN down (authz test)")

    response = client.post(
        f"/tickets/{ticket_id}/diagnose",
        headers={"X-Debug-Employee-Id": str(outsider.id)},
    )
    assert response.status_code == 403


def test_diagnose_ticket_404_for_nonexistent_ticket(diagnose_fixtures, client, raw_db_session):
    """A real analyst principal hitting a real nonexistent ticket id gets a
    real 404 -- `diagnose_ticket` raises `TicketNotFoundError` from its own
    internal `get_ticket_state_and_history` call, BEFORE building any real
    `Agent`, so this is also cheap (no live model call). The resulting
    `AgentRun` row is still written, `status="error"`, matching every other
    error path's real-row contract.
    """
    requester, analyst, outsider, queue, dept = diagnose_fixtures

    response = client.post(
        "/tickets/999999999/diagnose",
        headers={"X-Debug-Employee-Id": str(analyst.id)},
    )
    assert response.status_code == 404

    run = raw_db_session.scalar(
        select(AgentRun)
        .where(AgentRun.principal_employee_id == analyst.id)
        .order_by(AgentRun.id.desc())
    )
    assert run is not None
    assert run.status == "error"
    assert run.error_message is not None


def test_diagnose_ticket_real_end_to_end_returns_summary_and_writes_agent_run_and_span(
    diagnose_fixtures, client, raw_db_session
):
    """The real completion proof for this endpoint: a real analyst principal,
    a real seeded ticket (with a real active entitlement so the specialist
    has real data to find if it calls `get_employee_entitlements`), a real
    HTTP call through the real running app, asserting:
    - 200 with a real, non-empty `summary` and a `citations` list shaped
      exactly like `/chat`'s (`chunk_id`/`document_title` dicts),
    - a real `AgentRun` row (`status="completed"`, real `output_text`),
    - exactly one real `Span` row (`stage_name="diagnose_ticket"`,
      `status="success"`, a real non-zero `latency_ms`, and a real
      `detail_json` carrying `tool_calls_made`/`turn_count`/
      `model_calls_logged`).
    """
    requester, analyst, outsider, queue, dept = diagnose_fixtures
    grant_vpn_access(raw_db_session, requester.id, "pre-existing access, for diagnostic context")
    ticket_id = _create_ticket(client, requester.id, queue.id, "VPN connection fails after password reset")

    response = client.post(
        f"/tickets/{ticket_id}/diagnose",
        headers={"X-Debug-Employee-Id": str(analyst.id)},
    )

    assert response.status_code == 200
    body = response.json()
    # Real, documented exceptions to "always non-empty" (Phase 13 Task 6,
    # found via this exact test during real fresh-state/CI runs): qwen3:14b
    # sometimes hallucinates its final answer as a fake, unregistered tool
    # call instead of plain final text (`diagnostic_agent.diagnose_ticket`'s
    # own `except ValueError` branch) -- `diagnose_ticket` recovers real
    # text from that call's own arguments when possible, but an honestly
    # empty summary is the correct, non-fabricated outcome on the rare real
    # run where even that recovery finds nothing usable. Separately, under
    # slow-hardware conditions (confirmed for real in CI), qwen3:14b's
    # documented repeated-tool-re-invocation pathology can genuinely
    # exhaust `_MAX_LLM_CALLS=60` within one run, leaving no final answer
    # at all (`exceeded_call_limit`). A NORMAL (neither hallucinated nor
    # call-limit-exhausted) run must still produce real, non-empty content.
    assert body["summary"].strip() or body["hallucinated_tool_call"] or body["exceeded_call_limit"], (
        f"empty summary on an otherwise-normal run is a real bug: {body!r}"
    )
    assert isinstance(body["citations"], list)
    for citation in body["citations"]:
        assert set(citation.keys()) == {"chunk_id", "document_title"}
    assert isinstance(body["tool_calls_made"], list)
    assert body["thread_id"]

    run = raw_db_session.scalar(
        select(AgentRun)
        .where(AgentRun.principal_employee_id == analyst.id)
        .order_by(AgentRun.id.desc())
    )
    assert run is not None
    assert run.status == "completed"
    assert run.output_text == body["summary"]
    assert run.completed_at is not None
    assert run.thread_id == body["thread_id"]

    spans = raw_db_session.scalars(select(Span).where(Span.agent_run_id == run.id)).all()
    assert len(spans) == 1
    span = spans[0]
    assert span.stage_name == "diagnose_ticket"
    assert span.status == "success"
    assert span.latency_ms > 0
    detail = json.loads(span.detail_json)
    assert detail["tool_calls_made"] == body["tool_calls_made"]
    assert isinstance(detail["turn_count"], int)
    assert isinstance(detail["model_calls_logged"], int)
