from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.llm_gateway import CompletionResult, LLMGatewayError
from resolvegrid_api.main import app
from resolvegrid_api.models import (
    Department,
    Employee,
    Location,
    ModelCall,
    Queue,
    RoleAssignment,
    Ticket,
    TicketMessage,
    TicketStateTransition,
)

client = TestClient(app)

_SEED_EMAILS = [
    "summarize.requester@example.test",
    "summarize.analyst@example.test",
    "summarize.outsider@example.test",
]
_SEED_QUEUE_NAME = "Summarize Test Queue"
_SEED_DEPARTMENT_NAME = "Summarize Test Dept"
_SEED_LOCATION_NAME = "Summarize Test HQ"


@pytest.fixture
def summarize_fixtures(raw_db_session):
    # Mirrors apps/api/tests/test_tickets_api.py's ticket_fixtures pattern:
    # ticket/ticket_message/ticket_state_transition are wiped every run
    # (children before parent, to satisfy FKs); org data (employee/
    # department/location/queue) is get-or-create by natural key since
    # AuditLog rows created by earlier test runs hold real FKs into
    # employee.id that must never be deleted out from under the
    # append-only, hash-chained audit log.
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
            display_name="Summarize Requester", email=_SEED_EMAILS[0], title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(requester)
        raw_db_session.flush()

    analyst = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[1]))
    if analyst is None:
        analyst = Employee(
            display_name="Summarize Analyst", email=_SEED_EMAILS[1], title="Analyst",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(analyst)
        raw_db_session.flush()

    # A plain employee with no role grant, and NOT the ticket's requester --
    # authorize() gives them a self-scoped Decision (employee_id=own id),
    # so they must be rejected when trying to view/summarize someone else's
    # ticket, regardless of department.
    outsider = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAILS[2]))
    if outsider is None:
        outsider = Employee(
            display_name="Summarize Outsider", email=_SEED_EMAILS[2], title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(outsider)
        raw_db_session.flush()

    raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == analyst.id))
    raw_db_session.add(RoleAssignment(employee_id=analyst.id, role="analyst", scope="department", scope_id=dept.id))
    raw_db_session.flush()
    raw_db_session.commit()
    return requester, analyst, outsider, queue, dept


def _create_ticket(requester_id: int, queue_id: int, subject: str = "VPN down") -> int:
    response = client.post(
        "/tickets",
        json={"subject": subject, "body": "Cannot connect to VPN from home", "type": "incident", "queue_id": queue_id},
        headers={"X-Debug-Employee-Id": str(requester_id)},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_summarize_ticket_returns_summary_and_writes_model_call(summarize_fixtures, raw_db_session):
    requester, analyst, outsider, queue, dept = summarize_fixtures
    ticket_id = _create_ticket(requester.id, queue.id)

    # model="local-qwen3" -- llm_gateway.complete()'s CompletionResult.model
    # always carries the caller-requested/LiteLLM-alias name (its default,
    # "local-qwen3"), never the raw underlying Ollama tag "qwen3:14b" -- a
    # real call's response JSON echoes the same alias back too. Using the
    # raw tag here would silently mask exactly the pricing-lookup mismatch
    # this test's own pricing_version_id assertion is supposed to catch
    # (see migration 0007's docstring for the real bug this caused).
    fake_result = CompletionResult(
        text="User cannot connect to VPN from home.",
        input_tokens=128, output_tokens=52, latency_ms=340,
        provider="ollama", model="local-qwen3",
    )
    with patch("resolvegrid_api.routers.tickets.llm_gateway.complete", return_value=fake_result) as mock_complete:
        response = client.post(
            f"/tickets/{ticket_id}/summarize",
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == fake_result.text
    assert body["input_tokens"] == 128
    assert body["output_tokens"] == 52
    assert body["latency_ms"] == 340
    # Seeded ollama/qwen3:14b PricingVersion row is $0/$0.
    assert body["estimated_cost_usd"] == 0.0
    mock_complete.assert_called_once()

    call = raw_db_session.scalar(
        select(ModelCall).where(ModelCall.purpose == "ticket.summarize").order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "success"
    assert call.provider == "ollama"
    assert call.model == "local-qwen3"
    assert call.input_tokens == 128
    assert call.output_tokens == 52
    assert call.latency_ms == 340
    assert call.estimated_cost_usd == 0.0
    # The seeded PricingVersion row for ollama/local-qwen3 must have been found.
    assert call.pricing_version_id is not None


def test_summarize_ticket_falls_back_to_zero_cost_when_no_pricing_version_matches(summarize_fixtures, raw_db_session):
    # A model/provider combination with no seeded PricingVersion row (unlike
    # the real ollama/local-qwen3 row) must not crash -- pricing_version_id
    # stays None and cost is treated as 0.0 rather than raising.
    requester, analyst, outsider, queue, dept = summarize_fixtures
    ticket_id = _create_ticket(requester.id, queue.id)

    fake_result = CompletionResult(
        text="Summary from an unpriced model.",
        input_tokens=64, output_tokens=16, latency_ms=120,
        provider="anthropic", model="claude-not-yet-priced",
    )
    with patch("resolvegrid_api.routers.tickets.llm_gateway.complete", return_value=fake_result):
        response = client.post(
            f"/tickets/{ticket_id}/summarize",
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    assert response.json()["estimated_cost_usd"] == 0.0

    call = raw_db_session.scalar(
        select(ModelCall)
        .where(ModelCall.purpose == "ticket.summarize", ModelCall.provider == "anthropic")
        .order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "success"
    assert call.pricing_version_id is None
    assert call.estimated_cost_usd == 0.0


def test_summarize_ticket_rejects_employee_outside_scope(summarize_fixtures):
    requester, analyst, outsider, queue, dept = summarize_fixtures
    ticket_id = _create_ticket(requester.id, queue.id)

    response = client.post(
        f"/tickets/{ticket_id}/summarize",
        headers={"X-Debug-Employee-Id": str(outsider.id)},
    )
    assert response.status_code == 403


def test_summarize_ticket_records_fallback_when_gateway_reports_one(summarize_fixtures, raw_db_session):
    # Simulates what llm_gateway.complete() returns after a real forced
    # primary-failure-then-successful-fallback (empirically verified header
    # contract in docs/superpowers/plans/2026-08-25-phase5-cloud-fallback.md's
    # "Task 1 status": x-litellm-attempted-fallbacks > 0 and
    # x-litellm-model-group=cloud-fallback). This is a unit-level mock of the
    # gateway's OUTPUT, matching this file's established pattern for every
    # other test here -- no real LiteLLM/Anthropic/OpenAI call is made.
    requester, analyst, outsider, queue, dept = summarize_fixtures
    ticket_id = _create_ticket(requester.id, queue.id)

    fake_result = CompletionResult(
        text="Summary produced after a fallback to the secondary provider.",
        input_tokens=96, output_tokens=40, latency_ms=610,
        provider="openai", model="cloud-fallback",
        fallback_occurred=True, serving_model_group="cloud-fallback",
    )
    with patch("resolvegrid_api.routers.tickets.llm_gateway.complete", return_value=fake_result):
        response = client.post(
            f"/tickets/{ticket_id}/summarize",
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200

    call = raw_db_session.scalar(
        select(ModelCall)
        .where(ModelCall.purpose == "ticket.summarize", ModelCall.provider == "openai")
        .order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "success"
    assert call.fallback_occurred is True
    assert call.serving_model_group == "cloud-fallback"


def test_summarize_ticket_gateway_error_returns_502_and_records_model_call(summarize_fixtures, raw_db_session):
    requester, analyst, outsider, queue, dept = summarize_fixtures
    ticket_id = _create_ticket(requester.id, queue.id)

    with patch("resolvegrid_api.routers.tickets.llm_gateway.complete", side_effect=LLMGatewayError("boom")):
        response = client.post(
            f"/tickets/{ticket_id}/summarize",
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 502

    call = raw_db_session.scalar(
        select(ModelCall)
        .where(ModelCall.purpose == "ticket.summarize", ModelCall.status == "error")
        .order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "error"
    assert call.error_message == "boom"
    assert call.pricing_version_id is None
    assert call.input_tokens == 0
    assert call.output_tokens == 0
    assert call.latency_ms == 0
    assert call.estimated_cost_usd == 0.0
