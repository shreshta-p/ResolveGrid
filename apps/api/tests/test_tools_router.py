"""Integration tests for Phase 9 Task 7a's `POST /tools/{tool_name}/invoke`
endpoint (`resolvegrid_api.routers.tools`).

This is the first test suite that exercises a real, running LangGraph graph
(`app.state.tool_invocation_graph`, `build_tool_invocation_graph`) pausing at
a genuine `interrupt()` reachable through the actual app, rather than a
standalone ad hoc graph (Task 5's own tests) or direct function calls (Task
6's own tests) -- see `graph.py`'s "Phase 9 Task 7a" docstring section for
why this gap existed and why this endpoint closes it.

Docker/Postgres required (real `resolvegrid` Postgres container -- same
requirement as every other DB-touching test in this file's siblings).

Session/visibility note: employees and role grants are created via the
committing `raw_db_session` fixture (mirrors `test_chat_api.py`'s
`chat_fixtures` pattern), NOT the rolled-back `db_session` fixture -- the
HTTP request under test runs through `get_db()`'s own fresh `Session` on a
different DB connection, which cannot see an uncommitted transaction on
`db_session`'s separate connection. Everything this file creates is cleaned
up explicitly afterward (FK-dependency order: ToolCall -> ApprovalRequest ->
RoleAssignment -> Employee), keeping this suite's blast radius contained
per the empty-corpus-first testing discipline documented in
`docs/DECISION_LOG.md`'s 2026-08-28 entry.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.main import app
from resolvegrid_api.models import ApprovalRequest, RoleAssignment, ToolCall
from resolvegrid_api.models.org import Employee, EmployeeEntitlement

_GRANT_JUSTIFICATION = "tools-router-test: onboarding contractor"


@pytest.fixture(scope="module")
def client():
    # Module-scoped: re-running main.py's lifespan (init_tracing,
    # AsyncPostgresSaver.setup() for BOTH compiled graphs now,
    # build_graph/build_tool_invocation_graph) per test would work too
    # (setup() is documented+verified idempotent) but is unnecessary cost --
    # mirrors test_chat_api.py's own `client` fixture precedent exactly.
    with TestClient(app) as test_client:
        yield test_client


def _make_employee(session, suffix: str) -> Employee:
    employee = Employee(
        display_name=f"Tools Router Employee {suffix}",
        email=f"tools.router.{suffix}@example.test",
        title="IT Analyst",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    session.add(employee)
    session.flush()
    return employee


def _cleanup_employee(raw_db_session, employee_id: int) -> None:
    """Delete every row this file's fixtures/tests may have created for one
    employee, in FK-dependency order, so this suite leaves no residue for
    other tests (see module docstring)."""
    approval_ids = raw_db_session.scalars(
        select(ApprovalRequest.id).where(ApprovalRequest.requested_by_id == employee_id)
    ).all()
    if approval_ids:
        raw_db_session.execute(delete(ToolCall).where(ToolCall.approval_request_id.in_(approval_ids)))
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id.in_(approval_ids)))
    raw_db_session.execute(
        delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee_id)
    )
    raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == employee_id))
    raw_db_session.execute(delete(Employee).where(Employee.id == employee_id))
    raw_db_session.commit()


@pytest.fixture
def analyst_employee(raw_db_session):
    """An employee holding the "analyst" role globally -- satisfies both
    registered tools' `required_role` (see `packages/contracts/tools.py`)."""
    employee = _make_employee(raw_db_session, "analyst")
    raw_db_session.add(RoleAssignment(employee_id=employee.id, role="analyst", scope="global"))
    raw_db_session.commit()
    try:
        yield employee
    finally:
        _cleanup_employee(raw_db_session, employee.id)


@pytest.fixture
def no_role_employee(raw_db_session):
    """An employee with NO role grants at all -- `available_tools_for_
    principal` must filter both registered tools out for them entirely."""
    employee = _make_employee(raw_db_session, "norole")
    raw_db_session.commit()
    try:
        yield employee
    finally:
        _cleanup_employee(raw_db_session, employee.id)


def _delete_checkpoint_thread(thread_id: str) -> None:
    """Best-effort cleanup of the paused graph run's checkpoint rows via
    the app's own shared checkpointer -- mirrors `test_checkpoint_restore.py`'s
    `_delete_thread` pattern, but reuses `app.state.tool_invocation_graph`'s
    already-open checkpointer rather than opening a brand new one, since
    this file doesn't need a separate instance for anything else."""
    checkpointer = app.state.tool_invocation_graph.checkpointer
    asyncio.run(checkpointer.adelete_thread(thread_id))


def test_invoke_readonly_tool_returns_immediately_with_no_approval_request(
    client, analyst_employee, raw_db_session
):
    before_count = raw_db_session.scalar(
        select(ApprovalRequest.id).where(ApprovalRequest.requested_by_id == analyst_employee.id)
    )
    assert before_count is None

    response = client.post(
        "/tools/lookup_employee_entitlements/invoke",
        json={"params": {"employee_id": analyst_employee.id}},
        headers={"X-Debug-Employee-Id": str(analyst_employee.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["tool_name"] == "lookup_employee_entitlements"
    assert body["output"] == []  # no entitlements granted to this fresh employee

    # No ApprovalRequest was ever created for a read-only tool invocation --
    # it never touches request_approval/the tool-invocation graph at all.
    after = raw_db_session.scalar(
        select(ApprovalRequest.id).where(ApprovalRequest.requested_by_id == analyst_employee.id)
    )
    assert after is None


def test_invoke_mutating_tool_creates_pending_approval_and_genuinely_pauses_graph(
    client, analyst_employee, raw_db_session
):
    response = client.post(
        "/tools/grant_vpn_access/invoke",
        json={
            "params": {
                "employee_id": analyst_employee.id,
                "justification": _GRANT_JUSTIFICATION,
            }
        },
        headers={"X-Debug-Employee-Id": str(analyst_employee.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_approval"
    approval_request_id = body["approval_request_id"]
    thread_id = body["thread_id"]
    assert approval_request_id is not None
    assert thread_id

    try:
        # Real ApprovalRequest row exists, status="pending", and its
        # agent_run_id is exactly this call's own thread_id -- confirms the
        # agent_run_id/thread_id flow documented in routers/tools.py.
        row = raw_db_session.get(ApprovalRequest, approval_request_id)
        assert row is not None
        assert row.status == "pending"
        assert row.agent_run_id == thread_id
        assert row.action_type == "grant_vpn_access"
        assert row.requested_by_id == analyst_employee.id

        # Confirm the underlying graph thread is GENUINELY paused right now
        # -- not just "the endpoint didn't error" -- via the real
        # aget_state() API (verified against the installed langgraph==1.2.11
        # source before writing this: StateSnapshot.next names the paused
        # node, .interrupts carries the pending Interrupt).
        snapshot = asyncio.run(
            app.state.tool_invocation_graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
        )
        assert snapshot.next == ("request_approval",)
        assert len(snapshot.interrupts) == 1
        assert snapshot.interrupts[0].value["approval_request_id"] == approval_request_id
        assert snapshot.interrupts[0].value["action_type"] == "grant_vpn_access"
        # The node paused before its own return statement -- execute_mutation
        # has not run, and approval_decision is still unset in the persisted
        # channel values.
        assert snapshot.values.get("approval_decision") is None
    finally:
        _delete_checkpoint_thread(thread_id)


def test_invoke_mutating_tool_without_required_role_returns_403(client, no_role_employee):
    response = client.post(
        "/tools/grant_vpn_access/invoke",
        json={
            "params": {
                "employee_id": no_role_employee.id,
                "justification": _GRANT_JUSTIFICATION,
            }
        },
        headers={"X-Debug-Employee-Id": str(no_role_employee.id)},
    )

    assert response.status_code == 403
    # Safe, non-leaking message -- see ToolNotAllowedError's own docstring:
    # must not reveal whether the tool doesn't exist vs. isn't permitted.
    assert response.json()["detail"] == "tool not allowed"


def test_invoke_mutating_tool_with_missing_required_param_returns_422(client, analyst_employee):
    response = client.post(
        "/tools/grant_vpn_access/invoke",
        json={"params": {"employee_id": analyst_employee.id}},  # missing "justification"
        headers={"X-Debug-Employee-Id": str(analyst_employee.id)},
    )

    assert response.status_code == 422
