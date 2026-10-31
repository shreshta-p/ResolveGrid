"""Phase 14 Task 3: integration tests for the real admin performance
dashboard API (`resolvegrid_api.routers.admin_performance`):
`GET /performance/metrics`.

Structure mirrors `test_evals_router.py`: a module-scoped `TestClient`
fixture reusing `main.py`'s real lifespan, employee/role fixtures built via
the committing `raw_db_session` fixture (the HTTP request under test runs
its own, separate `get_db()` session/connection), everything created is
cleaned up explicitly afterward.

The central test this file exists for -- `test_metrics_reflect_a_newly_
inserted_real_agent_run` -- is this task's explicit exit-criterion check:
"a regression test proves this is NOT a static/mock dashboard ... asserting
the endpoint's response changes when a new real AgentRun row is inserted,
not a fixture-based snapshot that would pass even if the endpoint secretly
returned a hardcoded JSON blob." It queries a fixed historical window via
`since`/`until`, confirms it starts empty (zero AgentRun rows that old),
inserts one real `AgentRun` + one real `Span` row with timestamps inside
that exact window, re-queries, and asserts every affected number (count,
latency, throughput, span-by-stage) moved by EXACTLY the amount the new row
accounts for -- a hardcoded response could never pass both calls.
"""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from resolvegrid_api.main import app
from resolvegrid_api.models import AgentRun, LoadTestRun, RoleAssignment, Span
from resolvegrid_api.models.org import Employee

# A fixed historical slot unlikely to collide with any seed/other-test
# traffic (every other fixture in this suite creates AgentRun rows via
# `server_default=func.now()`, i.e. near "now", never backdated to 2024).
_WINDOW_START = datetime(2024, 1, 1, 0, 0, 0)
_WINDOW_END = datetime(2024, 1, 1, 1, 0, 0)
_RUN_CREATED_AT = datetime(2024, 1, 1, 0, 30, 0)
_RUN_LATENCY_MS = 5000
_RUN_COMPLETED_AT = _RUN_CREATED_AT + timedelta(milliseconds=_RUN_LATENCY_MS)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _make_employee(session, suffix: str) -> Employee:
    employee = Employee(
        display_name=f"Perf Router Employee {suffix}",
        email=f"perf.router.{suffix}@example.test",
        title="IT Analyst",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    session.add(employee)
    session.flush()
    return employee


@pytest.fixture
def admin_employee(raw_db_session):
    """Global-admin principal -- the shape `authorize(principal,
    "performance.view")` requires for the happy-path assertions below."""
    employee = _make_employee(raw_db_session, "admin")
    raw_db_session.add(RoleAssignment(employee_id=employee.id, role="admin", scope="global"))
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == employee.id))
        raw_db_session.execute(delete(Employee).where(Employee.id == employee.id))
        raw_db_session.commit()


@pytest.fixture
def no_role_employee(raw_db_session):
    """A plain employee with NO role grants at all -- `performance.view`
    must deny them outright (staff-only action, no self-scoped downgrade),
    matching `test_evals_router.py`'s identical fixture for `eval.view`."""
    employee = _make_employee(raw_db_session, "norole")
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(Employee).where(Employee.id == employee.id))
        raw_db_session.commit()


def _window_params() -> dict:
    return {"since": _WINDOW_START.isoformat(), "until": _WINDOW_END.isoformat()}


def test_requires_admin_or_department_grant(client, no_role_employee):
    response = client.get(
        "/performance/metrics",
        params=_window_params(),
        headers={"X-Debug-Employee-Id": str(no_role_employee.id)},
    )
    assert response.status_code == 403


def test_rejects_ambiguous_window_selection(client, admin_employee):
    """`load_test_run_id` combined with `since`/`until` is ambiguous --
    there's no well-defined single window to aggregate."""
    response = client.get(
        "/performance/metrics",
        params={**_window_params(), "load_test_run_id": 1},
        headers={"X-Debug-Employee-Id": str(admin_employee.id)},
    )
    assert response.status_code == 400


def test_rejects_window_minutes_combined_with_load_test_run_id(client, admin_employee):
    """Code-review follow-up: `window_minutes` given alongside
    `load_test_run_id` was previously silently ignored (the load-test-run
    window won, `window_minutes` was dropped) rather than rejected as
    ambiguous, inconsistent with the `load_test_run_id`-vs-`since`/`until`
    case just above, which already was rejected. Now checked the same way."""
    response = client.get(
        "/performance/metrics",
        params={"load_test_run_id": 1, "window_minutes": 30},
        headers={"X-Debug-Employee-Id": str(admin_employee.id)},
    )
    assert response.status_code == 400


def test_rejects_window_minutes_combined_with_explicit_range(client, admin_employee):
    """Same inconsistency as above, for the `since`/`until` pairing:
    `window_minutes` must not be silently dropped just because an explicit
    range was also given."""
    response = client.get(
        "/performance/metrics",
        params={**_window_params(), "window_minutes": 30},
        headers={"X-Debug-Employee-Id": str(admin_employee.id)},
    )
    assert response.status_code == 400


def test_metrics_reflect_a_newly_inserted_real_agent_run(client, admin_employee, raw_db_session):
    """The core "not a mock" proof: same endpoint, same window, two calls --
    the only thing that changes between them is one real DB insert."""
    headers = {"X-Debug-Employee-Id": str(admin_employee.id)}

    before = client.get("/performance/metrics", params=_window_params(), headers=headers)
    assert before.status_code == 200
    before_body = before.json()
    assert before_body["throughput"]["total_agent_run_count"] == 0
    assert before_body["latency_ms"]["overall"].get("p50") is None
    assert before_body["span_by_stage"] == {}

    thread_id = f"perf-router-test-{uuid.uuid4()}"
    agent_run = AgentRun(
        thread_id=thread_id,
        status="completed",
        input_text="integration test: does the dashboard reflect a real new row?",
        output_text="yes",
        created_at=_RUN_CREATED_AT,
        completed_at=_RUN_COMPLETED_AT,
        cache_hit=False,
    )
    raw_db_session.add(agent_run)
    raw_db_session.flush()
    span = Span(
        agent_run_id=agent_run.id,
        stage_name="classify_intent",
        status="success",
        latency_ms=1234,
        created_at=_RUN_CREATED_AT,
    )
    raw_db_session.add(span)
    raw_db_session.commit()

    try:
        after = client.get("/performance/metrics", params=_window_params(), headers=headers)
        assert after.status_code == 200
        after_body = after.json()

        assert after_body["throughput"]["total_agent_run_count"] == 1
        assert after_body["throughput"]["completed_count"] == 1
        # duration_seconds == 3600 for this fixed one-hour window -> exactly
        # 1/3600 completed AgentRuns/sec.
        assert after_body["throughput"]["completed_agent_runs_per_sec"] == pytest.approx(1 / 3600)

        assert after_body["latency_ms"]["overall"]["p50"] == pytest.approx(float(_RUN_LATENCY_MS))
        assert after_body["latency_ms"]["chat"]["n"] == 1
        assert after_body["latency_ms"]["chat"]["p50_ms"] == pytest.approx(float(_RUN_LATENCY_MS))
        assert after_body["latency_ms"]["diagnose"] is None

        assert after_body["error_rate"]["total_count"] == 1
        assert after_body["error_rate"]["agent_run_error_rate"] == 0.0

        assert "classify_intent" in after_body["span_by_stage"]
        assert after_body["span_by_stage"]["classify_intent"]["n"] == 1
        assert after_body["span_by_stage"]["classify_intent"]["p50_ms"] == pytest.approx(1234.0)

        # Known, honestly-reported gaps must still be reported, not silently
        # dropped now that there's real data to aggregate.
        assert after_body["ttft"]["available"] is False
        assert after_body["cache"]["hit_rate_by_tier"]["retrieval"] is None
    finally:
        raw_db_session.execute(delete(Span).where(Span.agent_run_id == agent_run.id))
        raw_db_session.execute(delete(AgentRun).where(AgentRun.id == agent_run.id))
        raw_db_session.commit()


def test_load_test_run_window_selection_matches_explicit_range(client, admin_employee, raw_db_session):
    """The admin can select a window via a real `LoadTestRun`'s own
    started_at/completed_at (Task 2 ids 16/17 in the live stack) -- here
    proven with a fresh LoadTestRun row bracketing the same fixed-window
    AgentRun used above, confirming the buffered load-test-run window
    counts the identical real row the explicit since/until window does.
    """
    headers = {"X-Debug-Employee-Id": str(admin_employee.id)}

    thread_id = f"perf-router-test-ltr-{uuid.uuid4()}"
    agent_run = AgentRun(
        thread_id=thread_id,
        status="completed",
        input_text="integration test: load_test_run_id window selection",
        output_text="yes",
        created_at=_RUN_CREATED_AT,
        completed_at=_RUN_COMPLETED_AT,
        cache_hit=False,
    )
    raw_db_session.add(agent_run)
    raw_db_session.flush()

    run = LoadTestRun(
        started_at=_RUN_CREATED_AT - timedelta(minutes=1),
        completed_at=_RUN_CREATED_AT + timedelta(minutes=1),
        config_json="{}",
        git_commit="f" * 40,
    )
    raw_db_session.add(run)
    raw_db_session.flush()
    run_id = run.id
    raw_db_session.commit()

    try:
        response = client.get(
            "/performance/metrics",
            params={"load_test_run_id": run_id},
            headers=headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["window"]["mode"] == "load_test_run"
        assert body["window"]["load_test_run_id"] == run_id
        assert body["throughput"]["total_agent_run_count"] == 1
    finally:
        raw_db_session.execute(delete(AgentRun).where(AgentRun.id == agent_run.id))
        raw_db_session.execute(delete(LoadTestRun).where(LoadTestRun.id == run_id))
        raw_db_session.commit()


def test_load_test_run_not_completed_yet_is_rejected(client, admin_employee, raw_db_session):
    run = LoadTestRun(started_at=datetime(2024, 1, 1), config_json="{}", git_commit="e" * 40)
    raw_db_session.add(run)
    raw_db_session.flush()
    run_id = run.id
    raw_db_session.commit()

    try:
        response = client.get(
            "/performance/metrics",
            params={"load_test_run_id": run_id},
            headers={"X-Debug-Employee-Id": str(admin_employee.id)},
        )
        assert response.status_code == 400
    finally:
        raw_db_session.execute(delete(LoadTestRun).where(LoadTestRun.id == run_id))
        raw_db_session.commit()


def test_load_test_run_not_found_is_404(client, admin_employee):
    response = client.get(
        "/performance/metrics",
        params={"load_test_run_id": 99_999_999},
        headers={"X-Debug-Employee-Id": str(admin_employee.id)},
    )
    assert response.status_code == 404
