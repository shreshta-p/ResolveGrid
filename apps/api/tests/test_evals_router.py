"""Integration tests for Phase 10 Task 8's read-only admin eval API
(`resolvegrid_api.routers.evals`): `GET /evals/runs` and
`GET /evals/runs/{id}`.

Structure mirrors `test_approvals_router.py`: a module-scoped `TestClient`
fixture reusing `main.py`'s real lifespan, employee/role fixtures built via
the committing `raw_db_session` fixture (the HTTP request under test runs
its own, separate `get_db()` session/connection), and everything created is
cleaned up explicitly afterward.

Seeding choice (per the task's own instruction to weigh cost vs. purpose):
this file seeds real `EvalRun`/`EvalCaseResult` rows directly via the ORM
rather than invoking `eval_worker.run_eval_suite` for real -- that function
drives the real chat/tool-invocation graphs against live Ollama/Postgres
and is already covered end-to-end by `test_eval_worker.py`. This file's own
job is testing the API layer (authz boundary, response shape, pass-rate
math) against KNOWN data, which direct ORM seeding gives with exact,
hand-computed control over pass/fail counts per dimension -- reusing the
real batch runner here would be slower, less deterministic (real judge/LLM
calls), and would re-prove something `test_eval_worker.py` already proves.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from resolvegrid_api.main import app
from resolvegrid_api.models import RoleAssignment
from resolvegrid_api.models.evaluation import EvalCaseResult, EvalRun
from resolvegrid_api.models.org import Employee


@pytest.fixture(scope="module")
def client():
    # Module-scoped -- mirrors test_approvals_router.py's/test_tools_router.py's
    # own `client` fixture precedent.
    with TestClient(app) as test_client:
        yield test_client


def _make_employee(session, suffix: str) -> Employee:
    employee = Employee(
        display_name=f"Evals Router Employee {suffix}",
        email=f"evals.router.{suffix}@example.test",
        title="IT Analyst",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    session.add(employee)
    session.flush()
    return employee


@pytest.fixture
def admin_employee(raw_db_session):
    """Global-admin principal -- the shape `authorize(principal, "eval.view")`
    requires for the happy-path list/detail assertions below."""
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
    """A plain employee with NO role grants at all -- `eval.view` must deny
    them outright (staff-only action, no self-scoped downgrade), matching
    `test_approvals_router.py`'s identical `no_role_employee` fixture."""
    employee = _make_employee(raw_db_session, "norole")
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(Employee).where(Employee.id == employee.id))
        raw_db_session.commit()


@pytest.fixture
def seeded_eval_run(raw_db_session):
    """A real `EvalRun` with a hand-controlled set of `EvalCaseResult` rows
    across two dimensions, chosen so the pass-rate math has one obvious,
    hand-computable answer per dimension and overall:
    - chat: 2 passed, 1 failed -> 2/3
    - retrieval: 1 passed, 1 failed -> 1/2
    - overall: 3 passed, 2 failed -> 3/5

    Also includes one row with `details_json=None` (a grader that recorded
    no structured details) to confirm the API's None-handling doesn't choke,
    and one row with a real nested `details_json` string to confirm it comes
    back as a parsed dict, not a raw string the frontend would have to
    re-parse.
    """
    run = EvalRun(
        dataset_version="v2",
        prompt_version="v1",
        workflow_version="chat-tool-graphs-v1",
        retriever_version="hybrid-rrf-v1",
        reranker_version="disabled",
        embeddings_version="nomic-embed-text-v1",
        generation_model_version="qwen3:14b",
        judge_version="v1",
        tool_schema_version="1.0.0",
        git_commit="deadbeef",
        status="completed",
    )
    raw_db_session.add(run)
    raw_db_session.flush()

    raw_db_session.add_all(
        [
            EvalCaseResult(
                eval_run_id=run.id,
                case_id="chat.greeting.001",
                dimension="chat",
                passed=True,
                score=None,
                grader_type="deterministic",
                details_json=json.dumps({"checks": [{"passed": True}]}),
            ),
            EvalCaseResult(
                eval_run_id=run.id,
                case_id="chat.greeting.002",
                dimension="chat",
                passed=True,
                score=None,
                grader_type="deterministic",
                details_json=None,
            ),
            EvalCaseResult(
                eval_run_id=run.id,
                case_id="chat.greeting.003",
                dimension="chat",
                passed=False,
                score=None,
                grader_type="deterministic",
                details_json=json.dumps({"error": "intent mismatch"}),
            ),
            EvalCaseResult(
                eval_run_id=run.id,
                case_id="retrieval.vpn.001",
                dimension="retrieval",
                passed=True,
                score=1.0,
                grader_type="deterministic",
                details_json=json.dumps({"recall_at_k": 1.0}),
            ),
            EvalCaseResult(
                eval_run_id=run.id,
                case_id="retrieval.vpn.002",
                dimension="retrieval",
                passed=False,
                score=0.0,
                grader_type="deterministic",
                details_json=json.dumps({"recall_at_k": 0.0, "leaked_chunk_ids": [42]}),
            ),
        ]
    )
    raw_db_session.commit()
    try:
        yield run
    finally:
        raw_db_session.execute(delete(EvalCaseResult).where(EvalCaseResult.eval_run_id == run.id))
        raw_db_session.execute(delete(EvalRun).where(EvalRun.id == run.id))
        raw_db_session.commit()


def test_list_eval_runs_denied_for_plain_employee(client, no_role_employee, seeded_eval_run):
    response = client.get("/evals/runs", headers={"X-Debug-Employee-Id": str(no_role_employee.id)})
    assert response.status_code == 403


def test_get_eval_run_denied_for_plain_employee(client, no_role_employee, seeded_eval_run):
    response = client.get(
        f"/evals/runs/{seeded_eval_run.id}", headers={"X-Debug-Employee-Id": str(no_role_employee.id)}
    )
    assert response.status_code == 403


def test_list_eval_runs_returns_newest_first_with_pass_rate_summary(client, admin_employee, seeded_eval_run):
    response = client.get("/evals/runs", headers={"X-Debug-Employee-Id": str(admin_employee.id)})
    assert response.status_code == 200
    rows = response.json()

    matching = next((row for row in rows if row["id"] == seeded_eval_run.id), None)
    assert matching is not None
    assert matching["dataset_version"] == "v2"
    assert matching["status"] == "completed"
    assert matching["git_commit"] == "deadbeef"

    summary = matching["summary"]
    assert summary["dimensions"]["chat"] == {"passed_count": 2, "total_count": 3, "pass_rate": pytest.approx(2 / 3)}
    assert summary["dimensions"]["retrieval"] == {
        "passed_count": 1,
        "total_count": 2,
        "pass_rate": pytest.approx(0.5),
    }
    assert summary["overall_passed_count"] == 3
    assert summary["overall_total_count"] == 5
    assert summary["overall_pass_rate"] == pytest.approx(3 / 5)

    # Newest-first ordering: this run should not be sorted after any run
    # with an earlier started_at than its own -- weakest safe assertion
    # given other tests/processes may have created other EvalRun rows
    # concurrently, but enough to catch a reversed/unsorted query.
    started_ats = [row["started_at"] for row in rows if row["started_at"] is not None]
    assert started_ats == sorted(started_ats, reverse=True)


def test_get_eval_run_returns_full_case_breakdown_with_parsed_details(client, admin_employee, seeded_eval_run):
    response = client.get(
        f"/evals/runs/{seeded_eval_run.id}", headers={"X-Debug-Employee-Id": str(admin_employee.id)}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["id"] == seeded_eval_run.id
    assert body["dataset_version"] == "v2"
    assert body["prompt_version"] == "v1"
    assert body["workflow_version"] == "chat-tool-graphs-v1"
    assert body["retriever_version"] == "hybrid-rrf-v1"
    assert body["reranker_version"] == "disabled"
    assert body["embeddings_version"] == "nomic-embed-text-v1"
    assert body["generation_model_version"] == "qwen3:14b"
    assert body["judge_version"] == "v1"
    assert body["tool_schema_version"] == "1.0.0"
    assert body["status"] == "completed"
    assert body["started_at"] is not None

    assert body["summary"]["overall_passed_count"] == 3
    assert body["summary"]["overall_total_count"] == 5

    case_results = {row["case_id"]: row for row in body["case_results"]}
    assert len(case_results) == 5

    passing_with_details = case_results["chat.greeting.001"]
    assert passing_with_details["dimension"] == "chat"
    assert passing_with_details["passed"] is True
    assert passing_with_details["grader_type"] == "deterministic"
    # details_json parsed back to a real dict, not a raw JSON string.
    assert isinstance(passing_with_details["details"], dict)
    assert passing_with_details["details"] == {"checks": [{"passed": True}]}

    no_details_case = case_results["chat.greeting.002"]
    assert no_details_case["details"] is None

    failing_retrieval_case = case_results["retrieval.vpn.002"]
    assert failing_retrieval_case["passed"] is False
    assert failing_retrieval_case["score"] == 0.0
    assert failing_retrieval_case["details"] == {"recall_at_k": 0.0, "leaked_chunk_ids": [42]}


def test_get_eval_run_missing_id_returns_404(client, admin_employee):
    response = client.get("/evals/runs/999999999", headers={"X-Debug-Employee-Id": str(admin_employee.id)})
    assert response.status_code == 404


def test_pass_rate_is_none_not_a_crash_when_a_run_has_zero_case_results(client, admin_employee, raw_db_session):
    """Regression guard for the division-by-zero handling `_pass_rate_summary`/
    `_rate` must apply: a real `EvalRun` with NO `EvalCaseResult` rows at all
    (e.g. a run that errored before grading any case) must report
    `overall_pass_rate=None`, never raise `ZeroDivisionError`."""
    run = EvalRun(
        dataset_version="v2",
        prompt_version="v1",
        workflow_version="chat-tool-graphs-v1",
        retriever_version="hybrid-rrf-v1",
        reranker_version="disabled",
        embeddings_version="nomic-embed-text-v1",
        generation_model_version="qwen3:14b",
        judge_version="v1",
        tool_schema_version="1.0.0",
        git_commit="deadbeef",
        status="error",
        error_message="boom",
    )
    raw_db_session.add(run)
    raw_db_session.commit()
    try:
        response = client.get(
            f"/evals/runs/{run.id}", headers={"X-Debug-Employee-Id": str(admin_employee.id)}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["case_results"] == []
        assert body["summary"] == {
            "dimensions": {},
            "overall_passed_count": 0,
            "overall_total_count": 0,
            "overall_pass_rate": None,
        }
    finally:
        raw_db_session.execute(delete(EvalRun).where(EvalRun.id == run.id))
        raw_db_session.commit()
