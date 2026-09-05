"""Integration tests for Phase 9 Task 7b's approver API
(`resolvegrid_api.routers.approvals`): `GET /approvals` and
`POST /approvals/{approval_id}/decide`.

This is the phase's core end-to-end proof: a real mutating tool
(`grant_vpn_access`) working proposal -> durable `ApprovalRequest` ->
approver decision -> resumed `interrupt()` -> executed mutation -> audit
trail, entirely through the real running app (Task 7a's
`POST /tools/{tool_name}/invoke` to create the pending request, this
router's `POST /approvals/{id}/decide` to resolve it), asserted against
real Postgres rows -- no mocking of the mutation path.

Docker/Postgres required (same requirement as every other DB-touching test
in this file's siblings, e.g. `test_tools_router.py`/`test_mutation_
execution.py`).

Session/visibility and cleanup conventions mirror `test_tools_router.py`
exactly: employees/role grants are created via the committing
`raw_db_session` fixture (the HTTP request under test runs its own,
separate `get_db()` session/connection), and everything this file creates
is cleaned up explicitly afterward -- EXCEPT the `requester_employee`
fixture's `Employee` row, which is deliberately never deleted (unique email
suffix per test run instead): a successful approve test's `execute_mutation`
call writes a permanent, append-only `AuditLog` row whose `actor_id` FK
references that employee (no cascade delete -- see `audit.py`'s module
docstring), matching `test_mutation_execution.py`'s
`test_execute_mutation_closes_the_concurrent_replay_race`'s already-
established "AuditLog rows outlive this test by design" precedent.

Server-side reject-comment validation: `approvals.py`'s `decide_approval`
enforces "a non-empty comment is required to reject" server-side (422),
not just client-side in `apps/web/app/approvals/page.tsx` -- see that
router's docstring for why this defense-in-depth choice was made (an
approver's rejection reason is a real audit artifact, not a cosmetic UI
nicety). `test_decide_reject_without_comment_is_rejected_server_side`
below is this choice's regression test.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from resolvegrid_api.approval_service import compute_snapshot_hash
from resolvegrid_api.main import app
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest, AuditLog, RoleAssignment, ToolCall
from resolvegrid_api.models.org import Employee, EmployeeEntitlement, Entitlement
from resolvegrid_api.operational_adapters.entitlements import VPN_ENTITLEMENT_NAME

_ACTION_TYPE = "grant_vpn_access"
_GRANT_JUSTIFICATION = "approvals-router-test: onboarding contractor"


@pytest.fixture(scope="module")
def client():
    # Module-scoped -- mirrors test_tools_router.py's/test_chat_api.py's own
    # `client` fixture precedent: re-running main.py's lifespan per test
    # would work (setup() is documented+verified idempotent) but is
    # unnecessary cost.
    with TestClient(app) as test_client:
        yield test_client


def _make_employee(session, suffix: str) -> Employee:
    employee = Employee(
        display_name=f"Approvals Router Employee {suffix}",
        email=f"approvals.router.{suffix}@example.test",
        title="IT Analyst",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    session.add(employee)
    session.flush()
    return employee


def _delete_checkpoint_thread(thread_id: str) -> None:
    """Best-effort cleanup of a paused/resumed graph run's checkpoint rows
    -- mirrors `test_tools_router.py`'s `_delete_checkpoint_thread`, reusing
    `app.state.tool_invocation_graph`'s already-open checkpointer."""
    checkpointer = app.state.tool_invocation_graph.checkpointer
    asyncio.run(checkpointer.adelete_thread(thread_id))


@pytest.fixture
def requester_employee(raw_db_session):
    """The analyst who submits `grant_vpn_access` requests via
    `POST /tools/grant_vpn_access/invoke`. Deliberately NEVER deleted in
    teardown -- see module docstring's cleanup-conventions section for why
    (a permanent `AuditLog.actor_id` FK reference from any test that
    reaches a successful `execute_mutation` call). A fresh uuid suffix per
    test run avoids an email collision with a prior run's un-deleted row.
    """
    unique_suffix = uuid.uuid4().hex[:12]
    employee = Employee(
        display_name="Approvals Router Requester",
        email=f"approvals.router.requester.{unique_suffix}@example.test",
        title="IT Analyst",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    raw_db_session.add(employee)
    raw_db_session.flush()
    raw_db_session.add(RoleAssignment(employee_id=employee.id, role="analyst", scope="global"))
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == employee.id))
        raw_db_session.commit()


@pytest.fixture
def approver_employee(raw_db_session):
    """An employee holding a department-scoped "approver" grant -- the
    shape `authorize(principal, "approval.list"/"approval.decide")` requires
    for a non-admin caller to pass (see `packages/authz/policy.py`:
    `_STAFF_ONLY_ACTIONS` denies outright unless `department_ids` is
    non-empty, which only a `scope="department"` analyst/approver grant --
    or global admin -- produces). `scope_id` has no FK constraint
    (`RoleAssignment.scope_id` is a plain nullable int column), so an
    arbitrary value is fine here -- this phase's `GET /approvals` doesn't
    filter by it anyway (see `approvals.py`'s documented department-scoping
    judgment call).
    """
    employee = _make_employee(raw_db_session, "approver")
    raw_db_session.add(RoleAssignment(employee_id=employee.id, role="approver", scope="department", scope_id=1))
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(RoleAssignment).where(RoleAssignment.employee_id == employee.id))
        raw_db_session.execute(delete(Employee).where(Employee.id == employee.id))
        raw_db_session.commit()


@pytest.fixture
def no_role_employee(raw_db_session):
    """An employee with NO role grants at all -- `approval.list`/
    `approval.decide` must deny them outright (staff-only actions, no
    self-scoped downgrade)."""
    employee = _make_employee(raw_db_session, "norole")
    raw_db_session.commit()
    try:
        yield employee
    finally:
        raw_db_session.execute(delete(Employee).where(Employee.id == employee.id))
        raw_db_session.commit()


def _submit_grant_request(client, requester_employee) -> tuple[int, str]:
    """Submits a real `grant_vpn_access` invoke request as `requester_
    employee` and returns `(approval_request_id, thread_id)` -- the setup
    every test below needs to have a real, durably-paused pending request
    to decide on."""
    response = client.post(
        "/tools/grant_vpn_access/invoke",
        json={
            "params": {
                "employee_id": requester_employee.id,
                "justification": _GRANT_JUSTIFICATION,
            }
        },
        headers={"X-Debug-Employee-Id": str(requester_employee.id)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_approval"
    return body["approval_request_id"], body["thread_id"]


def test_list_approvals_visible_to_approver_denied_for_plain_employee(
    client, requester_employee, approver_employee, no_role_employee, raw_db_session
):
    approval_request_id, thread_id = _submit_grant_request(client, requester_employee)
    try:
        response = client.get("/approvals", headers={"X-Debug-Employee-Id": str(approver_employee.id)})
        assert response.status_code == 200
        rows = response.json()
        matching = next((row for row in rows if row["id"] == approval_request_id), None)
        assert matching is not None
        assert matching["action_type"] == "grant_vpn_access"
        assert matching["status"] == "pending"
        assert matching["action_params"] == {
            "employee_id": requester_employee.id,
            "justification": _GRANT_JUSTIFICATION,
        }
        assert matching["requested_by_id"] == requester_employee.id

        denied = client.get("/approvals", headers={"X-Debug-Employee-Id": str(no_role_employee.id)})
        assert denied.status_code == 403
    finally:
        _delete_checkpoint_thread(thread_id)
        raw_db_session.execute(delete(ToolCall).where(ToolCall.approval_request_id == approval_request_id))
        raw_db_session.execute(
            delete(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
        )
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


def test_approve_valid_pending_request_grants_real_entitlement_and_records_audit_trail(
    client, requester_employee, approver_employee, raw_db_session
):
    approval_request_id, thread_id = _submit_grant_request(client, requester_employee)
    try:
        response = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "approved", "comment": "looks legitimate"},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approved"
        assert body["status"] == "approved"
        assert body.get("resume_error") is None
        assert body["tool_invocation_result"]["status"] == "success"
        assert body["tool_invocation_result"]["error"] is None

        row = raw_db_session.get(ApprovalRequest, approval_request_id)
        assert row.status == "approved"

        decisions = (
            raw_db_session.execute(
                select(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
            )
            .scalars()
            .all()
        )
        assert len(decisions) == 1
        assert decisions[0].decision == "approved"
        assert decisions[0].approver_id == approver_employee.id
        assert decisions[0].comment == "looks legitimate"

        # The real, granted end-to-end proof: an actual EmployeeEntitlement
        # row now exists for the "VPN Access" entitlement.
        entitlement = raw_db_session.execute(
            select(Entitlement).where(Entitlement.name == VPN_ENTITLEMENT_NAME)
        ).scalar_one()
        grants = (
            raw_db_session.execute(
                select(EmployeeEntitlement).where(
                    EmployeeEntitlement.employee_id == requester_employee.id,
                    EmployeeEntitlement.entitlement_id == entitlement.id,
                    EmployeeEntitlement.revoked_at.is_(None),
                )
            )
            .scalars()
            .all()
        )
        assert len(grants) == 1

        idempotency_key = f"approval:{approval_request_id}"
        success_calls = (
            raw_db_session.execute(
                select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "success")
            )
            .scalars()
            .all()
        )
        assert len(success_calls) == 1
        assert success_calls[0].tool_name == "grant_vpn_access"

        audit_rows = (
            raw_db_session.execute(
                select(AuditLog).where(
                    AuditLog.action == "tool.grant_vpn_access",
                    AuditLog.entity_id == grants[0].id,
                )
            )
            .scalars()
            .all()
        )
        assert len(audit_rows) == 1
        assert audit_rows[0].actor_type == "agent"
        assert audit_rows[0].actor_id == requester_employee.id
    finally:
        _delete_checkpoint_thread(thread_id)
        raw_db_session.execute(delete(ToolCall).where(ToolCall.approval_request_id == approval_request_id))
        raw_db_session.execute(
            delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == requester_employee.id)
        )
        raw_db_session.execute(
            delete(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
        )
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


def test_reject_request_does_not_grant_entitlement_or_execute_mutation(
    client, requester_employee, approver_employee, raw_db_session
):
    approval_request_id, thread_id = _submit_grant_request(client, requester_employee)
    try:
        response = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "rejected", "comment": "not a legitimate request"},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "rejected"
        assert body["status"] == "rejected"
        assert body["tool_invocation_result"] == {"status": "rejected", "output": None, "error": None}

        row = raw_db_session.get(ApprovalRequest, approval_request_id)
        assert row.status == "rejected"

        grants = (
            raw_db_session.execute(
                select(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == requester_employee.id)
            )
            .scalars()
            .all()
        )
        assert grants == []

        # execute_mutation_fn is never called on the rejected path (see
        # make_execute_mutation_node's docstring) -- no ToolCall row at all
        # for this approval_request_id, not even an error one.
        tool_calls = (
            raw_db_session.execute(select(ToolCall).where(ToolCall.approval_request_id == approval_request_id))
            .scalars()
            .all()
        )
        assert tool_calls == []
    finally:
        _delete_checkpoint_thread(thread_id)
        raw_db_session.execute(
            delete(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
        )
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


def test_decide_already_decided_request_returns_409_and_does_not_duplicate(
    client, requester_employee, approver_employee, raw_db_session
):
    approval_request_id, thread_id = _submit_grant_request(client, requester_employee)
    try:
        first = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "approved", "comment": None},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert first.status_code == 200

        second = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "approved", "comment": None},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert second.status_code == 409

        decisions = (
            raw_db_session.execute(
                select(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
            )
            .scalars()
            .all()
        )
        assert len(decisions) == 1

        idempotency_key = f"approval:{approval_request_id}"
        success_calls = (
            raw_db_session.execute(
                select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "success")
            )
            .scalars()
            .all()
        )
        assert len(success_calls) == 1
    finally:
        _delete_checkpoint_thread(thread_id)
        raw_db_session.execute(delete(ToolCall).where(ToolCall.approval_request_id == approval_request_id))
        raw_db_session.execute(
            delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == requester_employee.id)
        )
        raw_db_session.execute(
            delete(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
        )
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


def test_decide_expired_request_returns_409_without_executing_mutation(
    client, requester_employee, approver_employee, raw_db_session
):
    """Directly constructs an already-expired `ApprovalRequest` row (no real
    graph run behind it needed -- the expiry check runs, and this endpoint
    returns 409, strictly BEFORE any attempt to resume a graph thread, so
    `agent_run_id=None` here is fine; see `approvals.py`'s `decide_approval`
    docstring for the checked-order)."""
    params = {"employee_id": requester_employee.id, "justification": _GRANT_JUSTIFICATION}
    action_params_json = json.dumps(params, sort_keys=True)
    expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    snapshot_hash = compute_snapshot_hash(
        action_type=_ACTION_TYPE,
        params=params,
        actor=requester_employee.id,
        evidence_refs=None,
        risk_context="high",
        expires_at=expires_at,
    )
    approval_row = ApprovalRequest(
        agent_run_id=None,
        action_type=_ACTION_TYPE,
        action_params_json=action_params_json,
        bound_evidence_refs_json=None,
        risk_context="high",
        status="pending",
        snapshot_hash=snapshot_hash,
        requested_by_id=requester_employee.id,
        expires_at=expires_at,
    )
    raw_db_session.add(approval_row)
    raw_db_session.commit()
    approval_request_id = approval_row.id

    try:
        response = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "approved", "comment": None},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert response.status_code == 409

        row = raw_db_session.get(ApprovalRequest, approval_request_id)
        assert row.status == "pending"  # untouched by the rejected decide attempt

        grants = (
            raw_db_session.execute(
                select(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == requester_employee.id)
            )
            .scalars()
            .all()
        )
        assert grants == []

        decisions = (
            raw_db_session.execute(
                select(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
            )
            .scalars()
            .all()
        )
        assert decisions == []
    finally:
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


def test_decide_reject_without_comment_is_rejected_server_side(
    client, requester_employee, approver_employee, raw_db_session
):
    approval_request_id, thread_id = _submit_grant_request(client, requester_employee)
    try:
        response = client.post(
            f"/approvals/{approval_request_id}/decide",
            json={"decision": "rejected", "comment": None},
            headers={"X-Debug-Employee-Id": str(approver_employee.id)},
        )
        assert response.status_code == 422

        row = raw_db_session.get(ApprovalRequest, approval_request_id)
        assert row.status == "pending"  # rejected server-side before any write happened
    finally:
        _delete_checkpoint_thread(thread_id)
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()
