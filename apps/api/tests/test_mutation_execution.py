"""Real-Postgres tests for Phase 9 Task 6's `resolvegrid_api.mutation_execution`
(`execute_mutation` + `execute_readonly_tool`).

Most tests use the transactional `db_session` fixture (rolled back at the
end -- see conftest.py) and call `execute_mutation`/`execute_readonly_tool`
directly with that session; since neither function commits internally
(see `mutation_execution.py`'s module docstring for why), everything a test
writes -- including via two SEQUENTIAL calls sharing the same session -- is
visible to later reads/asserts within that same session/transaction, no
explicit commit needed (mirrors `test_audit.py`'s own established pattern).

The one exception is `test_execute_mutation_closes_the_concurrent_replay_race`,
which needs two REAL, separately-committing sessions on separate DB
connections to actually exercise the `pg_advisory_xact_lock` critical
section -- that test uses `resolvegrid_api.db.session_factory()` directly
(one fresh session per thread) plus `raw_db_session` for setup/cleanup,
mirroring `test_approval_service.py`'s
`test_request_approval_for_agent_closes_the_concurrent_insert_race`
`threading.Barrier` technique.

Docker/Postgres required: these tests need the real `resolvegrid` Postgres
container running (see repo's `docker-compose.yml` / `Makefile`).
"""

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from resolvegrid_api.approval_service import compute_snapshot_hash
from resolvegrid_api.audit import verify_chain_integrity
from resolvegrid_api.db import session_factory
from resolvegrid_api.models import ApprovalRequest, AuditLog, ToolCall
from resolvegrid_api.models.org import Employee, EmployeeEntitlement
from resolvegrid_api.mutation_execution import (
    ApprovalExpiredError,
    ApprovalNotDecidedError,
    ApprovalNotFoundError,
    ApprovalParamsMismatchError,
    ApprovalTamperError,
    execute_mutation,
    execute_readonly_tool,
)

_ACTION_TYPE = "grant_vpn_access"


def _make_employee(session, suffix: str) -> Employee:
    employee = Employee(
        display_name=f"Mutation Exec Employee {suffix}",
        email=f"mutation.exec.{suffix}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    session.add(employee)
    session.flush()
    return employee


def _make_approval_request(
    session,
    *,
    agent_run_id: str,
    employee_id: int,
    justification: str = "new hire onboarding",
    status: str = "approved",
    expires_delta: timedelta = timedelta(hours=24),
) -> tuple[ApprovalRequest, dict]:
    """Directly constructs an `ApprovalRequest` row with a real, correctly
    -computed `snapshot_hash` (via the same `compute_snapshot_hash` Task 6
    itself re-derives) -- mirrors `test_approvals_tools_models.py`'s direct
    -construction style rather than routing through `request_approval_for_
    agent` (which self-manages its own session/commit -- see that module's
    docstring -- and isn't needed here since these tests only exercise
    Task 6's post-approval execution path, not Task 5's upsert).
    """
    params = {"employee_id": employee_id, "justification": justification}
    action_params_json = json.dumps(params, sort_keys=True)
    expires_at = datetime.now(timezone.utc) + expires_delta
    snapshot_hash = compute_snapshot_hash(
        action_type=_ACTION_TYPE,
        params=params,
        actor=employee_id,
        evidence_refs=None,
        risk_context="medium",
        expires_at=expires_at,
    )
    row = ApprovalRequest(
        agent_run_id=agent_run_id,
        action_type=_ACTION_TYPE,
        action_params_json=action_params_json,
        bound_evidence_refs_json=None,
        risk_context="medium",
        status=status,
        snapshot_hash=snapshot_hash,
        requested_by_id=employee_id,
        expires_at=expires_at,
    )
    session.add(row)
    session.flush()
    return row, params


def _active_grant_count(session, employee_id: int) -> int:
    return len(
        session.execute(
            select(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee_id)
        )
        .scalars()
        .all()
    )


def _tool_call_count(session, idempotency_key: str, status: str) -> int:
    return len(
        session.execute(
            select(ToolCall).where(
                ToolCall.idempotency_key == idempotency_key, ToolCall.status == status
            )
        )
        .scalars()
        .all()
    )


# --- execute_mutation: happy path -------------------------------------------


def test_execute_mutation_happy_path_grants_vpn_and_records_audit_trail(db_session):
    employee = _make_employee(db_session, "happy")
    row, params = _make_approval_request(db_session, agent_run_id="test-mutexec-happy-1", employee_id=employee.id)

    result = execute_mutation(
        db_session,
        approval_request_id=row.id,
        tool_name=_ACTION_TYPE,
        tool_params=params,
        actor_employee_id=employee.id,
    )

    assert result["status"] == "success"
    assert result["approval_request_id"] == row.id
    assert result["output"]["employee_id"] == employee.id

    assert _active_grant_count(db_session, employee.id) == 1

    idempotency_key = f"approval:{row.id}"
    success_calls = (
        db_session.execute(
            select(ToolCall).where(
                ToolCall.idempotency_key == idempotency_key, ToolCall.status == "success"
            )
        )
        .scalars()
        .all()
    )
    assert len(success_calls) == 1
    assert success_calls[0].tool_name == _ACTION_TYPE
    assert success_calls[0].approval_request_id == row.id

    audit_rows = (
        db_session.execute(select(AuditLog).where(AuditLog.action == f"tool.{_ACTION_TYPE}"))
        .scalars()
        .all()
    )
    assert len(audit_rows) >= 1
    assert audit_rows[-1].entity_type == "employee_entitlement"
    assert audit_rows[-1].actor_type == "agent"
    assert audit_rows[-1].actor_id == employee.id

    assert verify_chain_integrity(db_session) is True


# --- execute_mutation: tamper defense ---------------------------------------


def test_execute_mutation_raises_tamper_error_when_action_params_json_is_altered(db_session):
    employee = _make_employee(db_session, "tamper")
    row, params = _make_approval_request(db_session, agent_run_id="test-mutexec-tamper-1", employee_id=employee.id)

    # Simulate tampering: directly rewrite the stored params after creation,
    # without recomputing snapshot_hash -- exactly what an attacker with raw
    # DB access, or a bug elsewhere, would produce.
    row.action_params_json = json.dumps({"employee_id": employee.id, "justification": "TAMPERED"}, sort_keys=True)
    db_session.flush()

    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalTamperError"
    except ApprovalTamperError:
        pass

    assert _active_grant_count(db_session, employee.id) == 0

    idempotency_key = f"approval:{row.id}"
    assert _tool_call_count(db_session, idempotency_key, "error") == 1
    assert _tool_call_count(db_session, idempotency_key, "success") == 0

    error_call = (
        db_session.execute(
            select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "error")
        )
        .scalars()
        .one()
    )
    assert error_call.error_taxonomy_code == "ApprovalTamperError"


def test_execute_mutation_raises_params_mismatch_error_when_tool_params_argument_does_not_match_approved_params(
    db_session,
):
    """The real gap this task caught in the plan doc's literal instructions:
    `execute_mutation`'s snapshot-hash check alone only re-verifies the
    ROW's own stored fields against themselves -- it says nothing about
    whether the caller's `tool_params` argument matches what was actually
    approved. Without an explicit equality check, a caller could pass a
    valid approved approval_request_id alongside a DIFFERENT tool_params
    (e.g. a different employee_id) and silently execute an action that was
    never approved. See mutation_execution.py's `execute_mutation` docstring.

    Raises `ApprovalParamsMismatchError` specifically (a subclass of
    `ApprovalTamperError`, so still caught by a bare `except
    ApprovalTamperError:`) -- distinct from a genuine stored-row hash
    mismatch, per code review: the recorded `ToolCall.error_taxonomy_code`
    must let the audit trail distinguish "the row itself was altered on
    disk" from "a caller tried to slip different params through a valid
    approval id" (a confused-deputy attempt), since those are materially
    different security narratives. See
    `test_execute_mutation_tamper_and_params_mismatch_produce_distinguishable_error_taxonomy_codes`
    for the side-by-side proof.
    """
    employee = _make_employee(db_session, "mismatch")
    other_employee = _make_employee(db_session, "mismatch-victim")
    row, _params = _make_approval_request(
        db_session, agent_run_id="test-mutexec-mismatch-1", employee_id=employee.id
    )

    forged_params = {"employee_id": other_employee.id, "justification": "new hire onboarding"}

    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=forged_params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalParamsMismatchError"
    except ApprovalParamsMismatchError:
        pass
    # Also still catchable as the base ApprovalTamperError -- confirms the
    # subclass relationship a caller relying on the broader catch depends on.
    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=forged_params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalTamperError (via the subclass)"
    except ApprovalTamperError:
        pass

    assert _active_grant_count(db_session, employee.id) == 0
    assert _active_grant_count(db_session, other_employee.id) == 0

    idempotency_key = f"approval:{row.id}"
    error_codes = {
        call.error_taxonomy_code
        for call in db_session.execute(
            select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "error")
        )
        .scalars()
        .all()
    }
    assert error_codes == {"ApprovalParamsMismatchError"}


def test_execute_mutation_tamper_and_params_mismatch_produce_distinguishable_error_taxonomy_codes(db_session):
    """Side-by-side proof (per code review) that a genuine stored-row hash
    tamper and a caller tool_params/row mismatch record DIFFERENT
    `ToolCall.error_taxonomy_code` values, even though both are instances
    of `ApprovalTamperError` -- the audit trail must be able to tell these
    two materially different security narratives apart.
    """
    hash_tamper_employee = _make_employee(db_session, "distinguish-hash")
    hash_tamper_row, hash_tamper_params = _make_approval_request(
        db_session, agent_run_id="test-mutexec-distinguish-hash-1", employee_id=hash_tamper_employee.id
    )
    hash_tamper_row.action_params_json = json.dumps(
        {"employee_id": hash_tamper_employee.id, "justification": "TAMPERED"}, sort_keys=True
    )
    db_session.flush()

    mismatch_employee = _make_employee(db_session, "distinguish-mismatch")
    victim_employee = _make_employee(db_session, "distinguish-mismatch-victim")
    mismatch_row, _mismatch_params = _make_approval_request(
        db_session, agent_run_id="test-mutexec-distinguish-mismatch-1", employee_id=mismatch_employee.id
    )
    forged_params = {"employee_id": victim_employee.id, "justification": "new hire onboarding"}

    for row, tool_params, expected_error_cls in (
        (hash_tamper_row, hash_tamper_params, ApprovalTamperError),
        (mismatch_row, forged_params, ApprovalParamsMismatchError),
    ):
        try:
            execute_mutation(
                db_session,
                approval_request_id=row.id,
                tool_name=_ACTION_TYPE,
                tool_params=tool_params,
                actor_employee_id=None,
            )
            assert False, f"expected {expected_error_cls.__name__}"
        except expected_error_cls:
            pass

    hash_tamper_error_call = (
        db_session.execute(
            select(ToolCall).where(ToolCall.idempotency_key == f"approval:{hash_tamper_row.id}", ToolCall.status == "error")
        )
        .scalars()
        .one()
    )
    mismatch_error_call = (
        db_session.execute(
            select(ToolCall).where(ToolCall.idempotency_key == f"approval:{mismatch_row.id}", ToolCall.status == "error")
        )
        .scalars()
        .one()
    )

    assert hash_tamper_error_call.error_taxonomy_code == "ApprovalTamperError"
    assert mismatch_error_call.error_taxonomy_code == "ApprovalParamsMismatchError"
    assert hash_tamper_error_call.error_taxonomy_code != mismatch_error_call.error_taxonomy_code


# --- execute_mutation: expiry ------------------------------------------------


def test_execute_mutation_raises_expired_error_for_a_past_expires_at(db_session):
    employee = _make_employee(db_session, "expired")
    row, params = _make_approval_request(
        db_session,
        agent_run_id="test-mutexec-expired-1",
        employee_id=employee.id,
        expires_delta=timedelta(hours=-1),
    )

    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalExpiredError"
    except ApprovalExpiredError:
        pass

    assert _active_grant_count(db_session, employee.id) == 0
    idempotency_key = f"approval:{row.id}"
    error_call = (
        db_session.execute(
            select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "error")
        )
        .scalars()
        .one()
    )
    assert error_call.error_taxonomy_code == "ApprovalExpiredError"


# --- execute_mutation: not approved ------------------------------------------


def test_execute_mutation_raises_not_decided_error_for_pending_status(db_session):
    employee = _make_employee(db_session, "pending")
    row, params = _make_approval_request(
        db_session, agent_run_id="test-mutexec-pending-1", employee_id=employee.id, status="pending"
    )

    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalNotDecidedError"
    except ApprovalNotDecidedError:
        pass

    assert _active_grant_count(db_session, employee.id) == 0


def test_execute_mutation_raises_not_decided_error_for_rejected_status(db_session):
    employee = _make_employee(db_session, "rejected")
    row, params = _make_approval_request(
        db_session, agent_run_id="test-mutexec-rejected-1", employee_id=employee.id, status="rejected"
    )

    try:
        execute_mutation(
            db_session,
            approval_request_id=row.id,
            tool_name=_ACTION_TYPE,
            tool_params=params,
            actor_employee_id=employee.id,
        )
        assert False, "expected ApprovalNotDecidedError"
    except ApprovalNotDecidedError:
        pass

    assert _active_grant_count(db_session, employee.id) == 0


# --- execute_mutation: not found ---------------------------------------------


def test_execute_mutation_raises_not_found_for_unknown_approval_request_id(db_session):
    try:
        execute_mutation(
            db_session,
            approval_request_id=999_999_999,
            tool_name=_ACTION_TYPE,
            tool_params={"employee_id": 1, "justification": "x"},
            actor_employee_id=None,
        )
        assert False, "expected ApprovalNotFoundError"
    except ApprovalNotFoundError:
        pass


# --- execute_mutation: duplicate-replay guard (sequential) -------------------


def test_execute_mutation_duplicate_replay_does_not_create_a_second_grant_or_tool_call(db_session):
    employee = _make_employee(db_session, "replay")
    row, params = _make_approval_request(db_session, agent_run_id="test-mutexec-replay-1", employee_id=employee.id)

    first = execute_mutation(
        db_session,
        approval_request_id=row.id,
        tool_name=_ACTION_TYPE,
        tool_params=params,
        actor_employee_id=employee.id,
    )
    second = execute_mutation(
        db_session,
        approval_request_id=row.id,
        tool_name=_ACTION_TYPE,
        tool_params=params,
        actor_employee_id=employee.id,
    )

    assert first["output"] == second["output"]
    assert _active_grant_count(db_session, employee.id) == 1

    idempotency_key = f"approval:{row.id}"
    assert _tool_call_count(db_session, idempotency_key, "success") == 1


# --- execute_mutation: real concurrent replay race ---------------------------


def test_execute_mutation_closes_the_concurrent_replay_race(raw_db_session):
    """Real, multi-threaded proof that two concurrent `execute_mutation`
    calls for the SAME approval_request_id are serialized by the
    `pg_advisory_xact_lock(approval_request_id)` critical section, closing
    the check-then-act race a naive "SELECT existing success ToolCall,
    INSERT if none found" guard would leave open -- mirrors
    `test_approval_service.py`'s `test_request_approval_for_agent_closes_
    the_concurrent_insert_race` `threading.Barrier` technique.

    Uses `raw_db_session` for setup/cleanup (real commits, visible to the
    two independently-connected threads below) and a fresh
    `resolvegrid_api.db.session_factory()` session per thread -- each
    thread commits its own session after `execute_mutation` returns,
    exactly what a real caller (Task 7's approver-decide endpoint) is
    expected to do (see `mutation_execution.py`'s module docstring for why
    `execute_mutation` itself does not commit).
    """
    # A unique email per run (not a fixed literal): this test's employee row
    # is deliberately NOT deleted in the finally block below (see its
    # comment -- AuditLog rows permanently reference it), so a fixed email
    # would collide with residue from a prior run and fail this test's own
    # setup on any re-run.
    unique_suffix = uuid.uuid4().hex[:12]
    employee = Employee(
        display_name="Mutation Exec Employee race",
        email=f"mutation.exec.race.{unique_suffix}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    raw_db_session.add(employee)
    raw_db_session.flush()

    params = {"employee_id": employee.id, "justification": "race test"}
    action_params_json = json.dumps(params, sort_keys=True)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    snapshot_hash = compute_snapshot_hash(
        action_type=_ACTION_TYPE,
        params=params,
        actor=employee.id,
        evidence_refs=None,
        risk_context="medium",
        expires_at=expires_at,
    )
    approval_row = ApprovalRequest(
        agent_run_id="test-mutexec-race-1",
        action_type=_ACTION_TYPE,
        action_params_json=action_params_json,
        bound_evidence_refs_json=None,
        risk_context="medium",
        status="approved",
        snapshot_hash=snapshot_hash,
        requested_by_id=employee.id,
        expires_at=expires_at,
    )
    raw_db_session.add(approval_row)
    raw_db_session.commit()

    approval_request_id = approval_row.id
    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def call() -> None:
        try:
            barrier.wait(timeout=10)
            with session_factory() as thread_session:
                result = execute_mutation(
                    thread_session,
                    approval_request_id=approval_request_id,
                    tool_name=_ACTION_TYPE,
                    tool_params=params,
                    actor_employee_id=employee.id,
                )
                thread_session.commit()
                results.append(result)
        except Exception as exc:  # noqa: BLE001 -- capture any failure so
            # the assertions below can report it, rather than letting a
            # background-thread exception vanish silently.
            errors.append(exc)

    try:
        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors, f"unexpected errors from concurrent calls: {errors!r}"
        assert len(results) == 2

        grants = (
            raw_db_session.execute(
                select(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee.id)
            )
            .scalars()
            .all()
        )
        assert len(grants) == 1, (
            f"expected exactly 1 EmployeeEntitlement grant for a concurrently-replayed "
            f"execute_mutation call, found {len(grants)}"
        )

        idempotency_key = f"approval:{approval_request_id}"
        success_calls = (
            raw_db_session.execute(
                select(ToolCall).where(ToolCall.idempotency_key == idempotency_key, ToolCall.status == "success")
            )
            .scalars()
            .all()
        )
        assert len(success_calls) == 1, (
            f"expected exactly 1 success ToolCall row for a concurrently-replayed "
            f"execute_mutation call, found {len(success_calls)} -- the advisory-lock "
            f"critical section failed to serialize the concurrent calls"
        )
    finally:
        # Employee is deliberately NOT deleted here: execute_mutation's
        # success path writes a permanent, append-only AuditLog row
        # (actor_id -> this employee -- see audit.py's module docstring:
        # "no cascade delete") for each of the two threads' calls, and
        # AuditLog rows outlive this test by design, same as
        # test_audit.py's own established discipline of never assuming
        # (or forcing) an empty audit_log table. The employee/grant/
        # tool-call/approval-request rows are cleaned up; the harmless
        # employee residue matches this codebase's real permanent-audit
        # -trail behavior rather than fighting it.
        raw_db_session.execute(delete(ToolCall).where(ToolCall.approval_request_id == approval_request_id))
        raw_db_session.execute(
            delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee.id)
        )
        raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()


# --- execute_readonly_tool ---------------------------------------------------


def test_execute_readonly_tool_happy_path_for_lookup_employee_entitlements(db_session):
    from resolvegrid_api.operational_adapters.entitlements import grant_vpn_access

    employee = _make_employee(db_session, "readonly")
    grant_vpn_access(db_session, employee_id=employee.id, justification="new hire onboarding")

    result = execute_readonly_tool(
        db_session,
        tool_name="lookup_employee_entitlements",
        tool_params={"employee_id": employee.id},
    )

    assert result["status"] == "success"
    assert result["tool_name"] == "lookup_employee_entitlements"
    assert len(result["output"]) == 1
    assert result["output"][0]["entitlement_name"] == "VPN Access"

    calls = (
        db_session.execute(
            select(ToolCall).where(
                ToolCall.tool_name == "lookup_employee_entitlements", ToolCall.status == "success"
            )
        )
        .scalars()
        .all()
    )
    assert len(calls) >= 1
