"""Real-Postgres integration tests for Phase 9 Task 5's real `RequestApprovalFn`
implementation, `resolvegrid_api.approval_service.request_approval_for_agent`,
plus unit tests of its `compute_snapshot_hash` helper.

Session-fixture note: `request_approval_for_agent` opens its OWN
`Session` internally via `resolvegrid_api.db.session_factory()` (mirrors
`agent_retrieval.retrieve_for_agent`'s established pattern -- see that
module's docstring) and really commits -- it does NOT take a `Session`
parameter the way `operational_adapters/entitlements.py`'s functions do.
That means `conftest.py`'s transactional `db_session` fixture (which rolls
back a *different* connection than the one `session_factory()` opens)
cannot isolate its writes. These tests use `raw_db_session` instead and
clean up their own rows by a distinctive `agent_run_id` per test in a
`finally` block -- the same discipline `test_checkpoint_restore.py`
established for real-commit, cross-connection state.

Docker/Postgres required: these tests need the real `resolvegrid` Postgres
container running (see repo's `docker-compose.yml` / `Makefile`).
"""

import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from resolvegrid_api.approval_service import compute_snapshot_hash, request_approval_for_agent
from resolvegrid_api.models.approvals import ApprovalRequest


def _cleanup(raw_db_session, agent_run_id: str) -> None:
    raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.agent_run_id == agent_run_id))
    raw_db_session.commit()


def _rows_for(raw_db_session, agent_run_id: str) -> list[ApprovalRequest]:
    return (
        raw_db_session.execute(select(ApprovalRequest).where(ApprovalRequest.agent_run_id == agent_run_id))
        .scalars()
        .all()
    )


# --- request_approval_for_agent: idempotent upsert -------------------------


def test_request_approval_for_agent_creates_a_pending_row_with_24h_expiry(raw_db_session):
    agent_run_id = "test-approval-svc-create-1"
    payload = {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 1, "justification": "new hire"},
        "actor": None,
        "evidence_refs": [],
        "risk_context": "medium",
        "agent_run_id": agent_run_id,
    }
    try:
        before = datetime.now(timezone.utc)
        outcome = request_approval_for_agent(payload)
        after = datetime.now(timezone.utc)

        assert outcome["status"] == "pending"
        assert isinstance(outcome["approval_request_id"], int)

        rows = _rows_for(raw_db_session, agent_run_id)
        assert len(rows) == 1
        row = rows[0]
        assert row.id == outcome["approval_request_id"]
        assert row.action_type == "grant_vpn_access"
        assert row.status == "pending"
        assert row.snapshot_hash  # a real sha256 hex digest was stored

        # expires_at is stored as a naive UTC-equivalent datetime (this
        # codebase's established DateTime-without-timezone convention --
        # see chat.py's AgentRun.completed_at) roughly 24h from creation.
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        assert before + timedelta(hours=23, minutes=59) <= expires_at <= after + timedelta(hours=24, minutes=1)
    finally:
        _cleanup(raw_db_session, agent_run_id)


def test_request_approval_for_agent_is_idempotent_across_repeated_identical_calls(raw_db_session):
    """The core Task 5 property: calling `request_approval_fn` twice with an
    identical payload (simulating the `request_approval` node re-executing
    from the top on a LangGraph resume/restart -- see graph.py's module
    docstring) must NOT create a second `ApprovalRequest` row.
    """
    agent_run_id = "test-approval-svc-idempotent-1"
    payload = {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 2, "justification": "role change"},
        "actor": None,
        "evidence_refs": ["chunk:1"],
        "risk_context": "high",
        "agent_run_id": agent_run_id,
    }
    try:
        first = request_approval_for_agent(payload)
        second = request_approval_for_agent(payload)
        third = request_approval_for_agent(payload)

        assert first["approval_request_id"] == second["approval_request_id"] == third["approval_request_id"]
        assert first["expires_at"] == second["expires_at"] == third["expires_at"]  # never recomputed
        assert first["status"] == second["status"] == third["status"] == "pending"

        rows = _rows_for(raw_db_session, agent_run_id)
        assert len(rows) == 1
    finally:
        _cleanup(raw_db_session, agent_run_id)


def test_request_approval_for_agent_closes_the_concurrent_insert_race(raw_db_session):
    """Real, multi-threaded proof that a genuine concurrent-insert race for
    an IDENTICAL payload is closed by the `pg_advisory_xact_lock`-based
    critical section in `request_approval_for_agent` -- code review on an
    earlier version of this module correctly flagged that the
    `_insert_new_request` `IntegrityError` catch alone never actually
    fires for this race (two concurrent callers compute different
    `expires_at` values, hence different `snapshot_hash` values, so no
    UNIQUE-constraint violation occurs and both inserts would silently
    succeed, producing two `ApprovalRequest` rows for one logical
    approval -- exactly what this phase's exit criteria "zero duplicate
    mutation side effects" exists to catch).

    Two real threads call `request_approval_for_agent` with the exact same
    payload, released simultaneously via a `Barrier` to maximize the
    chance of a genuine race window if the lock were not actually
    serializing them (a sequential call pair, by contrast, could pass even
    with no locking at all, since the first call's commit would already be
    visible before the second call's SELECT -- this test is deliberately
    concurrent, not sequential, to actually exercise the lock).
    """
    agent_run_id = "test-approval-svc-race-1"
    payload = {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 55, "justification": "race test"},
        "actor": None,
        "evidence_refs": [],
        "risk_context": "medium",
        "agent_run_id": agent_run_id,
    }
    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def call() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(request_approval_for_agent(payload))
        except Exception as exc:  # noqa: BLE001 -- capture any failure from
            # either thread so the assertions below can report it, rather
            # than letting a background-thread exception vanish silently.
            errors.append(exc)

    try:
        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors, f"unexpected errors from concurrent calls: {errors!r}"
        assert len(results) == 2
        assert results[0]["approval_request_id"] == results[1]["approval_request_id"]

        rows = _rows_for(raw_db_session, agent_run_id)
        assert len(rows) == 1, (
            f"expected exactly 1 ApprovalRequest row for identical concurrent "
            f"payloads, found {len(rows)} -- the advisory-lock critical "
            f"section failed to serialize the concurrent inserts"
        )
    finally:
        _cleanup(raw_db_session, agent_run_id)


def test_request_approval_for_agent_creates_separate_rows_for_different_identities(raw_db_session):
    agent_run_id_a = "test-approval-svc-distinct-a"
    agent_run_id_b = "test-approval-svc-distinct-b"
    payload_a = {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 3},
        "actor": None,
        "evidence_refs": None,
        "risk_context": "low",
        "agent_run_id": agent_run_id_a,
    }
    payload_b = {**payload_a, "agent_run_id": agent_run_id_b}
    try:
        outcome_a = request_approval_for_agent(payload_a)
        outcome_b = request_approval_for_agent(payload_b)

        assert outcome_a["approval_request_id"] != outcome_b["approval_request_id"]
        assert len(_rows_for(raw_db_session, agent_run_id_a)) == 1
        assert len(_rows_for(raw_db_session, agent_run_id_b)) == 1
    finally:
        _cleanup(raw_db_session, agent_run_id_a)
        _cleanup(raw_db_session, agent_run_id_b)


# --- compute_snapshot_hash: determinism + field sensitivity -----------------


def _base_hash_kwargs() -> dict:
    return {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 1, "justification": "new hire"},
        "actor": 7,
        "evidence_refs": ["chunk:1", "chunk:2"],
        "risk_context": "medium",
        "expires_at": datetime(2026, 9, 5, 0, 0, 0, tzinfo=timezone.utc),
    }


def test_compute_snapshot_hash_is_deterministic_for_identical_inputs():
    kwargs = _base_hash_kwargs()
    assert compute_snapshot_hash(**kwargs) == compute_snapshot_hash(**kwargs)


def test_compute_snapshot_hash_is_insensitive_to_params_key_order():
    kwargs = _base_hash_kwargs()
    reordered = {**kwargs, "params": {"justification": "new hire", "employee_id": 1}}
    assert compute_snapshot_hash(**kwargs) == compute_snapshot_hash(**reordered)


def test_compute_snapshot_hash_changes_with_action_type():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "action_type": "lookup_employee_entitlements"})
    assert baseline != changed


def test_compute_snapshot_hash_changes_with_params():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "params": {"employee_id": 999, "justification": "new hire"}})
    assert baseline != changed


def test_compute_snapshot_hash_changes_with_actor():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "actor": 8})
    assert baseline != changed


def test_compute_snapshot_hash_changes_with_evidence_refs():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "evidence_refs": ["chunk:1"]})
    assert baseline != changed


def test_compute_snapshot_hash_changes_with_risk_context():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "risk_context": "high"})
    assert baseline != changed


def test_compute_snapshot_hash_changes_with_expires_at():
    kwargs = _base_hash_kwargs()
    baseline = compute_snapshot_hash(**kwargs)
    changed = compute_snapshot_hash(**{**kwargs, "expires_at": kwargs["expires_at"] + timedelta(seconds=1)})
    assert baseline != changed
