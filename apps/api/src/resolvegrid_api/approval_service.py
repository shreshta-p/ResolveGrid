"""Phase 9 Task 5: the real `RequestApprovalFn` implementation injected
into `resolvegrid_agent_orchestration`'s (currently standalone, not yet
wired into `build_graph` -- see `graph.py`'s module docstring)
`request_approval` node.

Session lifecycle note: mirrors `agent_retrieval.py`'s established
"built once at app startup, closes over `apps/api` internals, opens its
own short-lived `Session` per call via `resolvegrid_api.db.session_factory()`"
pattern (see that module's docstring) -- required for exactly the same
reason: `AgentState` is checkpointed to Postgres after every graph
superstep, so a live SQLAlchemy `Session` can never be threaded through
state (see agent-orchestration's `state.py` module docstring), and
`request_approval_fn` (like `retrieve_fn`) is expected to be a closure
built once, not a per-request dependency.

Snapshot hash composition (READ THIS BEFORE CHANGING ANYTHING BELOW --
Task 6's `execute_mutation` re-derives this hash byte-for-byte from a
fetched `ApprovalRequest` row and compares it against the stored
`snapshot_hash` column as its tamper defense; any drift here is a live
Task 6 bug, not a Task 5 cosmetic detail):

    snapshot_hash = sha256(
        json.dumps(
            {
                "action_type": <str>,
                "params": <dict -- json.dumps's own recursive `sort_keys`
                    handles nested key ordering; no separate pre-sort is
                    needed or performed>,
                "actor": <int | None>,
                "evidence_refs": <list | None>,
                "risk_context": <str | None>,
                "expires_at": <str -- `datetime.isoformat()` of an
                    always-UTC-aware `datetime`, matching this codebase's
                    established `datetime.now(timezone.utc)` convention
                    (see chat.py's `AgentRun.completed_at` write) -- see
                    the "wall-clock idempotency hazard" note below for why
                    this must always be a value that is fixed *once* per
                    logical `ApprovalRequest`, never independently
                    recomputed>,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

See `compute_snapshot_hash` below for the literal implementation of this
composition -- `params`/`evidence_refs`/etc. are passed through exactly as
given (no re-typing, no round-tripping through `json.loads` first), so
Task 6's recomputation must reconstruct the identical Python values (an
`int` stays an `int`, not a numeric string, etc.) before calling this same
function, not just eyeball-match the JSON text.

Wall-clock idempotency hazard (a real bug this task caught while
implementing the plan doc's literal instructions, not a hypothetical):
LangGraph's `interrupt()` re-runs its ENTIRE enclosing node from the top on
every resume -- verified against the installed `langgraph==1.2.11` source
(`langgraph.types.interrupt`'s own docstring: "The graph resumes from the
start of the node, **re-executing** all logic"), since the HIL docs pages
this task was pointed at 404/redirect as of this writing. Concretely: the
`request_approval` node (`graph.py`) calls `request_approval_fn` BEFORE its
`interrupt()` call, so `request_approval_fn` runs again, in full, on every
resume AND on every restart-then-resume after a real process crash (the
Postgres checkpointer persists the paused state independent of process
lifetime). If this module computed `expires_at = datetime.now(timezone.utc)
+ _DEFAULT_APPROVAL_TTL` fresh on every one of those calls and then upserted
strictly "by exact `snapshot_hash` equality" (the plan doc's literal
wording), every resume/restart would compute a DIFFERENT `expires_at` (real
wall-clock time moves between calls), hence a DIFFERENT hash, hence no
existing row would ever be found by that hash lookup -- defeating the
entire point of this task by inserting a fresh duplicate `ApprovalRequest`
row on every single resume.

Fix: `request_approval_for_agent` below does NOT key its idempotency check
on `snapshot_hash` equality at all. It first looks up an existing row by
this request's *identity* -- `agent_run_id`, `action_type`,
`action_params_json`, `requested_by_id`, `bound_evidence_refs_json`,
`risk_context` (i.e. everything the caller's payload actually supplies,
deliberately excluding `expires_at`/`snapshot_hash`, both of which this
module itself derives). If a matching row already exists, its stored
`id`/`status`/`expires_at` are returned completely as-is -- `expires_at` is
never recomputed and `snapshot_hash` is never touched on this path, so no
wall-clock drift is possible. Only when no matching row exists does this
module pick a fresh `expires_at`, compute the hash over that concrete,
about-to-be-persisted value, and insert -- at which point `snapshot_hash`
is exactly what Task 6 needs it to be: a tamper-detection digest, computed
once at creation time over values that are then fixed and non-drifting for
the row's entire lifetime, re-derivable byte-for-byte from that row later.

Concurrent-insert race -- closed with a Postgres advisory lock, NOT a DB
uniqueness constraint (a real gap an earlier version of this module had --
recorded here, not silently fixed and forgotten): the identity tuple above
has no database-level uniqueness constraint. `snapshot_hash` IS unique
(Task 1's schema), but two truly concurrent callers with an IDENTICAL
payload compute DIFFERENT `expires_at` values (real wall-clock time
advances between them) and therefore DIFFERENT hashes -- so a naive "catch
the `snapshot_hash` UNIQUE-constraint violation" guard never actually
fires for this race; both concurrent inserts would succeed, producing two
`ApprovalRequest` rows for one logical approval. A composite unique
constraint over the identity fields was considered and rejected: several
of those columns are nullable (`agent_run_id`, `requested_by_id`,
`bound_evidence_refs_json`, `risk_context`), and Postgres unique
constraints treat NULL as distinct-from-every-other-NULL by default, so a
plain `UniqueConstraint` would silently fail to dedupe the common case
where several of those are `None` -- fixing that NULL-safety properly
needs a NULLS NOT DISTINCT / COALESCE-based expression index, plus
`action_params_json` is unbounded text that risks exceeding a btree
index's per-entry size limit for pathological large `params`. Instead,
`request_approval_for_agent` wraps its whole check-then-insert critical
section in a Postgres session-level advisory lock,
`pg_advisory_xact_lock(key)`, keyed on a hash of the SAME identity tuple
(`_advisory_lock_key`) -- this blocks a second concurrent caller with an
identical identity until the first caller's transaction (holding the
lock) commits or rolls back, at which point the second caller's own
`_find_existing_request` lookup (now running after the first row is
committed) finds the row the first caller just inserted, instead of racing
past the check and inserting a second one. The lock is transaction-scoped
(`_xact_`), so it releases automatically at `session.commit()` -- no
separate unlock call is needed or made. `_insert_new_request`'s
`IntegrityError` catch remains as residual defense-in-depth for a
genuinely different scenario (an actual `snapshot_hash` collision across
two DIFFERENT identities, astronomically unlikely but still guarded), not
as this race's primary defense -- see that function's docstring.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from resolvegrid_api.db import session_factory
from resolvegrid_api.models.approvals import ApprovalRequest

# Default approval validity window: 24 hours from creation. A fixed module
# constant for this phase, not yet driven by a real per-`action_type`
# `ApprovalPolicy.stages_json` lookup (Task 1's schema already has that
# table) -- flagged as real follow-up work for a later phase once staged
# approval policy is actually wired up, not silently assumed equivalent.
_DEFAULT_APPROVAL_TTL = timedelta(hours=24)


def compute_snapshot_hash(
    *,
    action_type: str,
    params: dict,
    actor: int | None,
    evidence_refs: list | None,
    risk_context: str | None,
    expires_at: datetime,
) -> str:
    """The exact sha256 composition documented in this module's docstring.

    `expires_at` must already be a concrete, decided `datetime` -- callers
    on the idempotent-return path (an existing row was found by identity)
    must never call this function at all; they return that row's own
    stored fields untouched instead (see `request_approval_for_agent`).
    This function is only ever called once, at the moment a brand new
    `ApprovalRequest` row is about to be inserted.

    Task 6's `execute_mutation` MUST import and call this exact function to
    re-derive a fetched row's hash for its tamper check -- it must NOT
    re-implement this composition by hand from the module docstring's
    description, since any incidental drift between a hand-rolled
    reimplementation and this function would itself produce false-positive
    tamper detections on perfectly legitimate, unmodified rows.
    """
    payload = {
        "action_type": action_type,
        "params": params,
        "actor": actor,
        "evidence_refs": evidence_refs,
        "risk_context": risk_context,
        "expires_at": expires_at.isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _find_existing_request(
    session: Session,
    *,
    agent_run_id: str | None,
    action_type: str,
    action_params_json: str,
    requested_by_id: int | None,
    bound_evidence_refs_json: str | None,
    risk_context: str | None,
) -> ApprovalRequest | None:
    """The identity lookup that makes `request_approval_for_agent`
    idempotent -- deliberately does NOT include `expires_at`/
    `snapshot_hash` (see module docstring's "wall-clock idempotency
    hazard" note for why those two must never participate in this check).

    Uses `.scalar_one_or_none()` (matching Task 3's
    `grant_vpn_access`/`ensure_vpn_entitlement_seeded` convention), not
    `.scalars().first()` -- `.first()` would silently return an arbitrary
    one of several matching rows if this identity ever somehow had more
    than one (which should be structurally impossible now that
    `request_approval_for_agent` serializes same-identity inserts via
    `pg_advisory_xact_lock` -- see module docstring's "Concurrent-insert
    race" section). `.scalar_one_or_none()` instead raises
    `MultipleResultsFound` in that case, surfacing real data corruption
    loudly instead of papering over it.
    """
    stmt = select(ApprovalRequest).where(
        ApprovalRequest.agent_run_id == agent_run_id,
        ApprovalRequest.action_type == action_type,
        ApprovalRequest.action_params_json == action_params_json,
        ApprovalRequest.requested_by_id == requested_by_id,
        ApprovalRequest.bound_evidence_refs_json == bound_evidence_refs_json,
        ApprovalRequest.risk_context == risk_context,
    )
    return session.execute(stmt).scalar_one_or_none()


def _advisory_lock_key(
    *,
    agent_run_id: str | None,
    action_type: str,
    action_params_json: str,
    requested_by_id: int | None,
    bound_evidence_refs_json: str | None,
    risk_context: str | None,
) -> int:
    """A stable, signed 64-bit integer derived from the same identity
    tuple `_find_existing_request` matches on -- used only as a
    `pg_advisory_xact_lock` key (see module docstring's "Concurrent-insert
    race" section), never persisted anywhere. Not `compute_snapshot_hash`
    itself (that hash's composition also includes `expires_at`, which this
    lock key must NOT depend on -- two concurrent callers with an
    identical *identity* but not-yet-decided, potentially different
    `expires_at` values must still map to the SAME lock key so they
    actually serialize against each other).

    Postgres advisory lock keys are a single `bigint` (signed 64-bit).
    `sha256`'s first 8 digest bytes, read as a big-endian unsigned 64-bit
    integer, are folded into that signed range the standard way
    (subtracting 2**64 when the top bit is set). A hash collision between
    two DIFFERENT identities here would only cause harmless, unnecessary
    extra serialization between unrelated requests (they'd briefly
    contend on the same lock key) -- never an incorrect result, since
    `_find_existing_request`'s actual column-equality check still runs
    inside the lock and is what determines correctness.
    """
    digest = hashlib.sha256(
        json.dumps(
            {
                "agent_run_id": agent_run_id,
                "action_type": action_type,
                "action_params_json": action_params_json,
                "requested_by_id": requested_by_id,
                "bound_evidence_refs_json": bound_evidence_refs_json,
                "risk_context": risk_context,
            },
            sort_keys=True,
        ).encode()
    ).digest()
    unsigned = int.from_bytes(digest[:8], "big")
    return unsigned - 2**64 if unsigned >= 2**63 else unsigned


def _insert_new_request(
    session: Session,
    *,
    agent_run_id: str | None,
    action_type: str,
    params: dict,
    action_params_json: str,
    actor: int | None,
    evidence_refs: list | None,
    bound_evidence_refs_json: str | None,
    risk_context: str | None,
) -> ApprovalRequest:
    """Picks a fresh `expires_at`, computes the snapshot hash over it, and
    inserts.

    Callers MUST already hold `_advisory_lock_key`'s Postgres advisory
    lock for this identity before calling this function (see
    `request_approval_for_agent` and module docstring's "Concurrent-insert
    race" section) -- that lock, not this function's `IntegrityError`
    catch, is what actually prevents two concurrent callers with an
    identical identity from both inserting. The `IntegrityError` catch
    here is narrower residual defense-in-depth for a genuinely different,
    much rarer scenario: an actual `snapshot_hash` collision between two
    DIFFERENT identities (astronomically unlikely for sha256, but still a
    real UNIQUE-constraint column, so still guarded rather than left to
    raise an unhandled 500).
    """
    expires_at = datetime.now(timezone.utc) + _DEFAULT_APPROVAL_TTL
    snapshot_hash = compute_snapshot_hash(
        action_type=action_type,
        params=params,
        actor=actor,
        evidence_refs=evidence_refs,
        risk_context=risk_context,
        expires_at=expires_at,
    )
    row = ApprovalRequest(
        agent_run_id=agent_run_id,
        action_type=action_type,
        action_params_json=action_params_json,
        bound_evidence_refs_json=bound_evidence_refs_json,
        risk_context=risk_context,
        status="pending",
        snapshot_hash=snapshot_hash,
        requested_by_id=actor,
        expires_at=expires_at,
    )
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = _find_existing_request(
            session,
            agent_run_id=agent_run_id,
            action_type=action_type,
            action_params_json=action_params_json,
            requested_by_id=actor,
            bound_evidence_refs_json=bound_evidence_refs_json,
            risk_context=risk_context,
        )
        if existing is not None:
            return existing
        raise
    session.refresh(row)
    return row


def request_approval_for_agent(payload: dict) -> dict:
    """The real `RequestApprovalFn` implementation (see
    `resolvegrid_agent_orchestration.graph.RequestApprovalFn`'s docstring
    for the exact payload shape this expects and the `ApprovalOutcome`
    shape this must return).

    Idempotent by construction (see module docstring): re-invoking this
    with an identical `payload` -- exactly what happens when the injected
    `request_approval` node re-executes on a LangGraph resume/restart --
    returns the SAME existing row's id/status/expires_at, never inserting
    a duplicate `ApprovalRequest`. This holds under genuine concurrency
    too (two overlapping calls with an identical payload), not just
    sequential re-execution -- see module docstring's "Concurrent-insert
    race" section for the `pg_advisory_xact_lock` mechanism this relies on,
    and `apps/api/tests/test_approval_service.py`'s
    `test_request_approval_for_agent_closes_the_concurrent_insert_race`
    for a real multi-threaded proof against Postgres.
    """
    action_type = payload["action_type"]
    params = payload.get("params") or {}
    actor = payload.get("actor")
    evidence_refs = payload.get("evidence_refs")
    risk_context = payload.get("risk_context")
    agent_run_id = payload.get("agent_run_id")

    action_params_json = json.dumps(params, sort_keys=True)
    bound_evidence_refs_json = (
        json.dumps(evidence_refs, sort_keys=True) if evidence_refs is not None else None
    )

    with session_factory() as session:
        # Acquire the identity's advisory lock BEFORE the identity lookup
        # below -- this is what actually closes the concurrent-insert race
        # (see module docstring's "Concurrent-insert race" section); a
        # second concurrent caller with the same identity blocks here
        # until the first caller's transaction commits (releasing this
        # transaction-scoped lock), then re-runs its own lookup and finds
        # the row the first caller just inserted.
        lock_key = _advisory_lock_key(
            agent_run_id=agent_run_id,
            action_type=action_type,
            action_params_json=action_params_json,
            requested_by_id=actor,
            bound_evidence_refs_json=bound_evidence_refs_json,
            risk_context=risk_context,
        )
        session.execute(select(func.pg_advisory_xact_lock(lock_key)))

        existing = _find_existing_request(
            session,
            agent_run_id=agent_run_id,
            action_type=action_type,
            action_params_json=action_params_json,
            requested_by_id=actor,
            bound_evidence_refs_json=bound_evidence_refs_json,
            risk_context=risk_context,
        )
        row = existing if existing is not None else _insert_new_request(
            session,
            agent_run_id=agent_run_id,
            action_type=action_type,
            params=params,
            action_params_json=action_params_json,
            actor=actor,
            evidence_refs=evidence_refs,
            bound_evidence_refs_json=bound_evidence_refs_json,
            risk_context=risk_context,
        )
        # Explicit commit even on the found-existing (no-write) path --
        # releases the advisory lock promptly rather than relying on the
        # `with session_factory()` context manager's implicit
        # close()-triggered rollback to eventually do so. A no-op if
        # `_insert_new_request` already committed.
        session.commit()

        return {
            "approval_request_id": row.id,
            "status": row.status,
            "expires_at": row.expires_at.isoformat(),
        }
