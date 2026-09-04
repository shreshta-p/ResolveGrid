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
`snapshot_hash` stays UNIQUE-constrained (Task 1's schema) as a
defense-in-depth backstop against a genuine concurrent-insert race for the
identical identity -- see `_insert_new_request`'s docstring.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
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
    """
    stmt = select(ApprovalRequest).where(
        ApprovalRequest.agent_run_id == agent_run_id,
        ApprovalRequest.action_type == action_type,
        ApprovalRequest.action_params_json == action_params_json,
        ApprovalRequest.requested_by_id == requested_by_id,
        ApprovalRequest.bound_evidence_refs_json == bound_evidence_refs_json,
        ApprovalRequest.risk_context == risk_context,
    )
    return session.execute(stmt).scalars().first()


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
    inserts. Guards against a genuine concurrent-insert race for the exact
    same identity (two near-simultaneous calls both missing
    `_find_existing_request`'s lookup before either commits) by catching
    the `snapshot_hash` UNIQUE-constraint violation... but a hash collision
    is not actually what a same-identity race would trigger here (each
    concurrent caller computes its OWN fresh `expires_at`, hence its own
    distinct hash) -- so the real guard against that race is re-running
    the identity lookup after an `IntegrityError` of any kind on insert,
    which catches both a literal hash collision (astronomically unlikely)
    and the more realistic case of a unique-constraint-adjacent conflict.
    This is defense-in-depth, not this task's primary correctness
    mechanism: LangGraph's own single-writer-per-node-per-checkpoint
    execution model means the actual re-execution scenario this task must
    handle (sequential resume/restart, not concurrent threads) never hits
    this race at all -- `_find_existing_request` alone handles it.
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
    a duplicate `ApprovalRequest`.
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

        return {
            "approval_request_id": row.id,
            "status": row.status,
            "expires_at": row.expires_at.isoformat(),
        }
