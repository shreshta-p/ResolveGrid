"""Phase 9 Task 6: `execute_mutation` (the post-`interrupt()` execution path
for an approved mutating tool call) and `execute_readonly_tool` (the direct,
no-approval path for a non-mutating tool call).

Both functions take an externally-owned SQLAlchemy `Session` as their first
parameter -- unlike `approval_service.py`'s `request_approval_for_agent`
(which opens its own short-lived session via `session_factory()`; see that
module's docstring for why), this module follows
`operational_adapters/entitlements.py`'s convention instead: a plain
function that receives a `Session` and does NOT commit it. Reasoning: Task
7's approver-decide endpoint (this module's real caller, not yet built) is
expected to be a FastAPI route with its own request-scoped session that may
need to do more than just this one write in a single transaction (e.g. also
transition `ApprovalRequest.status` in the same commit); a function that
unilaterally commits an externally-supplied session would take that control
away from its caller. This module only ever `session.add(...)` +
`session.flush()`s -- exactly `grant_vpn_access`/`record_audit_event`'s own
established pattern -- and leaves `session.commit()` to the caller.

Snapshot-hash tamper check (`execute_mutation` step 2): this module imports
and calls `approval_service.compute_snapshot_hash` directly, never
reimplementing its composition -- `approval_service.py`'s own module
docstring explicitly documents this as a hard requirement, precisely
because any incidental drift between a hand-rolled reimplementation and
that function would itself produce false-positive tamper detections on
perfectly legitimate, unmodified rows.

`expires_at` timezone round-trip (a real, easy-to-get-wrong subtlety):
`ApprovalRequest.expires_at` is stored as a naive (timezone-less) Postgres
`TIMESTAMP` column (`Mapped[datetime]` with no `DateTime(timezone=True)`
override anywhere in this codebase's `Base`/model definitions -- confirmed
by reading `models/base.py` and `models/approvals.py`), even though the
value written into it at INSERT time (`approval_service._insert_new_
request`) was always a tz-aware `datetime.now(timezone.utc) + ...` value.
Postgres does not shift the wall-clock value for a `TIMESTAMP WITHOUT TIME
ZONE` column -- it stores exactly the naive wall-clock digits it was given
-- so a row fetched back has `expires_at.tzinfo is None` but the SAME
wall-clock digits as what was hashed at insert time.
`compute_snapshot_hash` calls `.isoformat()` directly on whatever
`datetime` it's given, and a naive `datetime`'s `.isoformat()` omits the
`+00:00` offset suffix that the ORIGINAL tz-aware value's `.isoformat()`
included -- so re-hashing the fetched row's naive `expires_at` AS-IS would
produce a DIFFERENT string, hence a DIFFERENT hash, hence every legitimate,
untampered row would incorrectly fail tamper verification. The fix
(`_reattach_utc` below) is the exact pattern `test_approval_service.py`'s
`test_request_approval_for_agent_creates_a_pending_row_with_24h_expiry`
already established for comparing a fetched `expires_at`: if
`.tzinfo is None`, `.replace(tzinfo=timezone.utc)` it before use --
`.replace()` does not shift the wall-clock digits, so the re-attached value
produces byte-for-byte the same `.isoformat()` string the original tz-aware
value did, and the recomputed hash matches for an untampered row.

Duplicate-replay concurrency (a real design decision, not skipped
silently): `ToolCall.idempotency_key` is indexed but deliberately NOT
unique-constrained (see that model's docstring -- a NULL-safe composite
uniqueness story was rejected there for the same reasons `ApprovalRequest`'s
identity tuple rejected a composite `UniqueConstraint` in
`approval_service.py`'s module docstring). A naive "SELECT for an existing
successful ToolCall by idempotency_key, INSERT a new one only if none
found" guard is exactly the check-then-act race `approval_service.py`
closed for its own idempotent upsert. This module closes the identical race
the identical way: `execute_mutation` acquires a Postgres transaction-scoped
advisory lock, `pg_advisory_xact_lock(approval_request_id)`, immediately
after fetching the `ApprovalRequest` row and before any check (tamper/
status/expiry) or the duplicate-replay SELECT -- so two concurrent callers
racing to execute the SAME `approval_request_id` are fully serialized
against each other for this function's entire body, not just the
duplicate-replay SELECT.

Unlike `approval_service._advisory_lock_key` (which folds a sha256 digest
of a *composite* identity tuple into a signed 64-bit lock key, because that
module's idempotency identity has no single primitive key), this module's
idempotency identity already IS a single primitive: `approval_request_id`
itself is an `int` primary key, already well within Postgres's signed
64-bit advisory-lock key range. No hashing/folding step is needed or used
here -- `pg_advisory_xact_lock(approval_request_id)` directly is both
simpler and, unlike a hashed key, can never collide with a different
`approval_request_id`'s lock (a real, if harmless, possibility for
`approval_service.py`'s hashed composite key, discussed in that module's
own docstring).

Why this concern is real, not hypothetical, for this specific node: Task 7
(not yet built) is expected to call `execute_mutation` from an approver-
decide HTTP endpoint. An HTTP client's retried/duplicated request (a
network timeout followed by a client retry, a double-click, a load
balancer's at-least-once retry policy) can genuinely deliver two concurrent
in-flight calls for the same `approval_request_id`. Separately, if a future
task wires `execute_mutation` in as a LangGraph node placed after
`request_approval`'s `interrupt()` boundary, `graph.py`'s own module
docstring already establishes that LangGraph re-runs an enclosing node's
entire body on every resume/restart -- so a node wrapping this function
would face the exact same re-execution hazard `request_approval_fn` was
built to tolerate. Both are real call shapes this function must be safe
under, not just sequential single-caller reuse. The end-to-end entitlement
grant itself is already double-protected by Task 3's `grant_vpn_access`
being independently idempotent (see that function's docstring) -- but
`ToolCall`/`AuditLog` are this system's audit trail, and two "success" rows
for one logical mutation would itself be a real correctness bug in that
trail, distinct from (and not excused by) the adapter-level idempotency of
the thing being audited. The lock closes that gap directly rather than
leaving it to the adapter's own, unrelated idempotency to coincidentally
paper over.

`execute_readonly_tool` deliberately has NO approval gate and NO
duplicate-replay guard -- both are meaningless for a read-only, side-
effect-free tool (there is nothing to protect against re-running; re-
running just re-reads the same data). It still writes a `ToolCall` row
per call (`status="success"`/`"error"`) purely for telemetry/audit
completeness, per this task's brief.

Dispatch design: a small `tool_name -> handler` dict for each of
`execute_mutation`/`execute_readonly_tool`, not a generic plugin/registry
system. This phase has exactly one mutating tool (`grant_vpn_access`) and
one read-only tool (`lookup_employee_entitlements`) -- building an
extensible plugin architecture for two known call sites would be premature
abstraction this codebase's established convention (see e.g.
`tool_execution.py`'s docstring: modules here solve exactly the problem in
front of them, not a anticipated future one). Adding a third tool later
means adding one dict entry and one small `_dispatch_*` function, which is
easy to do without redesigning anything.

LangGraph node wrappers: deliberately NOT added by this task. Task 5's
`make_request_approval_node` is a standalone node builder specifically
because wiring it into `build_graph` requires conditional routing
(`ToolContract.requires_approval`-based branching) that doesn't exist yet
in this graph -- see `graph.py`'s module docstring's "Deliberate scope
limit" section, explicitly assigned to Task 6/select_tool routing work.
Writing a thin `make_execute_mutation_node`/`make_execute_readonly_tool_node`
pair here, with no real router to plug them into, would mean guessing at
that not-yet-built routing contract from outside its scope -- the same
reasoning Task 5 already used to justify staying standalone. `execute_
mutation`/`execute_readonly_tool` are therefore plain, directly-testable
`apps/api` functions for now; Task 7's approver-decide endpoint is their
real caller for this phase.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from resolvegrid_api.approval_service import compute_snapshot_hash
from resolvegrid_api.audit import record_audit_event
from resolvegrid_api.models import ApprovalRequest, ToolCall
from resolvegrid_api.models.org import EmployeeEntitlement
from resolvegrid_api.operational_adapters.entitlements import (
    ensure_vpn_entitlement_seeded,
    grant_vpn_access,
    lookup_employee_entitlements,
)
from resolvegrid_contracts.tools import TOOL_REGISTRY


class MutationExecutionError(Exception):
    """Base class for every typed error this module raises.

    Mirrors `tool_execution.py`'s `ToolNotAllowedError`/`ToolValidationError`
    precedent (plain `Exception` subclasses, no custom `__init__`/error-code
    payload) -- this codebase's established shape for a small, closed set
    of typed error conditions a caller may want to catch individually.
    """


class ApprovalNotFoundError(MutationExecutionError):
    """No `ApprovalRequest` row exists for the given `approval_request_id`.

    Raised before any DB write, deliberately: a `ToolCall` row can't be
    linked to a nonexistent `ApprovalRequest` (`ToolCall.approval_request_id`
    is a real FK), so there is nothing safe to log here -- an invalid id is
    a caller bug (Task 7's endpoint should never construct one), not a
    security-relevant event worth an audit trail entry of its own.
    """


class ApprovalTamperError(MutationExecutionError):
    """The `ApprovalRequest` row's recomputed snapshot hash does not match
    its stored `snapshot_hash` -- the row's bound fields were altered after
    creation (e.g. direct DB tampering, a bug elsewhere mutating the row).
    "Shouldn't be reachable" per the plan doc, but this is the actual
    defense plan.md requires, not a formality -- see module docstring.
    """


class ApprovalNotDecidedError(MutationExecutionError):
    """The `ApprovalRequest`'s `status` is not `"approved"` (still
    `"pending"`, or `"rejected"`, or -- once a later phase's expiry-sweep
    job exists -- `"expired"`). Covers both "nobody has decided yet" and
    "somebody explicitly rejected it" with one error type, mirroring
    `tool_execution.py`'s `ToolNotAllowedError` precedent of not leaking a
    finer-grained reason than the caller needs to act on (a caller of
    `execute_mutation` should not attempt to distinguish "still pending" -
    retry later - from "rejected" - never retry - from this exception type
    alone; it should consult `row.status`/`ApprovalRequest` directly for
    that, matching how Task 7's endpoint will already know the current
    status before ever calling this function).
    """


class ApprovalExpiredError(MutationExecutionError):
    """The `ApprovalRequest` is `"approved"` but its `expires_at` has
    already passed. Checked only AFTER the status check (see
    `execute_mutation`'s docstring for why that order -- matching the plan
    doc's literal step ordering -- is also the more informative choice:
    today nothing sets `status="expired"` on a background sweep, so an
    unapproved-and-expired row would otherwise misleadingly raise this
    instead of `ApprovalNotDecidedError`, which is the more accurate "this
    was never approved at all" signal for that row).
    """


class UnknownMutationToolError(MutationExecutionError):
    """`tool_name` has no dispatch handler registered in this module.

    Should be unreachable in normal operation -- Task 4's
    `available_tools_for_principal`/`select_tool` allowlist is meant to
    reject an unregistered tool name long before it ever reaches this
    module. Exists as defensive depth so a `TOOL_REGISTRY` entry that
    hasn't yet been given a dispatch handler here fails loudly with a
    clear message, instead of a bare `KeyError` (or, worse, silently doing
    nothing).
    """


def _reattach_utc(value: datetime) -> datetime:
    """If `value` came back from Postgres as a naive datetime (this
    codebase's established `TIMESTAMP WITHOUT TIME ZONE` convention -- see
    module docstring's "expires_at timezone round-trip" section), reattach
    `timezone.utc` without shifting the wall-clock digits, so `.isoformat()`
    on the result matches what the ORIGINAL tz-aware value produced at
    insert time. A no-op if `value` is already tz-aware.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _record_error_tool_call(
    session: Session,
    *,
    approval_row: ApprovalRequest,
    tool_name: str,
    tool_params: dict,
    idempotency_key: str,
    error_cls: type[MutationExecutionError],
) -> None:
    """Write a `status="error"` `ToolCall` row for one of `execute_
    mutation`'s pre-dispatch rejections (tamper/not-decided/expired) --
    called just before the corresponding exception is raised, so the
    audit trail records every rejected attempt, not just successful ones.
    """
    tool_version = TOOL_REGISTRY[tool_name].version if tool_name in TOOL_REGISTRY else "unknown"
    session.add(
        ToolCall(
            agent_run_id=approval_row.agent_run_id,
            tool_name=tool_name,
            tool_version=tool_version,
            input_params_json=json.dumps(tool_params, sort_keys=True),
            output_json=None,
            status="error",
            error_taxonomy_code=error_cls.__name__,
            idempotency_key=idempotency_key,
            approval_request_id=approval_row.id,
        )
    )
    session.flush()


def _dispatch_grant_vpn_access(session: Session, tool_params: dict):
    """The one mutating-tool dispatch target this phase has. Returns
    `(output, before, after, entity_type, entity_id)` for `execute_
    mutation` to record as the `ToolCall.output_json` and the `AuditLog`
    before/after diff.

    Queries "did this employee already hold an active grant" itself,
    separately from `grant_vpn_access`'s own internal idempotency check --
    not a redundant duplication of logic to fix, but a deliberate choice to
    build an honest `before` audit snapshot without changing Task 3's
    already-stable `grant_vpn_access(session, employee_id, justification)`
    signature/return shape (which returns only the resulting grant row, not
    a before/after pair).
    """
    employee_id = tool_params["employee_id"]
    justification = tool_params["justification"]

    entitlement = ensure_vpn_entitlement_seeded(session)
    had_active_grant_before = (
        session.execute(
            select(EmployeeEntitlement).where(
                EmployeeEntitlement.employee_id == employee_id,
                EmployeeEntitlement.entitlement_id == entitlement.id,
                EmployeeEntitlement.revoked_at.is_(None),
            )
        ).scalar_one_or_none()
        is not None
    )

    grant = grant_vpn_access(session, employee_id=employee_id, justification=justification)

    output = {
        "employee_entitlement_id": grant.id,
        "employee_id": grant.employee_id,
        "entitlement_id": grant.entitlement_id,
        "granted_at": grant.granted_at.isoformat() if grant.granted_at else None,
    }
    before = {"employee_id": employee_id, "had_vpn_access": had_active_grant_before}
    after = {
        "employee_id": employee_id,
        "had_vpn_access": True,
        "employee_entitlement_id": grant.id,
        "justification": justification,
    }
    return output, before, after, "employee_entitlement", grant.id


_MUTATION_DISPATCH = {
    "grant_vpn_access": _dispatch_grant_vpn_access,
}


def execute_mutation(
    session: Session,
    *,
    approval_request_id: int,
    tool_name: str,
    tool_params: dict,
    actor_employee_id: int | None,
) -> dict:
    """Execute an already-approved mutating tool call -- the function
    Task 7's approver-decide endpoint (not yet built) calls after an
    `ApprovalRequest` transitions to `status="approved"`, strictly after
    `request_approval`'s `interrupt()` resume boundary (see module
    docstring for exactly what this means and why).

    Order of operations (matches the plan doc's literal step ordering,
    verified deliberate -- see `ApprovalExpiredError`'s docstring for why
    the status check runs before the expiry check):
      1. Fetch the `ApprovalRequest` row; `ApprovalNotFoundError` if missing.
      2. Acquire this row's transaction-scoped advisory lock (see module
         docstring's "Duplicate-replay concurrency" section) -- held for
         the remainder of this call, released when the CALLER commits or
         rolls back `session` (this function does not commit).
      3. Re-verify the snapshot hash; `ApprovalTamperError` (+ a logged
         `ToolCall` error row) if it doesn't match.
      4. Check `status == "approved"`; `ApprovalNotDecidedError` (+ logged)
         otherwise.
      5. Check `expires_at` is still in the future; `ApprovalExpiredError`
         (+ logged) otherwise.
      6. Duplicate-replay guard: if a `ToolCall` with
         `idempotency_key=f"approval:{approval_request_id}"` and
         `status="success"` already exists, return its recorded
         `output_json` instead of re-executing the adapter.
      7. Dispatch to the real mutating adapter, record a `status="success"`
         `ToolCall` row, and write a hash-chained `AuditLog` entry via
         `record_audit_event` (actor_type="agent" -- see module docstring
         for why this is a new, but natural, addition to this codebase's
         established `actor_type` vocabulary of "employee"/"analyst").

    Does not commit `session` -- see module docstring's opening section for
    why, and for the advisory lock's resulting release timing.
    """
    row = session.get(ApprovalRequest, approval_request_id)
    if row is None:
        raise ApprovalNotFoundError(f"no ApprovalRequest with id={approval_request_id}")

    # Acquire the lock BEFORE any check below -- see module docstring's
    # "Duplicate-replay concurrency" section for why this is a plain int
    # key (approval_request_id itself), unlike approval_service.py's
    # hashed composite key.
    session.execute(select(func.pg_advisory_xact_lock(approval_request_id)))

    idempotency_key = f"approval:{approval_request_id}"

    stored_params = json.loads(row.action_params_json)
    stored_evidence_refs = (
        json.loads(row.bound_evidence_refs_json) if row.bound_evidence_refs_json is not None else None
    )
    expires_at = _reattach_utc(row.expires_at)

    recomputed_hash = compute_snapshot_hash(
        action_type=row.action_type,
        params=stored_params,
        actor=row.requested_by_id,
        evidence_refs=stored_evidence_refs,
        risk_context=row.risk_context,
        expires_at=expires_at,
    )
    if recomputed_hash != row.snapshot_hash:
        _record_error_tool_call(
            session,
            approval_row=row,
            tool_name=tool_name,
            tool_params=tool_params,
            idempotency_key=idempotency_key,
            error_cls=ApprovalTamperError,
        )
        raise ApprovalTamperError(
            f"snapshot hash mismatch for ApprovalRequest id={approval_request_id}: "
            "the row's bound fields no longer match its recorded snapshot_hash"
        )

    # Caller-supplied tool_name/tool_params vs. the row's own approved
    # action_type/action_params_json -- a real gap in the plan doc's literal
    # step ordering, caught the same way Task 5 caught its wall-clock hazard:
    # the plan's step 6 says to dispatch using the FUNCTION'S OWN tool_params
    # argument, but that argument is caller-supplied and was never covered by
    # the snapshot-hash check above (that check only re-verifies the ROW's
    # stored fields against themselves -- it says nothing about whether
    # `tool_params` passed into THIS call matches what was actually approved).
    # Taken literally, a caller could pass a valid, unexpired, approved
    # `approval_request_id` alongside a COMPLETELY DIFFERENT `tool_params`
    # (e.g. a different `employee_id`) and this function would happily grant
    # access to whoever the caller named, not whoever was actually approved --
    # silently defeating the entire approval gate while every tamper/status/
    # expiry check above still reports green. Closed here by requiring
    # `tool_name`/`tool_params` to exactly equal the row's own
    # `action_type`/parsed `action_params_json` -- treated as the same
    # category of defect as a snapshot-hash mismatch (an attempt to execute
    # something other than what was actually approved), not a separate softer
    # error. Dispatch below then uses `stored_params` (the hash-verified
    # source of truth), never the caller's `tool_params`, as further
    # defense-in-depth even after this equality check passes.
    if tool_name != row.action_type or tool_params != stored_params:
        _record_error_tool_call(
            session,
            approval_row=row,
            tool_name=tool_name,
            tool_params=tool_params,
            idempotency_key=idempotency_key,
            error_cls=ApprovalTamperError,
        )
        raise ApprovalTamperError(
            f"tool_name/tool_params passed to execute_mutation do not match the approved "
            f"ApprovalRequest id={approval_request_id}'s own action_type/action_params_json "
            "-- refusing to execute a different action than the one that was actually approved"
        )

    if row.status != "approved":
        _record_error_tool_call(
            session,
            approval_row=row,
            tool_name=tool_name,
            tool_params=tool_params,
            idempotency_key=idempotency_key,
            error_cls=ApprovalNotDecidedError,
        )
        raise ApprovalNotDecidedError(
            f"ApprovalRequest id={approval_request_id} is not approved (status={row.status!r})"
        )

    if expires_at <= datetime.now(timezone.utc):
        _record_error_tool_call(
            session,
            approval_row=row,
            tool_name=tool_name,
            tool_params=tool_params,
            idempotency_key=idempotency_key,
            error_cls=ApprovalExpiredError,
        )
        raise ApprovalExpiredError(
            f"ApprovalRequest id={approval_request_id} expired at {expires_at.isoformat()}"
        )

    # Duplicate-replay guard. `.scalar_one_or_none()` (not `.first()`),
    # matching `approval_service._find_existing_request`'s precedent --
    # more than one "success" row for this key should be structurally
    # impossible now that the advisory lock above serializes every caller
    # for this approval_request_id; surface real corruption loudly instead
    # of silently picking one if it somehow happens.
    existing_success = session.execute(
        select(ToolCall).where(
            ToolCall.idempotency_key == idempotency_key,
            ToolCall.status == "success",
        )
    ).scalar_one_or_none()
    if existing_success is not None:
        return {
            "approval_request_id": approval_request_id,
            "tool_name": tool_name,
            "status": "success",
            "output": json.loads(existing_success.output_json) if existing_success.output_json else None,
        }

    handler = _MUTATION_DISPATCH.get(tool_name)
    if handler is None:
        raise UnknownMutationToolError(f"no mutation dispatch registered for tool {tool_name!r}")

    # Dispatch using stored_params (the row's own hash-verified params),
    # NOT the caller-supplied tool_params -- see the equality check above
    # for why tool_params is required to match stored_params exactly, and
    # why execution still prefers the verified source of truth even after
    # that check passes.
    output, before, after, entity_type, entity_id = handler(session, stored_params)

    tool_version = TOOL_REGISTRY[tool_name].version if tool_name in TOOL_REGISTRY else "unknown"
    tool_call = ToolCall(
        agent_run_id=row.agent_run_id,
        tool_name=tool_name,
        tool_version=tool_version,
        input_params_json=json.dumps(stored_params, sort_keys=True),
        output_json=json.dumps(output, sort_keys=True),
        status="success",
        idempotency_key=idempotency_key,
        approval_request_id=row.id,
    )
    session.add(tool_call)
    session.flush()

    record_audit_event(
        session,
        actor_type="agent",
        actor_id=actor_employee_id,
        action=f"tool.{tool_name}",
        entity_type=entity_type,
        entity_id=entity_id,
        before=before,
        after=after,
        metadata={"approval_request_id": approval_request_id, "tool_call_id": tool_call.id},
    )

    return {
        "approval_request_id": approval_request_id,
        "tool_name": tool_name,
        "status": "success",
        "output": output,
    }


def _dispatch_lookup_employee_entitlements(session: Session, tool_params: dict):
    employee_id = tool_params["employee_id"]
    summaries = lookup_employee_entitlements(session, employee_id)
    return [
        {
            "entitlement_id": summary.entitlement_id,
            "entitlement_name": summary.entitlement_name,
            "access_group_name": summary.access_group_name,
            "granted_at": summary.granted_at.isoformat() if summary.granted_at else None,
        }
        for summary in summaries
    ]


_READONLY_DISPATCH = {
    "lookup_employee_entitlements": _dispatch_lookup_employee_entitlements,
}


def execute_readonly_tool(session: Session, *, tool_name: str, tool_params: dict) -> dict:
    """Execute a non-mutating tool directly -- no approval gate, no
    duplicate-replay guard (both are meaningless for a read-only,
    side-effect-free call -- see module docstring). Still logs a `ToolCall`
    row (`status="success"`/`"error"`) for audit/telemetry completeness.

    Does not commit `session` -- same "caller owns the transaction"
    convention as `execute_mutation` (see module docstring).
    """
    tool_version = TOOL_REGISTRY[tool_name].version if tool_name in TOOL_REGISTRY else "unknown"
    input_params_json = json.dumps(tool_params, sort_keys=True)
    handler = _READONLY_DISPATCH.get(tool_name)

    if handler is None:
        session.add(
            ToolCall(
                tool_name=tool_name,
                tool_version=tool_version,
                input_params_json=input_params_json,
                output_json=None,
                status="error",
                error_taxonomy_code=UnknownMutationToolError.__name__,
            )
        )
        session.flush()
        raise UnknownMutationToolError(f"no read-only dispatch registered for tool {tool_name!r}")

    try:
        output = handler(session, tool_params)
    except Exception as exc:
        session.add(
            ToolCall(
                tool_name=tool_name,
                tool_version=tool_version,
                input_params_json=input_params_json,
                output_json=None,
                status="error",
                error_taxonomy_code=type(exc).__name__,
            )
        )
        session.flush()
        raise

    session.add(
        ToolCall(
            tool_name=tool_name,
            tool_version=tool_version,
            input_params_json=input_params_json,
            output_json=json.dumps(output, sort_keys=True),
            status="success",
        )
    )
    session.flush()

    return {"tool_name": tool_name, "status": "success", "output": output}
