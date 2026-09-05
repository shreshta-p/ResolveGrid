"""Phase 9 Task 7b: the approver-facing API -- `GET /approvals` (list pending
requests) and `POST /approvals/{approval_id}/decide` (record a decision and
drive the paused tool-invocation graph to completion). Closes the loop Task
7a opened: that task's `POST /tools/{tool_name}/invoke` gets a mutating call
as far as a durable, paused `interrupt()`; this router is what an approver
actually calls to resume it for real.

Department-scoping judgment call (documented per the plan doc's explicit
request, not silently faked): `authorize(principal, "approval.list")` /
`authorize(principal, "approval.decide")` are staff-only actions (see
`packages/authz/policy.py`), so the only two `Decision` shapes a caller who
passes the check can get back are "admin: global access"
(`department_ids is None`, unrestricted) or "department-scoped access"
(`department_ids` a non-empty tuple of the analyst/approver department
grants this principal holds) -- never a self-scoped Decision (staff-only
actions never downgrade to that). `ApprovalRequest` has no department
column of its own, and this phase has exactly one registered `action_type`
(`grant_vpn_access`), which has no per-department routing built anywhere
yet -- `ApprovalPolicy` (Task 1's schema) exists but nothing populates or
reads `stages_json` in this phase. Building a real join here would mean
inventing a department-attribution rule (e.g. "the requester's own
department") that no requirement in this phase actually specifies, and
would silently look more precise than it is. Instead: `GET /approvals`
applies NO department filtering -- any principal who passes the
`approval.list` authz check (global admin, or holds ANY department-scoped
analyst/approver grant) sees every pending `ApprovalRequest` regardless of
department. This is a real, judged scope limitation for THIS phase's single
action type, not an oversight -- a future phase that adds a second
`action_type` with genuine department semantics (or wires up
`ApprovalPolicy` staging) should revisit this list query, not silently
assume it already filters.

Commit-then-resume ordering (the hard requirement the plan doc calls out):
`decide_approval` writes the `ApprovalDecision` row and updates
`ApprovalRequest.status`, then `session.commit()`s BEFORE ever touching the
graph. Only after that commit does it call
`tool_invocation_graph.ainvoke(Command(resume=...), ...)`. This ordering
means the human decision is durably recorded even if the resume call itself
blows up (a graph/checkpointer error, `execute_mutation_for_agent` raising
something `mutation_execution.py`'s own typed-error translation didn't
anticipate) -- the safer failure mode is "the approver's decision is
recorded but execution needs a retry/investigation," never "the decision
is lost because execution failed," which is exactly why the plan doc
requires this order and this router does not deviate from it.

Resumed-run response shape: unlike Task 7a's `POST /tools/{tool_name}/invoke`
(whose `ainvoke()` call is EXPECTED to pause at `interrupt()` and never
reach a final state), THIS router's `ainvoke(Command(resume=...))` call is
expected to run the paused thread all the way to `END` --
`build_tool_invocation_graph`'s only two nodes are `request_approval`
(already paused, now resuming past its `interrupt()` call) and
`execute_mutation` (this graph's last node, wired straight to `END` -- see
`graph.py`) -- so the returned dict is a real final `AgentState`, not an
interrupted return carrying `"__interrupt__"`. `state["tool_invocation_
result"]` (the field `make_execute_mutation_node` populates -- see
`state.py`) is what this router surfaces back to the caller. A defensive
check still guards against a future graph change reintroducing a second
pause (see `_ainvoke_resume`'s docstring) rather than silently
misinterpreting an interrupted return as a completed one.
"""

import json
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest
from resolvegrid_authz import Principal, authorize

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalDecideRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: str | None = None


def _approval_request_to_dict(row: ApprovalRequest) -> dict:
    return {
        "id": row.id,
        "action_type": row.action_type,
        # Parsed back to a real dict for the response -- callers (the
        # approver UI) shouldn't have to json.loads a nested JSON string
        # themselves, matching tickets.py's `_ticket_to_dict` convention of
        # never leaking this codebase's own JSON-as-text storage choices
        # into a response shape.
        "action_params": json.loads(row.action_params_json),
        "risk_context": row.risk_context,
        "bound_evidence_refs": (
            json.loads(row.bound_evidence_refs_json) if row.bound_evidence_refs_json is not None else None
        ),
        "requested_by_id": row.requested_by_id,
        "status": row.status,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("")
def list_approvals(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict]:
    """List pending approval requests. See module docstring's "Department-
    scoping judgment call" section for why this deliberately does NOT
    filter by `decision.department_ids` -- any principal who passes the
    `approval.list` authz check sees every pending request, for this
    phase's single, department-less `action_type`.

    Pending-only, not pending-plus-recently-decided: the more focused scope
    an approver actually needs ("what do I need to act on right now") --
    a decided request is already resolved and has no pending action for
    this endpoint's caller to take on it. A future audit-style view over
    decided requests, if ever needed, is a separate, differently-scoped
    endpoint, not a `?status=` filter bolted onto this one.
    """
    decision = authorize(principal, "approval.list")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    rows = session.scalars(
        select(ApprovalRequest)
        .where(ApprovalRequest.status == "pending")
        .order_by(ApprovalRequest.created_at)
    ).all()
    return [_approval_request_to_dict(row) for row in rows]


def _is_expired(expires_at: datetime) -> bool:
    """`ApprovalRequest.expires_at` is stored as a naive (timezone-less)
    Postgres `TIMESTAMP` column but always written as a tz-aware UTC value
    at insert time (see `approval_service.py`/`mutation_execution.py`'s
    matching `_reattach_utc` note) -- reattach `timezone.utc` before
    comparing against `datetime.now(timezone.utc)` so a naive value read
    back from the DB doesn't raise `TypeError: can't compare offset-naive
    and offset-aware datetimes`, and so the comparison is against the
    correct wall-clock instant rather than an accidentally-shifted one.
    """
    aware_expires_at = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=timezone.utc)
    return aware_expires_at <= datetime.now(timezone.utc)


@router.post("/{approval_id}/decide")
async def decide_approval(
    approval_id: int,
    payload: ApprovalDecideRequest,
    request: Request,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """Record an approver's decision, then resume the paused tool-invocation
    graph run for real. See module docstring for the full flow and the
    commit-then-resume ordering this deliberately follows.

    `async def` for the same reason `routers/tools.py`'s `invoke_tool` is --
    the resume branch below must `await ainvoke()` an `AsyncPostgresSaver`
    -backed graph inside a running event loop.

    Server-side reject-comment validation (a judgment call, documented per
    the plan doc's explicit invitation to make one): `apps/web`'s
    `/approvals` page already enforces "reject requires a non-empty
    comment" client-side, but this endpoint enforces the SAME rule
    server-side too, returning 422 if `decision == "rejected"` and
    `comment` is missing/blank. This mirrors Task 4's `validate_tool_schema`
    precedent -- this codebase's established "defense in depth even though
    a well-behaved caller would never send bad input" pattern -- and is
    more important here than a typical case: an approver's rejection reason
    is a real, security-relevant audit artifact (why was this mutating
    action refused?), not a cosmetic UI nicety, so a client bug or a direct
    API caller bypassing the browser must not be able to silently produce
    an undocumented rejection.
    """
    if payload.decision == "rejected" and (payload.comment is None or not payload.comment.strip()):
        raise HTTPException(status_code=422, detail="a non-empty comment is required when rejecting")

    decision = authorize(principal, "approval.decide")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    approval_request = session.get(ApprovalRequest, approval_id)
    if approval_request is None:
        raise HTTPException(status_code=404, detail="approval request not found")

    # Guard against a double-click / stale approver tab / retried request
    # re-deciding something -- checked BEFORE any write, so a rejected
    # second attempt leaves the original decision and status completely
    # untouched (no second ApprovalDecision row, no status flip-flop).
    if approval_request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"approval request {approval_id} has already been decided (status={approval_request.status!r})",
        )
    if _is_expired(approval_request.expires_at):
        raise HTTPException(
            status_code=409,
            detail=f"approval request {approval_id} expired at {approval_request.expires_at.isoformat()}",
        )

    session.add(
        ApprovalDecision(
            approval_request_id=approval_request.id,
            approver_id=principal.employee_id,
            decision=payload.decision,
            comment=payload.comment,
        )
    )
    approval_request.status = "approved" if payload.decision == "approved" else "rejected"
    # Commit BEFORE resuming the graph -- see module docstring's
    # "Commit-then-resume ordering" section. The decision is durably
    # recorded here regardless of what happens next.
    session.commit()

    if approval_request.agent_run_id is None:
        # Defensive: every `ApprovalRequest` this phase's real caller
        # (Task 7a's `POST /tools/{tool_name}/invoke`) creates always sets
        # `agent_run_id` to the minting `thread_id` -- see that router's
        # module docstring's "agent_run_id/thread_id flow" section. A row
        # with no `agent_run_id` has no paused graph run to resume against
        # at all (e.g. one inserted directly by a test, or by some future
        # non-graph caller) -- surfaced as a clear error rather than
        # passing `None` as a `thread_id` to `ainvoke`, which would either
        # raise an opaque error deeper in LangGraph or, worse, silently
        # address the wrong (or no) checkpointed thread.
        return {
            "approval_request_id": approval_request.id,
            "decision": payload.decision,
            "status": approval_request.status,
            "tool_invocation_result": None,
            "resume_error": "this approval request has no agent_run_id -- nothing to resume",
        }

    try:
        result_state = await request.app.state.tool_invocation_graph.ainvoke(
            Command(resume=payload.decision),
            config={"configurable": {"thread_id": approval_request.agent_run_id}},
        )
    except Exception as exc:
        # The decision is already durably committed above -- this is the
        # correct, safer failure mode (see module docstring): report the
        # resume failure clearly rather than raising an HTTPException that
        # would make the caller think the decision itself didn't take.
        return {
            "approval_request_id": approval_request.id,
            "decision": payload.decision,
            "status": approval_request.status,
            "tool_invocation_result": None,
            "resume_error": f"tool invocation resume failed: {exc}",
        }

    if isinstance(result_state, dict) and result_state.get("__interrupt__"):
        # Should be unreachable: `build_tool_invocation_graph` has exactly
        # two nodes (`request_approval` -> `execute_mutation` -> END), and
        # `execute_mutation` never calls `interrupt()` -- see graph.py. A
        # future graph change that added a second pause point would land
        # here instead of silently mis-surfacing an interrupted run's
        # partial state as if it were `tool_invocation_result`.
        return {
            "approval_request_id": approval_request.id,
            "decision": payload.decision,
            "status": approval_request.status,
            "tool_invocation_result": None,
            "resume_error": "tool invocation graph paused again unexpectedly instead of completing",
        }

    tool_invocation_result = result_state.get("tool_invocation_result") if isinstance(result_state, dict) else None
    return {
        "approval_request_id": approval_request.id,
        "decision": payload.decision,
        "status": approval_request.status,
        "tool_invocation_result": tool_invocation_result,
    }
