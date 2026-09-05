"""Phase 9 Task 7a: `POST /tools/{tool_name}/invoke` -- the endpoint that
makes Task 4's allowlist/validation logic and Task 5/6's approval/mutation
nodes reachable through the real running app for the first time, via
`build_tool_invocation_graph` (`services/agent-orchestration`'s `graph.py`,
wired up in `main.py`'s lifespan as `request.app.state.tool_invocation_graph`).

Flow (matches Task 4's own documented three-step allowlist/validate/execute
sequence, `tool_execution.py`'s module docstring):
  1. `available_tools_for_principal(principal)` -> `select_tool(tool_name,
     available)` -- resolves `tool_name` against the principal's own
     filtered allowlist, never `TOOL_REGISTRY` directly. `ToolNotAllowedError`
     (covers both "doesn't exist" and "not permitted" identically, per that
     exception's own documented safe-error-envelope design) maps to a plain
     403 that does not echo back which case occurred.
  2. `validate_tool_schema(tool, payload.params)` -- `ToolValidationError`
     maps to 422 with the underlying (non-security-sensitive, per that
     exception's docstring) validation message.
  3a. Non-mutating tool (`not tool.requires_approval`): `execute_readonly_tool`
      (Task 6) is called directly against this request's own `Depends(get_db)`
      session -- no graph, no `interrupt()`, matching Task 6's own "read-only
      tools skip the approval machinery" design. This function does not
      commit internally (see its docstring), so this endpoint commits after
      a successful call.
  3b. Mutating, approval-gated tool (`tool.requires_approval`): a fresh
      `thread_id` (mirrors `chat.py`'s `uuid4().hex` convention) seeds the
      tool-invocation graph's initial `AgentState`, which is `ainvoke()`d.
      This call is expected to hit `request_approval`'s real `interrupt()`
      and return WITHOUT completing -- verified against the installed
      `langgraph==1.2.11` runtime (`pregel/main.py`'s `ainvoke`): an
      interrupted run's result dict carries an extra `"__interrupt__"` key
      (`langgraph._internal._constants.INTERRUPT`) alongside whatever state
      values were already committed by the time the pause occurred, not a
      completed final state. Since `request_approval` is this graph's FIRST
      node and pauses before its own `return` statement (its `interrupt()`
      call raises to pause -- see that node's docstring), `result[
      "approval_request_id"]` is NOT populated on this interrupted return
      (the node's return dict, which is what would set it, never executed);
      the real, durable `ApprovalRequest.id` this call created (or
      idempotently found) is instead read off the interrupt's own payload --
      `result["__interrupt__"][0].value["approval_request_id"]` -- which
      `request_approval` populates from `request_approval_fn`'s return value
      BEFORE calling `interrupt()` (see that node's docstring, step 3). A
      small DB fallback (querying `ApprovalRequest` by `agent_run_id ==
      thread_id`) covers the defensive case where a future graph change
      alters what an interrupted run's result dict carries.

      `agent_run_id`/`thread_id` flow: this endpoint's `thread_id` is placed
      on `AgentState["thread_id"]`; `request_approval`'s node reads
      `state.get("thread_id")` into its payload's `agent_run_id` key (see
      `graph.py`'s `RequestApprovalFn` comment); `request_approval_for_agent`
      (`approval_service.py`) writes that value straight onto
      `ApprovalRequest.agent_run_id` at insert time. So the `thread_id`
      returned to this endpoint's caller IS the exact string a later
      Task 7b resume call would look up the paused run by (`config=
      {"configurable": {"thread_id": agent_run_id}}`) -- confirmed by
      tracing the real code path, not assumed.

Response shape for the mutating path, `{"status": "pending_approval",
"approval_request_id": <int>, "thread_id": <str>}`, is this task's own
design (Task 7b, not yet built, is the real consumer of `approval_request_id`/
`thread_id` for its list/decide endpoints).
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import ApprovalRequest
from resolvegrid_api.mutation_execution import execute_readonly_tool
from resolvegrid_api.tool_execution import (
    ToolNotAllowedError,
    ToolValidationError,
    available_tools_for_principal,
    select_tool,
    validate_tool_schema,
)
from resolvegrid_authz import Principal

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInvokeRequest(BaseModel):
    params: dict


@router.post("/{tool_name}/invoke")
async def invoke_tool(
    tool_name: str,
    payload: ToolInvokeRequest,
    request: Request,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """See this module's docstring for the full flow. `async def` for the
    same reason `chat.py`'s handler is `async def` -- the mutating branch
    below must `await ainvoke()` an `AsyncPostgresSaver`-backed graph inside
    a running event loop; there is no synchronous equivalent in use here.
    """
    available = available_tools_for_principal(principal)
    try:
        tool = select_tool(tool_name, available)
    except ToolNotAllowedError:
        # Deliberately a generic detail message -- see ToolNotAllowedError's
        # own docstring: "doesn't exist" vs. "exists but you're not
        # permitted" must stay indistinguishable to the caller.
        raise HTTPException(status_code=403, detail="tool not allowed") from None

    try:
        validate_tool_schema(tool, payload.params)
    except ToolValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not tool.requires_approval:
        result = execute_readonly_tool(session, tool_name=tool_name, tool_params=payload.params)
        session.commit()
        return result

    thread_id = uuid4().hex
    initial_state = {
        "thread_id": thread_id,
        "principal_employee_id": principal.employee_id,
        # Unused by this graph (no classify_intent/compose_response node
        # exists here) -- populated only because AgentState is the one
        # shared TypedDict both `build_graph` and `build_tool_invocation_
        # graph` compile against; a short, descriptive placeholder keeps a
        # checkpointed row human-readable if ever inspected directly.
        "input_text": f"[tool invocation] {tool_name}",
        "intent": None,
        # Coarse stand-in risk_context for `request_approval`'s payload
        # (see graph.py's `RequestApprovalFn` comment: `risk_context` <-
        # `state["risk_level"]`) -- this endpoint has no classify_intent
        # node to derive a real risk_level from, and every tool reaching
        # this branch is, by definition, `requires_approval=True` (today,
        # always also `mutating=True`), so a fixed "high" is a more honest
        # default than leaving this `None` for an approver's UI to show
        # blank. A real per-tool risk classification is future work, not
        # this task's scope.
        "risk_level": "high",
        "retrieval_scope": None,
        "retrieved_chunks": None,
        "retrieval_sufficient": None,
        "context_block": None,
        "output_text": None,
        "error": None,
        "citations_verified": None,
        "verified_chunk_ids": None,
        "fabricated_chunk_ids": None,
        "proposed_tool_name": tool_name,
        "proposed_tool_params": payload.params,
        "approval_request_id": None,
        "approval_decision": None,
        "tool_invocation_result": None,
    }

    try:
        result_state = await request.app.state.tool_invocation_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        # Mirrors chat.py's broad catch: a graph run can fail for reasons
        # beyond any one layer (checkpointer/DB errors, a node raising for
        # any other reason) -- this endpoint's contract is "the tool
        # invocation failed, cleanly" regardless of which layer raised.
        raise HTTPException(status_code=502, detail=f"tool invocation failed: {exc}") from exc

    interrupts = result_state.get("__interrupt__") if isinstance(result_state, dict) else None
    if not interrupts:
        # Should be unreachable: every tool reaching this branch has
        # requires_approval=True, so a fresh thread_id's first ainvoke()
        # must pause at request_approval's interrupt() -- see this
        # module's docstring. Surfaced as a clear 500 rather than silently
        # returning a misleading "pending_approval" for a run that may have
        # actually completed or errored.
        raise HTTPException(
            status_code=500,
            detail="tool invocation graph did not pause for approval as expected",
        )

    approval_request_id = interrupts[0].value.get("approval_request_id")
    if approval_request_id is None:
        # Defensive fallback -- see module docstring's "agent_run_id/
        # thread_id flow" section. Should not be needed given the above,
        # but a DB lookup by this run's own agent_run_id is a safe,
        # cheap backstop if it ever is.
        row = session.execute(
            select(ApprovalRequest).where(ApprovalRequest.agent_run_id == thread_id)
        ).scalar_one_or_none()
        approval_request_id = row.id if row is not None else None

    return {
        "status": "pending_approval",
        "approval_request_id": approval_request_id,
        "thread_id": thread_id,
    }
