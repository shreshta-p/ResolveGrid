import json
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.audit import record_audit_event
from resolvegrid_api.authz_scope import apply_decision_scope, resource_in_scope
from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.diagnostic_agent import diagnose_ticket
from resolvegrid_api.diagnostic_tools import TicketNotAuthorizedError, TicketNotFoundError
from resolvegrid_api.model_call_logging import current_pricing_version, log_completion
from resolvegrid_api.models import AgentRun, Queue, Span, Ticket, TicketMessage, TicketStateTransition
from resolvegrid_api.rate_limit import check_ticket_creation_rate_limit
from resolvegrid_authz import Decision, Principal, authorize
from resolvegrid_contracts import TicketCreateRequest, TicketTransitionRequest, is_valid_transition

router = APIRouter(prefix="/tickets", tags=["tickets"])

# The global tracer provider is already configured once at app startup by
# main.py's lifespan hook (resolvegrid_telemetry.init_tracing) -- this module
# must NOT call init_tracing again, just bind a tracer to whatever provider is
# globally registered by the time a span is actually started.
tracer = trace.get_tracer(__name__)


def _ticket_to_dict(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "subject": ticket.subject,
        "type": ticket.type,
        "priority": ticket.priority,
        "status": ticket.status,
        "queue_id": ticket.queue_id,
        "requester_id": ticket.requester_id,
        "assignee_id": ticket.assignee_id,
        # ISO strings, not raw datetimes: this dict is also passed straight
        # into record_audit_event()'s before/after payload, which json.dumps
        # with the plain stdlib encoder (no datetime support) -- a raw
        # datetime here raises TypeError inside the audit call, not in
        # FastAPI's own (datetime-aware) response serialization.
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
    }


def _apply_ticket_scope(query, decision: Decision):
    # The join is conditional (only when department-scoping is actually in
    # play) -- see `authz_scope.apply_decision_scope`'s docstring for why
    # that join can't move into the shared helper itself.
    if decision.department_ids is not None:
        query = query.join(Queue, Ticket.queue_id == Queue.id)
    return apply_decision_scope(
        query, decision, department_id_column=Queue.department_id, owner_id_column=Ticket.requester_id
    )


def _ticket_in_scope(decision: Decision, ticket: Ticket, ticket_department_id: int | None) -> bool:
    return resource_in_scope(decision, department_id=ticket_department_id, owner_id=ticket.requester_id)


@router.post("", status_code=201)
def create_ticket(
    payload: TicketCreateRequest,
    session: Session = Depends(get_db),
    principal: Principal = Depends(check_ticket_creation_rate_limit),
) -> dict:
    ticket = Ticket(
        subject=payload.subject,
        type=payload.type,
        priority=payload.priority,
        queue_id=payload.queue_id,
        requester_id=principal.employee_id,
    )
    session.add(ticket)
    session.flush()

    session.add(TicketMessage(ticket_id=ticket.id, author_type="employee", author_id=principal.employee_id, body=payload.body))
    record_audit_event(
        session, actor_type="employee", actor_id=principal.employee_id, action="ticket.create",
        entity_type="ticket", entity_id=ticket.id, after=_ticket_to_dict(ticket),
    )
    session.commit()
    return _ticket_to_dict(ticket)


@router.get("")
def list_tickets(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict]:
    decision = authorize(principal, "ticket.list")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    query = _apply_ticket_scope(select(Ticket), decision)
    tickets = session.scalars(query).all()
    return [_ticket_to_dict(t) for t in tickets]


@router.get("/{ticket_id}")
def get_ticket(
    ticket_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    decision = authorize(principal, "ticket.view")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    queue = session.get(Queue, ticket.queue_id)
    if not _ticket_in_scope(decision, ticket, queue.department_id if queue else None):
        raise HTTPException(status_code=403, detail="not authorized to view this ticket")

    return _ticket_to_dict(ticket)


@router.post("/{ticket_id}/transition")
def transition_ticket(
    ticket_id: int,
    payload: TicketTransitionRequest,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    decision = authorize(principal, "ticket.transition")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    queue = session.get(Queue, ticket.queue_id)
    if not _ticket_in_scope(decision, ticket, queue.department_id if queue else None):
        raise HTTPException(status_code=403, detail="not authorized to transition this ticket")

    if not is_valid_transition(ticket.status, payload.to_status):
        raise HTTPException(status_code=400, detail=f"cannot transition from {ticket.status} to {payload.to_status}")

    before = _ticket_to_dict(ticket)
    from_status = ticket.status
    ticket.status = payload.to_status
    session.add(
        TicketStateTransition(
            ticket_id=ticket.id, from_status=from_status, to_status=payload.to_status,
            actor_type="analyst", actor_id=principal.employee_id, reason=payload.reason,
        )
    )
    record_audit_event(
        session, actor_type="analyst", actor_id=principal.employee_id, action="ticket.transition",
        entity_type="ticket", entity_id=ticket.id, before=before, after=_ticket_to_dict(ticket),
    )
    session.commit()
    return _ticket_to_dict(ticket)


@router.post("/{ticket_id}/summarize")
def summarize_ticket(
    ticket_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    # Viewing a summary requires the same scope as viewing the ticket itself
    # -- reuse ticket.view rather than inventing a new authz action.
    decision = authorize(principal, "ticket.view")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    queue = session.get(Queue, ticket.queue_id)
    if not _ticket_in_scope(decision, ticket, queue.department_id if queue else None):
        raise HTTPException(status_code=403, detail="not authorized to view this ticket")

    messages = session.scalars(
        select(TicketMessage).where(TicketMessage.ticket_id == ticket.id).order_by(TicketMessage.created_at)
    ).all()
    body_text = "\n".join(m.body for m in messages)
    prompt = (
        f"Summarize the following support ticket in 2-3 short sentences.\n\n"
        f"Subject: {ticket.subject}\n\n"
        f"Messages:\n{body_text}"
    )

    try:
        result = log_completion(session, tracer, purpose="ticket.summarize", prompt=prompt)
    except llm_gateway.LLMGatewayError as exc:
        # log_completion() already added the error-shaped ModelCall row to
        # the session; this call site still owns the commit boundary (same
        # convention as the success path below), so the error row must be
        # committed here before turning the failure into an HTTP response --
        # raw_db_session-based tests read via a separate connection and only
        # ever see committed data.
        session.commit()
        raise HTTPException(status_code=502, detail=f"LLM gateway error: {exc}") from exc

    pricing = current_pricing_version(session, result.provider, result.model)
    if pricing is not None:
        estimated_cost_usd = (
            result.input_tokens / 1000 * pricing.input_cost_per_1k_tokens_usd
            + result.output_tokens / 1000 * pricing.output_cost_per_1k_tokens_usd
        )
    else:
        estimated_cost_usd = 0.0
    session.commit()

    return {
        "summary": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "estimated_cost_usd": estimated_cost_usd,
    }


@router.post("/{ticket_id}/diagnose")
async def diagnose_ticket_endpoint(
    ticket_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """Phase 13 Task 4: real Google ADK diagnostic specialist, exposed as an
    analyst-facing ticket-detail-page action (the "Diagnose" button), per
    `docs/superpowers/plans/2026-10-21-phase13-adk-diagnostic-specialist.md`.

    Architecture decision -- NO LangGraph node/graph wraps this call (see
    `docs/adr/0004-adk-diagnostic-specialist-wiring.md`'s Task 4 addendum
    for the full, documented reasoning): `diagnostic_agent.diagnose_ticket`
    already runs ADK's own `Runner`'s complete, bounded, single-shot
    tool-calling loop and returns a finished result -- there is no
    `interrupt()`/resume boundary, no conditional routing, and no
    multi-turn state needing LangGraph's checkpointing the way `build_
    tool_invocation_graph` genuinely needs it for a durable, human-paused
    approval. `diagnose_ticket`'s own signature (a live `Session`/`Tracer`/
    `Principal` in, a plain `DiagnosisResult` out) is also not AgentState
    -shaped -- wrapping it in a one-node `StateGraph` would only add
    checkpointer-sharing/state-marshalling ceremony for zero real
    capability, so this handler calls it directly, exactly like
    `summarize_ticket` above calls `log_completion` directly.

    Authz, two independent layers (defense in depth, mirroring `diagnostic_
    tools.py`'s own documented two-layer pattern):
      1. `ticket.diagnose` (a new, Phase 13 Task 4 `_STAFF_ONLY_ACTIONS`
         entry -- `packages/authz/policy.py`) gates this endpoint to
         admin/department-scoped analyst-or-approver principals only,
         checked FIRST, before any `AgentRun` row or real model call is
         spent on a principal who cannot use this feature AT ALL --
         mirrors `routers/approvals.py`'s `approval.list`/`approval.decide`
         precedent exactly (same staff-only action shape, same "deny
         outright, never downgrade to self-scope" semantic). A ticket's own
         requester gets no special self-view carve-out here, unlike
         `ticket.view` -- "diagnose my own ticket" is a staff capability in
         this phase's plan, not a personal one.
      2. The specific target ticket's own 404-then-403 (mirrors `get_
         ticket`'s existing precedent exactly): `diagnose_ticket` calls the
         real `get_ticket_state_and_history` tool internally, BEFORE
         building any real `Agent`/spending any real model call, raising
         `TicketNotFoundError`/`TicketNotAuthorizedError` -- caught below
         and translated to 404/403 respectively.

    Real `AgentRun`/`Span` telemetry (per this task's own requirement --
    does NOT need to match `/chat`'s exact 4-row `_STAGE_NAMES` shape,
    since this is a genuinely different, single-call workflow, not a
    5-node LangGraph chain): one `AgentRun` row per real diagnostic run
    (status/timing/output, same columns `/chat` populates), and exactly ONE
    `Span` row (`stage_name="diagnose_ticket"`) carrying the REAL measured
    wall-clock latency of the whole `diagnose_ticket` call. A per-tool-call
    breakdown the way `/chat`'s `node_latencies_ms` provides is not
    available here and is not fabricated: ADK's own `Runner` owns its
    internal multi-turn tool-calling loop as one opaque unit (unlike
    LangGraph's own per-node `time.monotonic()` instrumentation, nothing
    inside that loop is separately timed), so `Span.detail_json` instead
    carries the real `tool_calls_made`/`turn_count`/`model_calls_logged`
    counts `DiagnosisResult` already reports -- real, structured,
    non-fabricated detail, just not a second latency breakdown this
    codebase has no real measurement for. `AgentRun.thread_id` is populated
    with a fresh `uuid4().hex` for this row's own uniqueness constraint
    (matching `/chat`'s minting convention), but -- since no LangGraph
    graph is invoked here -- it is NOT a real LangGraph checkpoint key for
    this run, unlike every other `AgentRun` row in this codebase; this is a
    direct, documented consequence of this handler's own "no graph" design
    decision, not an oversight.

    `async def` for the same reason `chat.py`'s/`tools.py`'s handlers are:
    `diagnose_ticket` is itself `async` (ADK's `Runner.run_async` requires
    a running event loop).
    """
    decision = authorize(principal, "ticket.diagnose")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    agent_run = AgentRun(
        status="running",
        thread_id=uuid4().hex,
        principal_employee_id=principal.employee_id,
        input_text=f"[diagnose] ticket #{ticket_id}",
    )
    session.add(agent_run)
    session.commit()

    start = time.monotonic()
    try:
        result = await diagnose_ticket(session, tracer, principal, ticket_id)
    except TicketNotFoundError as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        agent_run.status = "error"
        agent_run.error_message = str(exc)
        agent_run.completed_at = datetime.now(timezone.utc)
        session.add(
            Span(agent_run_id=agent_run.id, stage_name="diagnose_ticket", status="error", latency_ms=latency_ms)
        )
        session.commit()
        raise HTTPException(status_code=404, detail="ticket not found") from exc
    except TicketNotAuthorizedError as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        agent_run.status = "error"
        agent_run.error_message = str(exc)
        agent_run.completed_at = datetime.now(timezone.utc)
        session.add(
            Span(agent_run_id=agent_run.id, stage_name="diagnose_ticket", status="error", latency_ms=latency_ms)
        )
        session.commit()
        raise HTTPException(status_code=403, detail="not authorized to diagnose this ticket") from exc
    except Exception as exc:
        # Mirrors chat.py's broad catch: a real ADK/LiteLLM/DB failure can
        # come from several layers beneath this call; the endpoint's
        # contract is "the diagnostic run failed, cleanly" regardless of
        # which layer raised.
        latency_ms = int((time.monotonic() - start) * 1000)
        agent_run.status = "error"
        agent_run.error_message = str(exc)
        agent_run.completed_at = datetime.now(timezone.utc)
        session.add(
            Span(agent_run_id=agent_run.id, stage_name="diagnose_ticket", status="error", latency_ms=latency_ms)
        )
        session.commit()
        raise HTTPException(status_code=502, detail=f"diagnosis failed: {exc}") from exc

    latency_ms = int((time.monotonic() - start) * 1000)
    agent_run.status = "completed"
    agent_run.output_text = result.summary
    agent_run.completed_at = datetime.now(timezone.utc)
    session.add(
        Span(
            agent_run_id=agent_run.id,
            stage_name="diagnose_ticket",
            status="success",
            latency_ms=latency_ms,
            detail_json=json.dumps(
                {
                    "tool_calls_made": result.tool_calls_made,
                    "turn_count": result.turn_count,
                    "model_calls_logged": result.model_calls_logged,
                    "hallucinated_tool_call": result.hallucinated_tool_call,
                }
            ),
        )
    )
    session.commit()

    return {
        "summary": result.summary,
        "citations": result.cited_chunks,
        "tool_calls_made": result.tool_calls_made,
        "thread_id": agent_run.thread_id,
        # Real, honest signal (Phase 13 Task 6): a real, reproducible
        # qwen3:14b behavior sometimes hallucinates its final answer as a
        # fake tool call instead of plain final text (see `diagnostic_
        # agent.diagnose_ticket`'s docstring) -- this run still completes
        # cleanly (200, never a 502) either way, but the UI/caller can use
        # this to show an honest "degraded run" indicator rather than
        # silently treating every 200 as equally content-complete.
        "hallucinated_tool_call": result.hallucinated_tool_call,
    }
