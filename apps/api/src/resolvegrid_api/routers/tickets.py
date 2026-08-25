from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.audit import record_audit_event
from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import ModelCall, PricingVersion, Queue, Ticket, TicketMessage, TicketStateTransition
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
    if decision.department_ids is not None:
        query = query.join(Queue, Ticket.queue_id == Queue.id).where(Queue.department_id.in_(decision.department_ids))
    if decision.employee_id is not None:
        query = query.where(Ticket.requester_id == decision.employee_id)
    return query


def _ticket_in_scope(decision: Decision, ticket: Ticket, ticket_department_id: int | None) -> bool:
    if decision.employee_id is not None and decision.employee_id != ticket.requester_id:
        return False
    if decision.department_ids is not None and ticket_department_id not in decision.department_ids:
        return False
    return True


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


def _current_pricing_version(session: Session, provider: str, model: str) -> PricingVersion | None:
    # No `effective_at <= now()` filter -- harmless today (one seeded $0 row
    # for ollama/local-qwen3), but once a real paid provider adds a
    # future-dated PricingVersion row (a scheduled rate change), this would
    # pick it up early and misprice calls made before that date. Fix before
    # Phase 5 adds real provider pricing.
    return session.scalar(
        select(PricingVersion)
        .where(PricingVersion.provider == provider, PricingVersion.model == model)
        .order_by(PricingVersion.effective_at.desc())
    )


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

    with tracer.start_as_current_span("llm.summarize_ticket") as span:
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", llm_gateway.DEFAULT_MODEL)
        try:
            result = llm_gateway.complete(prompt)
        except llm_gateway.LLMGatewayError as exc:
            span.set_attribute("error.type", type(exc).__name__)
            session.add(
                ModelCall(
                    purpose="ticket.summarize", provider="ollama", model=llm_gateway.DEFAULT_MODEL,
                    pricing_version_id=None, input_tokens=0, output_tokens=0, latency_ms=0,
                    estimated_cost_usd=0.0, status="error", error_message=str(exc),
                )
            )
            session.commit()
            raise HTTPException(status_code=502, detail=f"LLM gateway error: {exc}") from exc

        span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
        # "gen_ai.response.model" is the currently-installed OTel GenAI
        # semconv name for the model that actually served the response (see
        # opentelemetry.semconv._incubating.attributes.gen_ai_attributes.
        # GEN_AI_RESPONSE_MODEL) -- flagged deprecated-in-favor-of-the-
        # standalone genai-semconv-repo in that module's docstring, but it's
        # still the only "response model" constant this installed package
        # ships, so it's the correct name to emit today. Only set when a
        # fallback (or any distinct serving model group) actually occurred;
        # a non-fallback call already has the request model on this span.
        if result.serving_model_group:
            span.set_attribute("gen_ai.response.model", result.serving_model_group)

    pricing = _current_pricing_version(session, result.provider, result.model)
    if pricing is not None:
        estimated_cost_usd = (
            result.input_tokens / 1000 * pricing.input_cost_per_1k_tokens_usd
            + result.output_tokens / 1000 * pricing.output_cost_per_1k_tokens_usd
        )
    else:
        estimated_cost_usd = 0.0

    call = ModelCall(
        purpose="ticket.summarize", provider=result.provider, model=result.model,
        pricing_version_id=pricing.id if pricing is not None else None,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        latency_ms=result.latency_ms, estimated_cost_usd=estimated_cost_usd, status="success",
        fallback_occurred=result.fallback_occurred, serving_model_group=result.serving_model_group,
    )
    session.add(call)
    session.commit()

    return {
        "summary": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "estimated_cost_usd": estimated_cost_usd,
    }
