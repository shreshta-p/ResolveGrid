from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.audit import record_audit_event
from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import Queue, Ticket, TicketMessage, TicketStateTransition
from resolvegrid_api.rate_limit import check_ticket_creation_rate_limit
from resolvegrid_authz import Decision, Principal, authorize
from resolvegrid_contracts import TicketCreateRequest, TicketTransitionRequest, is_valid_transition

router = APIRouter(prefix="/tickets", tags=["tickets"])


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
