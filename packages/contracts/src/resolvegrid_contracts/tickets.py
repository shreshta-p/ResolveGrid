from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TicketStatus = Literal["open", "in_progress", "resolved", "closed", "reopened"]
TicketType = Literal["incident", "service_request"]
TicketPriority = Literal["low", "medium", "high", "urgent"]

# Explicit, centralized transition table -- the single source of truth for
# which ticket status changes are legal. Nothing outside this module should
# encode transition rules; both apps/api's enforcement and any future UI
# affordance (e.g. disabling illegal buttons) derive from this.
ALLOWED_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    "open": {"in_progress", "closed"},
    "in_progress": {"resolved", "open"},
    "resolved": {"closed", "reopened"},
    "closed": {"reopened"},
    "reopened": {"in_progress", "closed"},
}


def is_valid_transition(from_status: TicketStatus, to_status: TicketStatus) -> bool:
    """Return True if moving a ticket from from_status to to_status is legal."""
    return to_status in ALLOWED_TRANSITIONS.get(from_status, set())


class TicketCreateRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)
    type: TicketType
    priority: TicketPriority = "medium"
    queue_id: int


class TicketRead(BaseModel):
    id: int
    subject: str
    type: TicketType
    priority: TicketPriority
    status: TicketStatus
    queue_id: int
    requester_id: int
    assignee_id: int | None
    created_at: datetime
    updated_at: datetime


class TicketTransitionRequest(BaseModel):
    to_status: TicketStatus
    reason: str | None = Field(default=None, max_length=2000)


class TicketMessageCreate(BaseModel):
    body: str = Field(min_length=1, max_length=10_000)
