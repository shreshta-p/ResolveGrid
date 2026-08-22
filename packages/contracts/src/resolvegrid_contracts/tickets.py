from datetime import datetime
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, Field

TicketStatus = Literal["open", "in_progress", "resolved", "closed", "reopened"]
TicketType = Literal["incident", "service_request"]
TicketPriority = Literal["low", "medium", "high", "urgent"]

# Explicit, centralized transition table -- the single source of truth for
# which ticket status changes are legal. Nothing outside this module should
# encode transition rules; both apps/api's enforcement and any future UI
# affordance (e.g. disabling illegal buttons) derive from this.
#
# Immutable (MappingProxyType + frozenset values) -- this is a shared,
# cross-package module-level constant used for a real security/business-rule
# boundary (apps/api's ticket-transition enforcement), so it must not be
# mutable in the way packages/authz's Decision/Principal already guard
# against via frozen=True dataclasses and tuple fields. A stray
# `ALLOWED_TRANSITIONS["open"].add(...)` anywhere in the process would
# otherwise silently corrupt enforcement for every caller.
ALLOWED_TRANSITIONS: Mapping[TicketStatus, frozenset[TicketStatus]] = MappingProxyType(
    {
        "open": frozenset({"in_progress", "closed"}),
        "in_progress": frozenset({"resolved", "open", "closed"}),
        "resolved": frozenset({"closed", "reopened"}),
        "closed": frozenset({"reopened"}),
        "reopened": frozenset({"in_progress", "closed"}),
    }
)


def is_valid_transition(from_status: TicketStatus, to_status: TicketStatus) -> bool:
    """Return True if moving a ticket from from_status to to_status is legal."""
    return to_status in ALLOWED_TRANSITIONS.get(from_status, frozenset())


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
