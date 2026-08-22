import pytest
from pydantic import ValidationError

from resolvegrid_contracts.tickets import (
    ALLOWED_TRANSITIONS,
    TicketCreateRequest,
    TicketMessageCreate,
    TicketTransitionRequest,
    is_valid_transition,
)


def test_open_can_transition_to_in_progress_or_closed():
    assert is_valid_transition("open", "in_progress") is True
    assert is_valid_transition("open", "closed") is True
    assert is_valid_transition("open", "resolved") is False


def test_closed_can_only_reopen():
    assert is_valid_transition("closed", "reopened") is True
    assert is_valid_transition("closed", "in_progress") is False
    assert is_valid_transition("closed", "open") is False


def test_resolved_can_close_or_reopen():
    assert is_valid_transition("resolved", "closed") is True
    assert is_valid_transition("resolved", "reopened") is True
    assert is_valid_transition("resolved", "in_progress") is False


def test_every_status_has_at_least_one_legal_transition():
    # Guards against a future status being added to ALLOWED_TRANSITIONS'
    # keys with an empty set, which would silently make it a dead end.
    for status, targets in ALLOWED_TRANSITIONS.items():
        assert len(targets) > 0, f"{status} has no legal outgoing transition"


def test_ticket_create_request_rejects_empty_subject():
    with pytest.raises(ValidationError):
        TicketCreateRequest(subject="", body="something broke", type="incident", queue_id=1)


def test_ticket_create_request_rejects_empty_body():
    with pytest.raises(ValidationError):
        TicketCreateRequest(subject="VPN down", body="", type="incident", queue_id=1)


def test_ticket_create_request_accepts_valid_input():
    req = TicketCreateRequest(subject="VPN down", body="Can't connect since this morning", type="incident", queue_id=1)
    assert req.priority == "medium"  # default


def test_ticket_create_request_rejects_invalid_type():
    with pytest.raises(ValidationError):
        TicketCreateRequest(subject="x", body="y", type="not_a_real_type", queue_id=1)


def test_transition_request_rejects_invalid_status_literal():
    with pytest.raises(ValidationError):
        TicketTransitionRequest(to_status="not_a_real_status")


def test_message_create_rejects_empty_body():
    with pytest.raises(ValidationError):
        TicketMessageCreate(body="")
