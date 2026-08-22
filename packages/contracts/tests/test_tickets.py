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


def test_in_progress_can_resolve_reopen_to_open_or_close_directly():
    # Direct in_progress -> closed matters for real "won't fix"/duplicate/spam
    # closures discovered mid-triage -- nothing was actually resolved, so
    # forcing a detour through "resolved" first would be semantically wrong.
    assert is_valid_transition("in_progress", "resolved") is True
    assert is_valid_transition("in_progress", "open") is True
    assert is_valid_transition("in_progress", "closed") is True


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


def test_self_transition_is_never_valid():
    for status in ALLOWED_TRANSITIONS:
        assert is_valid_transition(status, status) is False, f"{status} -> {status} should not be valid"


def test_unknown_status_strings_are_rejected_not_raised():
    # Literal only constrains static type-checkers -- nothing stops a raw
    # string from reaching this function at runtime (e.g. before Pydantic
    # validation runs). It must degrade to False, never raise.
    assert is_valid_transition("open", "not_a_real_status") is False  # type: ignore[arg-type]
    assert is_valid_transition("not_a_real_status", "open") is False  # type: ignore[arg-type]


def test_allowed_transitions_is_immutable():
    import pytest as _pytest

    with _pytest.raises(TypeError):
        ALLOWED_TRANSITIONS["open"] = frozenset()  # type: ignore[index]
    with _pytest.raises(AttributeError):
        ALLOWED_TRANSITIONS["open"].add("resolved")  # type: ignore[attr-defined]
