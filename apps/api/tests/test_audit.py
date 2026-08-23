from sqlalchemy import select

from resolvegrid_api.audit import record_audit_event, verify_chain_integrity
from resolvegrid_api.models import AuditLog, Employee


def _make_employee(db_session, email: str) -> int:
    """AuditLog.actor_id is a real FK to employee.id, so tests need an actual
    row to reference rather than an arbitrary literal int."""
    employee = Employee(
        display_name="Test Employee", email=email, title="Engineer",
        hire_date="2024-01-01T00:00:00", timezone="America/Chicago",
    )
    db_session.add(employee)
    db_session.flush()
    return employee.id


def test_new_event_chains_to_whatever_was_previously_last(db_session):
    # Deliberately does NOT assume the table starts empty: apps/api's own
    # committed AuditLog rows are permanent (append-only, no cascade delete --
    # see resolvegrid_api.audit's module docstring and
    # resolvegrid_api.seed.generate_org's protection logic), so a local dev
    # database that has real prior ticket activity will NOT have an empty
    # audit_log table even though this test's own db_session fixture rolls
    # back everything IT writes. Capture whatever the actual previous state
    # is first, then assert the new entry correctly chains from it -- this
    # is correct whether that baseline is None (a genuinely fresh table, e.g.
    # in CI) or a real hash (a local dev DB with prior history).
    baseline = db_session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)).first()
    baseline_hash = baseline.record_hash if baseline is not None else None

    actor_id = _make_employee(db_session, "audit-test-1@example.test")
    entry = record_audit_event(
        db_session, actor_type="employee", actor_id=actor_id, action="ticket.create",
        entity_type="ticket", entity_id=1, after={"subject": "VPN down"},
    )
    assert entry.previous_record_hash == baseline_hash
    assert entry.record_hash is not None


def test_second_event_chains_to_first(db_session):
    actor_id_1 = _make_employee(db_session, "audit-test-2@example.test")
    actor_id_2 = _make_employee(db_session, "audit-test-3@example.test")
    first = record_audit_event(
        db_session, actor_type="employee", actor_id=actor_id_1, action="ticket.create",
        entity_type="ticket", entity_id=1, after={"subject": "VPN down"},
    )
    second = record_audit_event(
        db_session, actor_type="analyst", actor_id=actor_id_2, action="ticket.transition",
        entity_type="ticket", entity_id=1,
        before={"status": "open"}, after={"status": "in_progress"},
    )
    assert second.previous_record_hash == first.record_hash


def test_verify_chain_integrity_true_for_untampered_chain(db_session):
    actor_id = _make_employee(db_session, "audit-test-4@example.test")
    record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="a", entity_type="ticket", entity_id=1)
    record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="b", entity_type="ticket", entity_id=1)
    record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="c", entity_type="ticket", entity_id=1)
    assert verify_chain_integrity(db_session) is True


def test_verify_chain_integrity_false_if_a_row_is_tampered(db_session):
    actor_id = _make_employee(db_session, "audit-test-5@example.test")
    record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="a", entity_type="ticket", entity_id=1)
    tampered = record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="b", entity_type="ticket", entity_id=1)
    record_audit_event(db_session, actor_type="employee", actor_id=actor_id, action="c", entity_type="ticket", entity_id=1)

    # Simulate tampering: directly rewrite a historical row's action without
    # updating its hash (exactly what an attacker with raw DB access would do).
    tampered.action = "tampered"
    db_session.flush()

    assert verify_chain_integrity(db_session) is False


def test_metadata_round_trips(db_session):
    actor_id = _make_employee(db_session, "audit-test-6@example.test")
    entry = record_audit_event(
        db_session, actor_type="employee", actor_id=actor_id, action="ticket.create",
        entity_type="ticket", entity_id=1, metadata={"ip": "10.0.0.1"},
    )
    assert entry.metadata_json == '{"ip": "10.0.0.1"}'


def test_verify_chain_integrity_false_if_metadata_is_tampered(db_session):
    actor_id = _make_employee(db_session, "audit-test-7@example.test")
    tampered = record_audit_event(
        db_session, actor_type="employee", actor_id=actor_id, action="ticket.create",
        entity_type="ticket", entity_id=1, metadata={"ip": "10.0.0.1"},
    )

    # Simulate tampering with ONLY the metadata field -- before this fix,
    # this went completely undetected since metadata_json wasn't part of
    # the hashed payload at all.
    tampered.metadata_json = '{"ip": "6.6.6.6"}'
    db_session.flush()

    assert verify_chain_integrity(db_session) is False
