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


def test_first_event_has_no_previous_hash(db_session):
    actor_id = _make_employee(db_session, "audit-test-1@example.test")
    entry = record_audit_event(
        db_session, actor_type="employee", actor_id=actor_id, action="ticket.create",
        entity_type="ticket", entity_id=1, after={"subject": "VPN down"},
    )
    assert entry.previous_record_hash is None
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
