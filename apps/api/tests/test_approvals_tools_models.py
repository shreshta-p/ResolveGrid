"""Phase 9 Task 1 smoke test: every new model round-trips a row.

Schema-only proof (no business logic yet, per the task's scope) that
ApprovalRequest/ApprovalDecision/ApprovalPolicy, ToolCall, and
AccessGroup/Entitlement/EmployeeEntitlement can each be created, committed
(flushed within the transactional db_session fixture), and queried back
with their FK relationships intact.
"""
from datetime import datetime, timedelta, timezone

from resolvegrid_api.models import (
    AccessGroup,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    Department,
    Employee,
    EmployeeEntitlement,
    Entitlement,
    Location,
    Queue,
    Ticket,
    ToolCall,
)


def _make_employee(db_session, suffix: str) -> Employee:
    location = Location(name=f"Test HQ Approvals {suffix}", region="US", timezone="America/Chicago")
    department = Department(name=f"Test Dept Approvals {suffix}")
    db_session.add_all([location, department])
    db_session.flush()

    employee = Employee(
        display_name=f"Employee {suffix}",
        email=f"employee.{suffix}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(employee)
    db_session.flush()
    return employee


def test_approval_request_and_decision_round_trip(db_session):
    requester = _make_employee(db_session, "req")
    approver = _make_employee(db_session, "appr")

    queue = Queue(name="Test Queue Approvals", department_id=requester.department_id)
    db_session.add(queue)
    db_session.flush()

    ticket = Ticket(
        subject="Grant VPN access", type="request", queue_id=queue.id, requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    approval_request = ApprovalRequest(
        ticket_id=ticket.id,
        agent_run_id="thread-approval-round-trip-1",
        action_type="grant_vpn_access",
        action_params_json='{"employee_id": 1, "justification": "new hire"}',
        bound_evidence_refs_json='["chunk:42"]',
        risk_context="low",
        snapshot_hash="a" * 64,
        requested_by_id=requester.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db_session.add(approval_request)
    db_session.flush()

    fetched_request = db_session.get(ApprovalRequest, approval_request.id)
    assert fetched_request is not None
    assert fetched_request.status == "pending"  # default
    assert fetched_request.ticket_id == ticket.id
    assert fetched_request.agent_run_id == "thread-approval-round-trip-1"
    assert fetched_request.snapshot_hash == "a" * 64
    assert fetched_request.created_at is not None
    assert fetched_request.updated_at is not None

    decision = ApprovalDecision(
        approval_request_id=approval_request.id,
        approver_id=approver.id,
        decision="approved",
        comment="looks fine",
    )
    db_session.add(decision)
    db_session.flush()

    fetched_decision = db_session.get(ApprovalDecision, decision.id)
    assert fetched_decision is not None
    assert fetched_decision.approval_request_id == approval_request.id
    assert fetched_decision.approver_id == approver.id
    assert fetched_decision.decision == "approved"
    assert fetched_decision.decided_at is not None


def test_approval_policy_round_trips(db_session):
    policy = ApprovalPolicy(
        action_type="grant_vpn_access",
        stages_json='[{"role": "analyst", "scope": "department"}, {"role": "manager", "scope": "global"}]',
        description="Peer review then manager sign-off",
    )
    db_session.add(policy)
    db_session.flush()

    fetched = db_session.get(ApprovalPolicy, policy.id)
    assert fetched is not None
    assert fetched.action_type == "grant_vpn_access"
    assert fetched.description == "Peer review then manager sign-off"


def test_tool_call_round_trips_with_approval_link(db_session):
    requester = _make_employee(db_session, "tool")

    approval_request = ApprovalRequest(
        agent_run_id="thread-tool-call-round-trip-1",
        action_type="grant_vpn_access",
        action_params_json='{"employee_id": 1}',
        status="approved",
        snapshot_hash="b" * 64,
        requested_by_id=requester.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db_session.add(approval_request)
    db_session.flush()

    tool_call = ToolCall(
        agent_run_id="thread-tool-call-round-trip-1",
        tool_name="grant_vpn_access",
        tool_version="1.0",
        input_params_json='{"employee_id": 1, "justification": "new hire"}',
        output_json='{"employee_entitlement_id": 1}',
        status="success",
        idempotency_key=f"approval:{approval_request.id}",
        approval_request_id=approval_request.id,
    )
    db_session.add(tool_call)
    db_session.flush()

    fetched = db_session.get(ToolCall, tool_call.id)
    assert fetched is not None
    assert fetched.tool_name == "grant_vpn_access"
    assert fetched.status == "success"
    assert fetched.idempotency_key == f"approval:{approval_request.id}"
    assert fetched.approval_request_id == approval_request.id
    assert fetched.error_taxonomy_code is None


def test_entitlement_grant_round_trips(db_session):
    employee = _make_employee(db_session, "ent")

    queue = Queue(name="Test Queue Entitlement", department_id=employee.department_id)
    db_session.add(queue)
    db_session.flush()

    source_ticket = Ticket(
        subject="VPN access request", type="request", queue_id=queue.id, requester_id=employee.id,
    )
    db_session.add(source_ticket)
    db_session.flush()

    access_group = AccessGroup(name="Networking", description="Network-related access")
    db_session.add(access_group)
    db_session.flush()

    entitlement = Entitlement(
        access_group_id=access_group.id, name="VPN Access", description="Corporate VPN access",
    )
    db_session.add(entitlement)
    db_session.flush()

    grant = EmployeeEntitlement(
        employee_id=employee.id,
        entitlement_id=entitlement.id,
        source_ticket_id=source_ticket.id,
    )
    db_session.add(grant)
    db_session.flush()

    fetched = db_session.get(EmployeeEntitlement, grant.id)
    assert fetched is not None
    assert fetched.employee_id == employee.id
    assert fetched.entitlement_id == entitlement.id
    assert fetched.source_ticket_id == source_ticket.id
    assert fetched.granted_at is not None
    assert fetched.revoked_at is None

    fetched_entitlement = db_session.get(Entitlement, entitlement.id)
    assert fetched_entitlement.access_group_id == access_group.id
    assert fetched_entitlement.name == "VPN Access"
