"""Tests for `resolvegrid_api.operational_adapters.entitlements` (Phase 9
Task 3).

Covers the read-only lookup's revoked-grant exclusion, and the two
idempotency properties Task 9's restart-mid-approval test will later
depend on being real: `ensure_vpn_entitlement_seeded` never creates a
duplicate `Entitlement` row, and `grant_vpn_access` never creates a
duplicate `EmployeeEntitlement` row for the same employee.
"""

from resolvegrid_api.models.org import (
    AccessGroup,
    Department,
    Employee,
    EmployeeEntitlement,
    Entitlement,
    Location,
)
from resolvegrid_api.operational_adapters.entitlements import (
    VPN_ENTITLEMENT_NAME,
    EntitlementSummary,
    ensure_vpn_entitlement_seeded,
    grant_vpn_access,
    lookup_employee_entitlements,
)
from sqlalchemy import select


def _make_employee(db_session, suffix: str) -> Employee:
    location = Location(name=f"Test HQ Entitlements {suffix}", region="US", timezone="America/Chicago")
    department = Department(name=f"Test Dept Entitlements {suffix}")
    db_session.add_all([location, department])
    db_session.flush()

    employee = Employee(
        display_name=f"Employee {suffix}",
        email=f"employee.entitlements.{suffix}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(employee)
    db_session.flush()
    return employee


def test_lookup_employee_entitlements_returns_nothing_for_employee_with_no_grants(db_session):
    employee = _make_employee(db_session, "none")

    result = lookup_employee_entitlements(db_session, employee.id)

    assert result == []


def test_lookup_employee_entitlements_excludes_revoked_and_returns_active_summaries(db_session):
    employee = _make_employee(db_session, "mixed")

    access_group = AccessGroup(name="Test Access Group Mixed", description="Test group")
    db_session.add(access_group)
    db_session.flush()

    active_entitlement = Entitlement(access_group_id=access_group.id, name="Active Tool")
    revoked_entitlement = Entitlement(access_group_id=access_group.id, name="Revoked Tool")
    db_session.add_all([active_entitlement, revoked_entitlement])
    db_session.flush()

    active_grant = EmployeeEntitlement(employee_id=employee.id, entitlement_id=active_entitlement.id)
    revoked_grant = EmployeeEntitlement(
        employee_id=employee.id,
        entitlement_id=revoked_entitlement.id,
        revoked_at="2024-06-01T00:00:00",
    )
    db_session.add_all([active_grant, revoked_grant])
    db_session.flush()

    result = lookup_employee_entitlements(db_session, employee.id)

    assert len(result) == 1
    summary = result[0]
    assert isinstance(summary, EntitlementSummary)
    assert summary.entitlement_id == active_entitlement.id
    assert summary.entitlement_name == "Active Tool"
    assert summary.access_group_name == "Test Access Group Mixed"
    assert summary.granted_at is not None


def test_ensure_vpn_entitlement_seeded_is_idempotent(db_session):
    first = ensure_vpn_entitlement_seeded(db_session)
    second = ensure_vpn_entitlement_seeded(db_session)

    assert first.id == second.id
    assert first.name == VPN_ENTITLEMENT_NAME

    rows = db_session.execute(
        select(Entitlement).where(Entitlement.name == VPN_ENTITLEMENT_NAME)
    ).scalars().all()
    assert len(rows) == 1


def test_grant_vpn_access_is_idempotent_for_the_same_employee(db_session):
    employee = _make_employee(db_session, "grant-once")

    first = grant_vpn_access(db_session, employee.id, justification="new hire onboarding")
    second = grant_vpn_access(db_session, employee.id, justification="new hire onboarding")

    assert first.id == second.id

    entitlement = ensure_vpn_entitlement_seeded(db_session)
    rows = (
        db_session.execute(
            select(EmployeeEntitlement).where(
                EmployeeEntitlement.employee_id == employee.id,
                EmployeeEntitlement.entitlement_id == entitlement.id,
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


def test_grant_vpn_access_creates_separate_rows_for_different_employees(db_session):
    employee_a = _make_employee(db_session, "grant-a")
    employee_b = _make_employee(db_session, "grant-b")

    grant_a = grant_vpn_access(db_session, employee_a.id, justification="role change")
    grant_b = grant_vpn_access(db_session, employee_b.id, justification="role change")

    assert grant_a.id != grant_b.id
    assert grant_a.employee_id == employee_a.id
    assert grant_b.employee_id == employee_b.id

    entitlement = ensure_vpn_entitlement_seeded(db_session)
    rows = (
        db_session.execute(
            select(EmployeeEntitlement).where(EmployeeEntitlement.entitlement_id == entitlement.id)
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
