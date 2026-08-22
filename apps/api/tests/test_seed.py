from sqlalchemy import func, select

from resolvegrid_api.models import Employee, RoleAssignment
from resolvegrid_api.seed import generate_org


def test_seed_is_deterministic_and_idempotent(raw_db_session):
    generate_org(raw_db_session, seed=123, num_employees=20)
    first_emails = raw_db_session.scalars(select(Employee.email).order_by(Employee.id)).all()
    first_role_count = raw_db_session.scalar(select(func.count()).select_from(RoleAssignment))

    # Re-running with the same seed must wipe and reproduce identical data.
    generate_org(raw_db_session, seed=123, num_employees=20)
    second_emails = raw_db_session.scalars(select(Employee.email).order_by(Employee.id)).all()
    second_role_count = raw_db_session.scalar(select(func.count()).select_from(RoleAssignment))

    assert len(first_emails) == 20
    assert first_emails == second_emails
    assert first_role_count == second_role_count
    assert first_role_count == 4  # manager_pool_size (20 // 5) role assignments: 1 admin + 3 dept-scoped


def test_different_seeds_produce_different_data(raw_db_session):
    generate_org(raw_db_session, seed=1, num_employees=10)
    emails_seed_1 = set(raw_db_session.scalars(select(Employee.email)).all())

    generate_org(raw_db_session, seed=2, num_employees=10)
    emails_seed_2 = set(raw_db_session.scalars(select(Employee.email)).all())

    assert emails_seed_1 != emails_seed_2
