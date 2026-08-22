from resolvegrid_api.models import Employee, RoleAssignment
from resolvegrid_api.seed import generate_org


def test_seed_is_deterministic_and_idempotent(raw_db_session):
    generate_org(raw_db_session, seed=123, num_employees=20)
    first_emails = [
        e.email for e in raw_db_session.query(Employee).order_by(Employee.id).all()
    ]
    first_role_count = raw_db_session.query(RoleAssignment).count()

    # Re-running with the same seed must wipe and reproduce identical data.
    generate_org(raw_db_session, seed=123, num_employees=20)
    second_emails = [
        e.email for e in raw_db_session.query(Employee).order_by(Employee.id).all()
    ]
    second_role_count = raw_db_session.query(RoleAssignment).count()

    assert len(first_emails) == 20
    assert first_emails == second_emails
    assert first_role_count == second_role_count
    assert first_role_count > 0


def test_different_seeds_produce_different_data(raw_db_session):
    generate_org(raw_db_session, seed=1, num_employees=10)
    emails_seed_1 = {e.email for e in raw_db_session.query(Employee).all()}

    generate_org(raw_db_session, seed=2, num_employees=10)
    emails_seed_2 = {e.email for e in raw_db_session.query(Employee).all()}

    assert emails_seed_1 != emails_seed_2
