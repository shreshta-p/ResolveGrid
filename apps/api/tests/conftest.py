import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)


@pytest.fixture
def db_session():
    """Transactional session: every write in a test is rolled back at the end.

    Use this for anything that reads/writes org data but shouldn't leave
    residue for other tests (cycle-guard checks, directory-API assertions).
    """
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


@pytest.fixture
def raw_db_session():
    """Real, committing session against the live database.

    Use this only when a test needs actual commit/rollback semantics to work
    (e.g. the seed generator's own delete-then-commit idempotency, which the
    rollback-based db_session fixture above would mask).
    """
    engine = create_engine(DATABASE_URL)
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
