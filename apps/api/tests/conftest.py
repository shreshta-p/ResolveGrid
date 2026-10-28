import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)


@pytest.fixture(autouse=True)
def _bound_diagnostic_specialist_llm_calls_for_tests(monkeypatch):
    """Phase 13 Task 6 real CI finding: `diagnostic_agent._MAX_LLM_CALLS=60`
    is a deliberate PRODUCTION safety bound (ADR 0004 Finding 5 -- a real,
    reasoned middle ground between Task 1's observed single-tool worst
    case and this specialist's 4-tool compounding worst case), not a test
    -timing knob -- left untouched here for real production behavior.

    But a worst-case real run approaching that full cap, combined with
    CI's real, measured per-call speed (~10-30s/call on a CPU-only
    runner, vs ~1-3s locally with a warm GPU), can take over an hour BY
    ITSELF -- confirmed for real, twice, in this phase's own CI history
    (`test_diagnose_ticket_real_end_to_end_against_live_stack` alone ran
    33m39s before failing in run 40668281723; the whole `api` job was
    cancelled at the 75-minute timeout in run 40717788981 without even
    reaching every real diagnostic test). Raising the timeout further was
    already rejected once (Phase 10's own established precedent: reduce
    real call volume, don't just inflate the budget) and a THIRD blind
    increase would risk the same "90+ minutes for one job" outcome that
    precedent explicitly avoided.

    This autouse fixture lowers `_MAX_LLM_CALLS` to 15 for the DURATION OF
    THE TEST SUITE ONLY (monkeypatch reverts it automatically after each
    test) -- every real diagnostic test still exercises the exact same
    real mechanism (real `Agent`/`Runner`, real tool calls, real
    citation/safety checks, real degrade-on-hallucination/degrade-on-
    limit-exceeded behavior proven by the dedicated regression tests
    above), just with a tighter real bound on how many pathological
    real-model retries one test is allowed to absorb before reporting a
    bounded, honest (if content-incomplete) result instead of running for
    up to an hour. 15 is still comfortably above ADR 0003 Finding 1's
    single-tool worst case divided across this specialist's 4 tools in
    the common (non-pathological) case, while bounding a genuinely
    pathological run's CI wall-clock cost to a small, fixed multiple of a
    single real call's CI-observed latency, not an open-ended one.
    """
    from resolvegrid_api import diagnostic_agent

    monkeypatch.setattr(diagnostic_agent, "_MAX_LLM_CALLS", 15)


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
