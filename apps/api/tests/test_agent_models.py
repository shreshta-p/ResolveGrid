import pytest
from sqlalchemy.exc import IntegrityError

from resolvegrid_api.models import AgentRun, Span


def test_agent_run_round_trips_with_spans(db_session):
    run = AgentRun(
        thread_id="thread-round-trip-1",
        input_text="What is the ticket escalation policy?",
    )
    db_session.add(run)
    db_session.flush()

    spans = [
        Span(agent_run_id=run.id, stage_name="classify_intent", status="success", latency_ms=120),
        Span(agent_run_id=run.id, stage_name="compose_response", status="success", latency_ms=430,
             detail_json='{"intent": "general_question"}'),
        Span(agent_run_id=run.id, stage_name="finalize", status="success", latency_ms=5),
    ]
    db_session.add_all(spans)
    db_session.flush()

    run.status = "completed"
    run.output_text = "The escalation policy is documented in the runbook."
    db_session.flush()

    fetched_run = db_session.get(AgentRun, run.id)
    assert fetched_run is not None
    assert fetched_run.thread_id == "thread-round-trip-1"
    assert fetched_run.principal_employee_id is None
    assert fetched_run.status == "completed"
    assert fetched_run.input_text == "What is the ticket escalation policy?"
    assert fetched_run.output_text == "The escalation policy is documented in the runbook."
    assert fetched_run.error_message is None
    assert fetched_run.completed_at is None  # not set by application code in this test

    fetched_spans = (
        db_session.query(Span)
        .filter(Span.agent_run_id == run.id)
        .order_by(Span.id)
        .all()
    )
    assert len(fetched_spans) == 3
    assert [s.stage_name for s in fetched_spans] == ["classify_intent", "compose_response", "finalize"]
    assert fetched_spans[1].detail_json == '{"intent": "general_question"}'
    assert all(s.status == "success" for s in fetched_spans)
    assert fetched_spans[0].latency_ms == 120


def test_agent_run_status_defaults_to_running(db_session):
    run = AgentRun(thread_id="thread-default-status", input_text="hello")
    db_session.add(run)
    db_session.flush()

    fetched = db_session.get(AgentRun, run.id)
    assert fetched.status == "running"


def test_agent_run_cache_hit_kind_requires_cache_hit_true(db_session):
    """Phase 12 Task 5 code-review follow-up: a real DB CHECK constraint
    (migration 0021, `ck_agent_run_cache_hit_kind_requires_cache_hit`) must
    make `cache_hit_kind` set alongside `cache_hit=False` structurally
    impossible to write, not merely unlikely-by-convention -- a row saying
    "semantically/exactly cache-hit" while simultaneously saying "this
    wasn't a cache hit at all" is nonsensical and must fail at the DB layer
    regardless of which future code path tries to write it.
    """
    invalid_run = AgentRun(
        thread_id="thread-invalid-cache-hit-kind",
        input_text="hello",
        cache_hit=False,
        cache_hit_kind="exact",
    )
    db_session.add(invalid_run)

    with pytest.raises(IntegrityError):
        db_session.flush()

    # Same Postgres-transaction-abort cleanup as the FK-enforcement test
    # below -- the session must be rolled back to a clean state before the
    # fixture's own teardown rollback runs.
    db_session.rollback()


def test_agent_run_cache_hit_true_with_cache_hit_kind_is_allowed(db_session):
    """The positive case: `cache_hit=True` with a real `cache_hit_kind`
    ("exact" or "semantic") is exactly the valid combination the CHECK
    constraint must still permit -- proves the constraint isn't
    overly-broad and blocking legitimate rows too.
    """
    valid_run = AgentRun(
        thread_id="thread-valid-cache-hit-kind",
        input_text="hello",
        cache_hit=True,
        cache_hit_kind="semantic",
    )
    db_session.add(valid_run)
    db_session.flush()  # must not raise

    fetched = db_session.get(AgentRun, valid_run.id)
    assert fetched.cache_hit is True
    assert fetched.cache_hit_kind == "semantic"


def test_span_agent_run_fk_is_enforced(db_session):
    """A Span referencing a nonexistent agent_run_id must fail to commit/flush."""
    orphan_span = Span(agent_run_id=999_999_999, stage_name="classify_intent", status="success", latency_ms=10)
    db_session.add(orphan_span)

    with pytest.raises(IntegrityError):
        db_session.flush()

    # Postgres aborts the transaction on a constraint violation; roll the
    # session back to a clean state so the fixture's own teardown rollback
    # doesn't warn about an already-deassociated transaction.
    db_session.rollback()
