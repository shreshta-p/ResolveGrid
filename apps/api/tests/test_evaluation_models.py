"""Phase 10 Task 1 smoke test: EvalRun/EvalCaseResult round-trip, FK
constraint, index existence, and status default.

Mirrors test_approvals_tools_models.py's shape: schema-only proof (no
business logic yet, per this task's scope) that the two new tables can be
created, committed (flushed within the transactional db_session fixture),
queried back with their FK relationship intact, and enforce the FK
constraint / carry the expected index.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from resolvegrid_api.models import EvalCaseResult, EvalRun


def _make_eval_run(db_session, *, status: str | None = None) -> EvalRun:
    kwargs = dict(
        dataset_version="v2",
        prompt_version="1.0",
        workflow_version="1.0",
        retriever_version="1.0",
        reranker_version="1.0",
        embeddings_version="nomic-embed-text",
        generation_model_version="1.0",
        judge_version="1.0",
        tool_schema_version="1.0",
        git_commit="a" * 40,
    )
    if status is not None:
        kwargs["status"] = status
    eval_run = EvalRun(**kwargs)
    db_session.add(eval_run)
    db_session.flush()
    return eval_run


def test_eval_run_status_defaults_to_running(db_session):
    eval_run = _make_eval_run(db_session)

    fetched = db_session.get(EvalRun, eval_run.id)
    assert fetched is not None
    assert fetched.status == "running"  # default
    assert fetched.started_at is not None
    assert fetched.completed_at is None
    assert fetched.error_message is None


def test_eval_case_result_round_trips_with_eval_run_link(db_session):
    eval_run = _make_eval_run(db_session, status="completed")
    eval_run.completed_at = datetime.now(timezone.utc)
    db_session.flush()

    result = EvalCaseResult(
        eval_run_id=eval_run.id,
        case_id="retrieval.vpn.001",
        dimension="retrieval",
        passed=True,
        score=1.0,
        grader_type="deterministic",
        details_json='{"recall_at_k": 1.0}',
    )
    db_session.add(result)
    db_session.flush()

    fetched = db_session.get(EvalCaseResult, result.id)
    assert fetched is not None
    assert fetched.eval_run_id == eval_run.id
    assert fetched.case_id == "retrieval.vpn.001"
    assert fetched.dimension == "retrieval"
    assert fetched.passed is True
    assert fetched.score == 1.0
    assert fetched.grader_type == "deterministic"
    assert fetched.created_at is not None


def test_eval_case_result_score_and_details_are_nullable(db_session):
    eval_run = _make_eval_run(db_session)

    result = EvalCaseResult(
        eval_run_id=eval_run.id,
        case_id="tool.grant_vpn_access.001",
        dimension="tool",
        passed=False,
        grader_type="deterministic",
    )
    db_session.add(result)
    db_session.flush()

    fetched = db_session.get(EvalCaseResult, result.id)
    assert fetched is not None
    assert fetched.score is None
    assert fetched.details_json is None


def test_eval_case_result_rejects_missing_eval_run_fk(db_session):
    result = EvalCaseResult(
        eval_run_id=999_999_999,  # no such eval_run row
        case_id="adversarial.injected_document.001",
        dimension="adversarial",
        passed=True,
        grader_type="deterministic",
    )
    db_session.add(result)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_eval_case_result_has_eval_run_id_index(db_session):
    inspector = inspect(db_session.bind)
    index_names = {ix["name"] for ix in inspector.get_indexes("eval_case_result")}
    assert "ix_eval_case_result_eval_run_id" in index_names
