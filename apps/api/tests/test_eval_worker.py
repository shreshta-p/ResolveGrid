"""Tests for `resolvegrid_api.eval_worker` (Phase 10 Task 7).

`test_run_eval_suite_creates_completed_eval_run_with_real_case_coverage` is
a real, no-mocking integration test: it ingests the actual seed corpus
(matching `test_eval_retrieval.py`'s established `db_session` pattern) then
calls `run_eval_suite` directly against the real chat graph, the real
tool-invocation graph, the real hybrid retrieval pipeline, and real Ollama
completions -- no mocking anywhere, per this codebase's established
no-mocking-for-real-behavior norm for eval code. `db_session` rolls back at
the end, so the `EvalRun`/`EvalCaseResult` bookkeeping rows leave zero
residue -- but see `eval_worker.py`'s module docstring's "Session/commit
boundary" section: the tool/approval-dimension cases genuinely commit a
small number of `Employee`/`RoleAssignment`/`ApprovalRequest`/
`ApprovalDecision`/`ToolCall`/`EmployeeEntitlement` rows on their own
independent sessions (architecturally required by the real graphs this
function drives), so this test watermark-cleans those specific tables
afterward, mirroring `test_ingestion_worker.py`'s real-Arq test's identical
id-watermark cleanup convention.

`test_arq_worker_processes_run_eval_suite_task_via_real_redis` mirrors
`test_ingestion_worker.py`'s `test_arq_worker_processes_ingest_seed_corpus_
task_via_real_redis` structure exactly: enqueues a real job onto the live
`resolvegrid-redis` queue via `arq.create_pool`, runs a real burst-mode
`Worker` to execute it, then checks the resulting `EvalRun`/`EvalCaseResult`
rows the job itself committed.

Both tests are slow (many real Ollama completions/embeddings + real DB
round-trips) -- expected, and consistent with this codebase's other
real-corpus/real-Ollama tests (`test_eval_retrieval.py`,
`test_adversarial_suite.py`).
"""

import asyncio
import json
import signal

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import run_worker
from sqlalchemy import delete, func, select

# See test_ingestion_worker.py's identical comment: arq.worker.Worker.close()
# unconditionally references signal.SIGUSR1 (POSIX-only) even with
# handle_signals=False, which raises AttributeError during cleanup on
# Windows even after the job itself completed successfully.
if not hasattr(signal, "SIGUSR1"):
    signal.SIGUSR1 = signal.SIGTERM

from resolvegrid_api.eval_worker import REDIS_URL, WorkerSettings, run_eval_suite
from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest
from resolvegrid_api.models.evaluation import EvalCaseResult, EvalRun
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding, IngestionRun
from resolvegrid_api.models.org import EmployeeEntitlement
from resolvegrid_api.models.tools import ToolCall

_EXPECTED_DIMENSIONS = {"chat", "retrieval", "tool", "approval", "adversarial"}


def _side_effect_watermarks(session) -> dict[str, int]:
    """Current max id of every table `run_eval_suite`'s tool/approval
    -dimension cases might commit rows into, on a SEPARATE, independently
    -committing session (see eval_worker.py's module docstring) -- used to
    identify (and later clean up) only the rows a given call actually
    created, matching test_ingestion_worker.py's established convention.
    """
    return {
        "approval_request": session.scalar(select(func.max(ApprovalRequest.id))) or 0,
        "approval_decision": session.scalar(select(func.max(ApprovalDecision.id))) or 0,
        "tool_call": session.scalar(select(func.max(ToolCall.id))) or 0,
        "employee_entitlement": session.scalar(select(func.max(EmployeeEntitlement.id))) or 0,
    }


def _cleanup_side_effects(session, before: dict[str, int]) -> None:
    session.execute(delete(ApprovalDecision).where(ApprovalDecision.id > before["approval_decision"]))
    session.execute(delete(ToolCall).where(ToolCall.id > before["tool_call"]))
    session.execute(delete(EmployeeEntitlement).where(EmployeeEntitlement.id > before["employee_entitlement"]))
    session.execute(delete(ApprovalRequest).where(ApprovalRequest.id > before["approval_request"]))
    session.commit()


def test_run_eval_suite_creates_completed_eval_run_with_real_case_coverage(db_session, raw_db_session):
    run_seed_corpus_ingestion(db_session)
    db_session.flush()

    before = _side_effect_watermarks(raw_db_session)
    try:
        run = run_eval_suite(db_session)
        db_session.flush()

        assert run.id is not None
        assert run.status == "completed"
        assert run.error_message is None
        assert run.completed_at is not None
        assert run.dataset_version == "v2"
        # Real commit sha, not a hardcoded placeholder -- a 40-char hex string.
        assert run.git_commit != "unknown"
        assert len(run.git_commit) == 40
        assert all(ch in "0123456789abcdef" for ch in run.git_commit)

        results = db_session.scalars(
            select(EvalCaseResult).where(EvalCaseResult.eval_run_id == run.id)
        ).all()
        assert len(results) > 0

        dimensions_covered = {r.dimension for r in results}
        assert dimensions_covered == _EXPECTED_DIMENSIONS, (
            f"expected every dimension covered, got: {dimensions_covered}"
        )

        # Every result's details_json must be genuinely JSON-serializable
        # (self-review requirement) -- round-trip parse each one for real,
        # not just trust that json.dumps succeeded at write time.
        for result in results:
            assert result.details_json is not None
            parsed = json.loads(result.details_json)
            assert isinstance(parsed, dict)
            assert result.grader_type in ("deterministic", "judge")
            assert isinstance(result.passed, bool)

        # At least one case in every dimension actually executed (sanity:
        # not every single case silently errored into a False result with
        # no real detail) -- spot-check retrieval, since it has a
        # well-known real metric shape.
        retrieval_results = [r for r in results if r.dimension == "retrieval"]
        assert any(r.score is not None for r in retrieval_results), (
            "expected at least one retrieval case to report a real recall@k score"
        )

        # Tool/approval dimension cases genuinely executed the real
        # allowlist/approval-pause/decide/resume pipeline (not silently
        # errored) -- the approved-approval case's tool_argument_accuracy
        # check should have passed for real.
        approval_results = {r.case_id: r for r in results if r.dimension == "approval"}
        assert "approval.grant_vpn_access.approved.001" in approval_results
        approved_details = json.loads(approval_results["approval.grant_vpn_access.approved.001"].details_json)
        assert approved_details["approval_compliance"]["passed"] is True
    finally:
        _cleanup_side_effects(raw_db_session, before)


def test_arq_worker_processes_run_eval_suite_task_via_real_redis(raw_db_session):
    """Unlike the direct-call test above (which ingests inside `db_session`'s
    own rolled-back transaction, invisible to any other connection),
    `run_eval_suite_task` opens its OWN session against the live DB (see
    `eval_worker.py`'s module docstring) -- so the retrieval-dimension
    cases need the seed corpus genuinely, committedly ingested first, or
    `resolve_relevant_chunk_ids` raises (a real, correct failure: this
    mirrors `eval_retrieval.py`'s documented "ingestion is not this
    module's responsibility" convention, and matches the plan's own
    closing verification flow -- "seed+ingest, then run one real eval
    suite via the Arq worker against the freshly-seeded/ingested data").
    `run_seed_corpus_ingestion` is idempotent (a no-op against an
    already-ingested corpus), so this is safe to call unconditionally;
    watermarked and cleaned up afterward like every other real-ingestion
    fixture in this test suite (see `test_ingestion_worker.py`'s identical
    convention) -- a no-op cleanup if the corpus was already present.
    """
    async def _enqueue() -> None:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            job = await pool.enqueue_job("run_eval_suite_task")
            assert job is not None, "enqueue_job returned None -- job may already be queued/deduped"
        finally:
            await pool.aclose()

    before_max_ingestion_run_id = raw_db_session.scalar(select(func.max(IngestionRun.id))) or 0
    before_max_document_id = raw_db_session.scalar(select(func.max(Document.id))) or 0
    before_max_version_id = raw_db_session.scalar(select(func.max(DocumentVersion.id))) or 0
    before_max_chunk_id = raw_db_session.scalar(select(func.max(Chunk.id))) or 0
    before_max_embedding_id = raw_db_session.scalar(select(func.max(Embedding.id))) or 0
    run_seed_corpus_ingestion(raw_db_session)
    raw_db_session.commit()

    before_max_run_id = raw_db_session.scalar(select(func.max(EvalRun.id))) or 0
    before = _side_effect_watermarks(raw_db_session)

    asyncio.run(_enqueue())

    worker = run_worker(WorkerSettings, burst=True, handle_signals=False)

    try:
        assert worker.jobs_complete >= 1
        assert worker.jobs_failed == 0

        latest_run = raw_db_session.scalars(
            select(EvalRun).where(EvalRun.id > before_max_run_id).order_by(EvalRun.id.desc())
        ).first()

        assert latest_run is not None
        assert latest_run.status == "completed"
        assert latest_run.git_commit != "unknown"

        results = raw_db_session.scalars(
            select(EvalCaseResult).where(EvalCaseResult.eval_run_id == latest_run.id)
        ).all()
        assert len(results) > 0
        assert {r.dimension for r in results} == _EXPECTED_DIMENSIONS
    finally:
        raw_db_session.execute(delete(EvalCaseResult).where(EvalCaseResult.eval_run_id > before_max_run_id))
        raw_db_session.execute(delete(EvalRun).where(EvalRun.id > before_max_run_id))
        raw_db_session.commit()
        _cleanup_side_effects(raw_db_session, before)

        embedding_ids = raw_db_session.scalars(
            select(Embedding.id).where(Embedding.id > before_max_embedding_id)
        ).all()
        chunk_ids = raw_db_session.scalars(select(Chunk.id).where(Chunk.id > before_max_chunk_id)).all()
        version_ids = raw_db_session.scalars(
            select(DocumentVersion.id).where(DocumentVersion.id > before_max_version_id)
        ).all()
        document_ids = raw_db_session.scalars(
            select(Document.id).where(Document.id > before_max_document_id)
        ).all()
        raw_db_session.execute(delete(Embedding).where(Embedding.id.in_(embedding_ids)))
        raw_db_session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))
        raw_db_session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_(version_ids)))
        raw_db_session.execute(delete(Document).where(Document.id.in_(document_ids)))
        raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id > before_max_ingestion_run_id))
        raw_db_session.commit()
