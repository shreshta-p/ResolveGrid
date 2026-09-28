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
from datetime import datetime

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import run_worker
from sqlalchemy import delete, func, select, text

# See test_ingestion_worker.py's identical comment: arq.worker.Worker.close()
# unconditionally references signal.SIGUSR1 (POSIX-only) even with
# handle_signals=False, which raises AttributeError during cleanup on
# Windows even after the job itself completed successfully.
if not hasattr(signal, "SIGUSR1"):
    signal.SIGUSR1 = signal.SIGTERM

import resolvegrid_api.eval_worker as eval_worker_module
from resolvegrid_api.eval_worker import REDIS_URL, WorkerSettings, run_eval_suite
from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest, Employee, ModelCall
from resolvegrid_api.models.evaluation import EvalCaseResult, EvalRun
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding, IngestionRun
from resolvegrid_api.models.org import EmployeeEntitlement
from resolvegrid_api.models.tools import ToolCall

_EXPECTED_DIMENSIONS = {"chat", "retrieval", "tool", "approval", "adversarial"}

# Real, previously-undiscovered CI-cost problem found during Phase 10 Task
# 9's fresh-state verification (see docs/DECISION_LOG.md's 2026-09-11
# entry): qwen3:14b text generation is dramatically slower on GitHub's
# CPU-only hosted runners than on local dev hardware. Each of this file's
# three real-model tests independently runs the FULL ~49-case suite
# (chat/tool/approval dimensions all need real generation calls), and all
# three together pushed the `api` CI job's pytest step past even a
# generous 40-minute timeout, forcing a hard cancellation. Since the
# actual PURPOSE of these three tests is proving the batch runner's
# wiring is correct (every dimension gets real coverage, one case's
# failure doesn't abort the batch, the real Arq round-trip works) --
# not exhaustively grading all 49 cases every CI run -- each is
# restricted, via `_patch_small_case_sample` below, to this small,
# every-dimension-covered representative subset. Every sampled case still
# runs for real (real DB, real graphs, real Ollama/LiteLLM) -- only the
# CASE COUNT is reduced, not the reality of what each remaining case
# exercises. The full 49-case suite remains fully exercisable via the
# documented production entry point (`uv run --package resolvegrid-api
# python -m resolvegrid_api.eval_worker`) and was genuinely, exhaustively
# run that way during this same task's fresh-state verification -- see
# docs/PROGRESS.md's Phase 10 row for that real EvalRun's per-dimension
# results.
_CI_SAMPLE_CASE_IDS = frozenset(
    {
        "chat.greeting.001",
        "retrieval.public_doc.001",
        "retrieval.public_doc.002",
        "tool.grant_vpn_access.001",
        "approval.grant_vpn_access.approved.001",
        "approval.grant_vpn_access.rejected.001",
        "adversarial.injected_document.001",
        "adversarial.cross_user_data_request.001",
        "adversarial.conflicting_stale_policy.001",
        "adversarial.unsupported_question.001",
        "adversarial.fabricated_ticket_id.001",
        "adversarial.malformed_tool_call.001",
        "adversarial.simulated_provider_outage.001",
        "adversarial.duplicate_approval_replay.001",
        "adversarial.expired_approval_replay.001",
        "adversarial.empty_retrieval_result.001",
    }
)


def _patch_small_case_sample(monkeypatch) -> None:
    """Filters `_load_all_cases`'s REAL output (real JSONL parse, no fake
    case list) down to `_CI_SAMPLE_CASE_IDS` for the duration of one test.
    See `_CI_SAMPLE_CASE_IDS`'s comment above for why this exists.
    """
    real_load_all_cases = eval_worker_module._load_all_cases

    def _small_case_loader(dataset_version):
        return [c for c in real_load_all_cases(dataset_version) if c.case_id in _CI_SAMPLE_CASE_IDS]

    monkeypatch.setattr(eval_worker_module, "_load_all_cases", _small_case_loader)


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


def test_run_eval_suite_creates_completed_eval_run_with_real_case_coverage(monkeypatch, db_session, raw_db_session):
    _patch_small_case_sample(monkeypatch)
    run_seed_corpus_ingestion(db_session)
    db_session.flush()

    before = _side_effect_watermarks(raw_db_session)
    # Phase 11 Task 3: watermark ModelCall.id too -- this test's CI sample
    # set (_CI_SAMPLE_CASE_IDS above) includes "chat.greeting.001", so this
    # run genuinely exercises the chat dimension's real graph invocation
    # (_execute_graph_dimension_cases -> _run_chat_case), which is exactly
    # where eval_worker.py's own real ModelCall-logging closures (built
    # fresh per suite run, distinct from main.py's app-lifetime ones -- see
    # that module's docstring) get exercised. A targeted assertion added to
    # this ALREADY-real, already-slow test, per this task's brief, rather
    # than a whole new test that would re-pay the same real-Ollama/real-DB
    # setup cost just to prove the same wiring.
    before_max_model_call_id = raw_db_session.scalar(select(func.max(ModelCall.id))) or 0
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

        # Phase 11 Task 3's core exit criterion for the eval-worker call
        # site: this run's real chat-dimension case ("chat.greeting.001")
        # must have produced real ModelCall rows for BOTH classify_intent
        # and compose_response, via eval_worker.py's OWN real logging
        # closures (confirmed distinct from main.py's -- see that module's
        # docstring) -- not just main.py's app-lifetime chat traffic.
        new_model_calls = raw_db_session.scalars(
            select(ModelCall).where(ModelCall.id > before_max_model_call_id).order_by(ModelCall.id)
        ).all()
        eval_chat_calls = [c for c in new_model_calls if c.purpose.startswith("eval.chat.")]
        eval_chat_purposes = [c.purpose for c in eval_chat_calls]
        assert "eval.chat.classify_intent" in eval_chat_purposes
        assert "eval.chat.compose_response" in eval_chat_purposes
        # Code-review fix: also confirm these rows are success-shaped, not
        # just present -- an error-shaped row (e.g. a real Ollama timeout)
        # would still satisfy the membership checks above alone.
        assert all(c.status == "success" for c in eval_chat_calls)
    finally:
        _cleanup_side_effects(raw_db_session, before)


def test_run_eval_suite_isolates_a_single_case_failure_and_still_completes(monkeypatch, db_session, raw_db_session):
    """Proves the Critical fix from code review: one case's execution
    raising (simulating a real transient failure, e.g. an Ollama timeout)
    must NOT discard every other already-computed case's result, and the
    run as a whole must still reach `status="completed"` -- see
    `eval_worker.py`'s module docstring's "Per-case failure isolation"
    section. Forces `_execute_retrieval_case` to raise for exactly one
    retrieval-dimension case_id via monkeypatch (every other call passes
    through to the real function unchanged), then asserts: (1) a real
    `EvalCaseResult` row exists for every one of the 49 loaded cases,
    including the forced failure (recorded as `passed=False` with the
    real exception message in `details_json`, not silently dropped), and
    (2) at least one OTHER retrieval-dimension case still produced a real,
    passing result -- proving the failure was truly isolated, not a
    symptom of the whole retrieval dimension having been aborted.
    """
    _patch_small_case_sample(monkeypatch)
    run_seed_corpus_ingestion(db_session)
    db_session.flush()

    failing_case_id = "retrieval.public_doc.001"
    real_execute_retrieval_case = eval_worker_module._execute_retrieval_case

    def _flaky_execute_retrieval_case(session, case):
        if case.case_id == failing_case_id:
            raise RuntimeError("simulated transient failure (e.g. a real Ollama timeout)")
        return real_execute_retrieval_case(session, case)

    monkeypatch.setattr(eval_worker_module, "_execute_retrieval_case", _flaky_execute_retrieval_case)

    before = _side_effect_watermarks(raw_db_session)
    try:
        run = eval_worker_module.run_eval_suite(db_session)
        db_session.flush()

        # The RUN as a whole still completes -- the Critical fix this test exists to prove.
        assert run.status == "completed"
        assert run.error_message is None

        results = {
            r.case_id: r
            for r in db_session.scalars(select(EvalCaseResult).where(EvalCaseResult.eval_run_id == run.id)).all()
        }
        all_cases = eval_worker_module._load_all_cases("v2")
        # Every case produced a real result row -- none silently dropped
        # because a sibling case blew up.
        assert len(results) == len(all_cases)

        # The forced failure IS recorded, as a real failed case, not
        # silently swallowed into a false pass.
        assert failing_case_id in results
        failing_result = results[failing_case_id]
        assert failing_result.passed is False
        failing_details = json.loads(failing_result.details_json)
        assert "simulated transient failure" in failing_details.get("error", "")

        # At least one OTHER retrieval-dimension case still produced a
        # real, passing result -- proving isolation, not a dimension-wide abort.
        other_retrieval_case_ids = [
            c.case_id for c in all_cases if c.dimension == "retrieval" and c.case_id != failing_case_id
        ]
        assert any(results[cid].passed for cid in other_retrieval_case_ids)

        # Every non-retrieval dimension still has real coverage too.
        assert {r.dimension for r in results.values()} == _EXPECTED_DIMENSIONS
    finally:
        _cleanup_side_effects(raw_db_session, before)


def test_load_all_cases_filters_golden_by_dataset_version_but_always_includes_adversarial():
    """Fast, pure unit test (no DB/network) of `_load_all_cases`'s
    documented `dataset_version` convention (see `eval_worker.py`'s module
    docstring's "dataset_version convention" section): golden-file cases
    (chat/retrieval/tool/approval) ARE filtered by the passed-in
    `dataset_version`, while every adversarial case is included
    UNCONDITIONALLY, regardless of that argument.
    """
    cases_v2 = eval_worker_module._load_all_cases("v2")
    golden_dimensions = {"chat", "retrieval", "tool", "approval"}
    assert golden_dimensions.issubset({c.dimension for c in cases_v2})
    assert all(c.dataset_version == "v2" for c in cases_v2 if c.dimension in golden_dimensions)
    adversarial_cases_v2 = [c for c in cases_v2 if c.dimension == "adversarial"]
    assert len(adversarial_cases_v2) > 0

    # No golden file has ever used "v3" -- filtering by it must exclude
    # every golden-dimension case, while adversarial cases (whose own
    # dataset_version is "v1", independent of this argument) are still
    # included, identically to the "v2" call above.
    cases_v3 = eval_worker_module._load_all_cases("v3")
    assert not any(c.dimension in golden_dimensions for c in cases_v3)
    adversarial_cases_v3 = [c for c in cases_v3 if c.dimension == "adversarial"]
    assert {c.case_id for c in adversarial_cases_v3} == {c.case_id for c in adversarial_cases_v2}
    assert len(adversarial_cases_v3) == len(adversarial_cases_v2)


def test_ensure_target_employee_exists_bumps_sequence_so_later_autoincrement_inserts_never_collide(raw_db_session):
    """Regression test for a real bug found and fixed during Phase 10 Task
    9's fresh-state verification (see `docs/DECISION_LOG.md`'s 2026-09-11
    entry): `_ensure_target_employee_exists`'s explicit-id insert (for the
    fixed literal target-employee ids `eval/golden/v2/tools_approvals_v1
    .jsonl` references -- 42/101/205) must bump `employee_id_seq` past the
    id it just used, or a later, unrelated autoincrement `Employee` insert
    ANYWHERE ELSE in the shared test database can organically reach that
    exact id and fail with a real `UniqueViolation` -- reproduced directly
    (not hypothetically) during this task's own from-empty full-suite run,
    where the sequence reached exactly 101 and collided with this
    function's already-planted fixture row from `test_eval_worker.py`'s own
    earlier tests in the same session.

    This test picks an id one past the current real max (so it's
    guaranteed free), calls the function with it directly, then performs a
    normal autoincrement `Employee` insert (mirroring every other
    Employee-creating test in this suite, e.g. `test_retrieval.py`'s
    `_make_employee`) and asserts its id lands strictly ABOVE the explicit
    id just used -- the exact property the original, unfixed code violated.
    """
    max_id = raw_db_session.execute(text("SELECT COALESCE(MAX(id), 0) FROM employee")).scalar_one()
    target_id = max_id + 1

    eval_worker_module._ensure_target_employee_exists(target_id)

    probe_employee = Employee(
        display_name="Sequence Regression Probe",
        email=f"sequence.regression.probe.{target_id}@example.test",
        title="Employee",
        hire_date=datetime(2024, 1, 1),
        timezone="UTC",
    )
    raw_db_session.add(probe_employee)
    raw_db_session.commit()

    try:
        assert probe_employee.id > target_id
    finally:
        raw_db_session.execute(text("DELETE FROM employee WHERE id IN (:a, :b)"), {"a": target_id, "b": probe_employee.id})
        raw_db_session.commit()


def test_arq_worker_processes_run_eval_suite_task_via_real_redis(monkeypatch, raw_db_session):
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

    _patch_small_case_sample(monkeypatch)
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
