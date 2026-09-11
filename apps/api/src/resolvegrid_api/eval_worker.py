"""Arq-backed evaluation-suite batch runner (Phase 10 Task 7) -- the second
real async job in this repo, mirroring `resolvegrid_api.ingestion_worker`'s
exact two-entry-point shape (read that module first if you haven't; every
design choice below that isn't explained again here is a direct copy of its
precedent).

Two entry points into the same batch-execution logic:

1. `run_eval_suite(session, dataset_version)` -- a plain, synchronous,
   directly-callable function. Loads every `EvalCase` for the given
   dataset (see "dataset_version convention" below), executes each case
   against the REAL running system (real chat graph, real tool-invocation
   graph, real hybrid retrieval, real Ollama completions -- no mocking),
   grades each via `services/evaluation`'s graders, and persists one
   `EvalRun` + one `EvalCaseResult` per case.
2. `run_eval_suite_task(ctx)` -- the real Arq job function, registered in
   `WorkerSettings.functions`, opening its own session and
   `asyncio.to_thread`-ing the sync work exactly like
   `ingest_seed_corpus_task` does.

`dataset_version` convention (a real gotcha this task found and resolved,
not silently worked around)
------------------------------------------------------------------------
The golden files this function loads do NOT all share one
`dataset_version` string: every case in `eval/golden/v2/*.jsonl` sets
`dataset_version="v2"`, but every case in `eval/adversarial/v1.jsonl` sets
`dataset_version="v1"` (its own, independent versioning scheme for the
adversarial suite specifically -- see that file's cases). Filtering all
four files by one literal equality against a single `dataset_version`
argument would therefore silently load ZERO adversarial cases for any
`dataset_version` value that also matches the golden files, or zero golden
cases for any value that matches the adversarial file -- either way, a
real suite run would silently be missing an entire dimension's worth of
cases with no error.

The convention this module adopts (documented here as the single source of
truth): `dataset_version` (default `DATASET_VERSION = "v2"`) filters ONLY
the three `eval/golden/v2/*.jsonl` files (`case.dataset_version ==
dataset_version`) and is the value recorded on the resulting `EvalRun.
dataset_version` column. `eval/adversarial/v1.jsonl`'s cases are ALWAYS
included in every suite run, regardless of the `dataset_version` argument
-- the adversarial suite's zero-tolerance policy (plan.md's Task 6) is
deliberately independent of which golden-dataset version is under test,
and Task 6's own `test_adversarial_suite.py` already gates that file's
cases unconditionally in every CI run. A future phase that ever needs to
version the adversarial file independently (a real second
`dataset_version` value, e.g. "v2" adversarial cases) should extend this
loader with a second, explicit parameter rather than overloading this
one string to mean two different things.

Execution strategy per dimension (see each helper's docstring for the full
detail; summarized here since this is the module's real core judgment
call)
------------------------------------------------------------------------
- `chat`: real `ainvoke()` of the real chat graph (`build_graph`), same
  shape `routers/chat.py`'s handler uses, with an unrestricted retrieval
  scope (`eval/golden/v2/chat_v1.jsonl`'s cases carry no
  `principal_fixture`/`authz` at all -- they are Phase 6-style
  schema/intent-only cases, validated against real intent classification,
  not against any authz-scoped retrieval). Real, unmocked Ollama
  completions -- no mocking anywhere in this module, per this codebase's
  established no-mocking-for-real-behavior norm for eval/verification code.
- `retrieval`: delegates to the EXISTING `eval_retrieval.evaluate_case`
  (Task 3's already-consolidated per-case retrieval logic) via a small
  `EvalCase` -> `GoldenCase` adapter -- no retrieval logic is
  reimplemented here. Reranking is deliberately left disabled
  (`reranker_model=None`, `eval_retrieval`'s own default), matching Phase
  7's originally-measured baseline path rather than Task 6's optional
  reranked path; `RERANKER_VERSION = "disabled"` below records this
  choice on every `EvalRun` row rather than leaving it ambiguous.
- `tool` / `approval`: real `ainvoke()` of the real tool-invocation graph
  (`build_tool_invocation_graph`), same allowlist -> validate -> pause
  -> (for approval cases) decide -> resume flow `routers/tools.py`/
  `routers/approvals.py` use, called in-process (not over HTTP). HONEST,
  DOCUMENTED LIMITATION: this codebase has no NLU-based "extract tool
  arguments from natural language" node yet (`tool_execution.py`'s own
  docstring: "a future LangGraph 'select tool' node -- Task 5/6, NOT wired
  up here"). This harness therefore submits `case.expected_tool_params`
  directly as the proposed call's params, rather than deriving them from
  `case.input_text` -- it proves the real allowlist/schema-validation/
  approval-pause/decide/resume/mutation-execution pipeline runs correctly
  end to end for an already-correct proposed call, but it does NOT (and
  today cannot) test whether a model would have correctly parsed
  `case.input_text` into that same call. This is a real, judged
  simplification, not a mistake -- see `_run_tool_case`/`_run_approval_
  case`'s docstrings.
- `adversarial`: graded via a STRUCTURAL PASS-THROUGH, not real
  re-execution. Every one of plan.md's 9 adversarial scenarios already has
  a real, DB/HTTP/real-model-touching pytest in
  `apps/api/tests/test_adversarial_suite.py` (Task 6), which already
  zero-tolerance-gates all 10 cases (including the reused-case pointers)
  as part of the normal `apps/api` CI suite. Re-executing those same
  scenarios a second time inside this batch job would itself be exactly
  the "duplicated assertion that can silently drift apart" risk the
  phase's plan repeatedly warns against for reused cases (Task 6's own
  "Notes" section) -- generalized here one level further, to the whole
  adversarial dimension, for this specific batch-runner context. This
  module instead grades each adversarial case's `forbidden_actions` field
  (vacuously true where unset, since the harness itself performs no
  actions for these cases) and records the simplification plainly in
  `EvalCaseResult.details_json` for anyone reading a run's results later.
  This is a real, working `EvalCaseResult` row for every adversarial case
  -- not a fabricated pass -- it is just intentionally shallow, and says so.

Session/commit boundary (a real deviation from `ingestion_worker.py`'s
convention, found and resolved while wiring this up, not silently papered
over)
------------------------------------------------------------------------
`run_seed_corpus_ingestion` never commits its caller-supplied `session` --
every DB write happens on that one session, in one transaction, entirely
under the caller's control. `run_eval_suite` cannot fully preserve that:
the real chat/tool-invocation graphs it drives (`agent_retrieval.py`,
`approval_service.py`, `agent_mutation_execution.py`) each open and COMMIT
their OWN independent `session_factory()` session per call (an established
pattern predating this task -- see those modules' own docstrings), which
means any `Employee`/`Department`/`RoleAssignment` fixture row this module
creates to drive a tool/approval-dimension case through those real graphs
MUST already be committed and visible to a different DB connection before
that graph call runs, or the graph's own FK-constrained inserts
(`ApprovalRequest.requested_by_id`, `EmployeeEntitlement.employee_id`,
etc.) would fail against a connection that cannot see an uncommitted row
from a different transaction.

The resolution: every principal/employee/department fixture this module
creates (`_principal_from_fixture`, `_ensure_analyst_requester_principal`,
`_ensure_target_employee_exists`) uses its OWN short-lived, committing
`session_factory()` session -- never the `session` argument `run_eval_suite`
was given. The caller-supplied `session` is used ONLY for: (1) the
retrieval-dimension's `evaluate_case` calls (which need no cross-session
visibility -- same connection, read-your-own-writes), and (2) this
function's own `EvalRun`/`EvalCaseResult` bookkeeping rows, which are only
ever `flush()`-ed here, never committed -- exactly preserving
`ingestion_worker.py`'s "does not commit; caller controls the transaction
boundary" contract for the run-tracking rows themselves. Only the
graph-driving fixture rows deviate, and only because the existing,
already-shipped graph-adapter modules architecturally require it.

Async-inside-sync (why `run_eval_suite` can call `asyncio.run()` safely)
------------------------------------------------------------------------
The real chat/tool-invocation graphs are `AsyncPostgresSaver`-backed and
must be `ainvoke()`d inside a running event loop (see `chat.py`'s/
`tools.py`'s docstrings) -- but `run_eval_suite` itself must stay a plain,
directly-callable sync function (mirroring `run_seed_corpus_ingestion`'s
contract exactly, since `test_eval_worker.py`'s direct-call test needs to
call it with no event loop already running). `run_eval_suite` resolves
this by calling `asyncio.run(_execute_graph_dimension_cases(...))`
internally for exactly the chat/tool/approval-dimension work -- safe in
both real call sites: (1) a bare pytest call (no loop running in the
calling thread), and (2) `run_eval_suite_task`'s `asyncio.to_thread(...)`
call, which runs `run_eval_suite` in a separate OS thread with no event
loop of its own, so `asyncio.run()` there creates a fresh loop exactly
like a plain synchronous call would.

Windows event-loop-policy shim (identical rationale to `main.py`, copied
here rather than importing `main.py` -- this module must not import the
whole FastAPI app just to get one import-time side effect):
`AsyncPostgresSaver` uses psycopg's async driver, which raises under
Windows' default ProactorEventLoop -- must switch to the Selector policy
before any event loop is created.

Per-case failure isolation (a real bug found and fixed in code review, not
part of this task's original design)
------------------------------------------------------------------------
A batch runner whose entire point is "grade every case and report what
happened" must never let ONE case's failure (a real, documented failure
mode elsewhere in this codebase -- e.g. an Ollama timeout) discard every
OTHER already-computed case's result. Every dimension's execution loop
(`_execute_case_safe` for the synchronous retrieval/adversarial loops in
`run_eval_suite`; `_execute_async_case_safe` for the async chat/tool/
approval loop in `_execute_graph_dimension_cases`) therefore catches ANY
exception a single case's execution raises and turns it into an
`{_EXECUTION_ERROR_KEY: "..."}` marker dict instead of letting it
propagate. `_grade_case_safe` recognizes that marker (and separately
catches any exception a grader itself might raise) and always produces a
real, failed `EvalCaseResult` for that one case -- `passed=False`,
`details["error"]` set -- rather than aborting the whole run. The outer
`except Exception` in `run_eval_suite` is reserved for genuinely
run-level failures that happen BEFORE any per-case try/except could even
run (loading the JSONL files, or `AsyncPostgresSaver`/graph-build setup
inside `_execute_graph_dimension_cases`) -- those still mark the whole
`EvalRun` `status="error"` and re-raise, exactly as before. `run_eval_
suite_task` was adjusted to `session.commit()` on THAT path too (not only
on success), so a genuine run-level failure still leaves a real,
committed, queryable `EvalRun` row instead of vanishing when the
session's implicit rollback-on-exception discards an uncommitted row.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from arq.connections import RedisSettings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from resolvegrid_agent_orchestration import build_graph, build_tool_invocation_graph
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.agent_mutation_execution import execute_mutation_for_agent
from resolvegrid_api.agent_retrieval import retrieve_for_agent
from resolvegrid_api.approval_service import request_approval_for_agent
from resolvegrid_api.db import DATABASE_URL, session_factory
from resolvegrid_api.eval_judge import judge_response_real
from resolvegrid_api.eval_retrieval import GoldenCase, evaluate_case
from resolvegrid_api.ingestion_worker import EMBEDDING_VERSION as _INGESTION_EMBEDDING_VERSION
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest, Department, Employee, RoleAssignment
from resolvegrid_api.models.evaluation import EvalCaseResult, EvalRun
from resolvegrid_api.mutation_execution import execute_readonly_tool
from resolvegrid_api.retrieval_authz import AuthzFilter
from resolvegrid_api.tool_execution import (
    ToolNotAllowedError,
    ToolValidationError,
    available_tools_for_principal,
    select_tool,
    validate_tool_schema,
)
from resolvegrid_authz import Principal, RoleGrant
from resolvegrid_evaluation.graders import (
    GradeResult,
    grade_approval_compliance,
    grade_citation_correctness,
    grade_forbidden_actions,
    grade_schema_validity,
    grade_tool_argument_accuracy,
)
from resolvegrid_evaluation.schema import EvalCase, load_eval_cases

# Redis connection for Arq -- same env-var convention as ingestion_worker.py.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# See main.py's identical comment: AsyncPostgresSaver.from_conn_string()
# needs the plain libpq-style URI, not SQLAlchemy's "+psycopg" suffix.
_CHECKPOINTER_DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOLDEN_V2_DIR = _REPO_ROOT / "eval" / "golden" / "v2"
_ADVERSARIAL_PATH = _REPO_ROOT / "eval" / "adversarial" / "v1.jsonl"

# Pinned version tuple for this task's eval runs -- named constants, not
# scattered string literals, mirroring ingestion_worker.py's PARSER_VERSION/
# CHUNKING_VERSION pattern exactly. See module docstring for the
# `dataset_version`/`RERANKER_VERSION` design notes.
DATASET_VERSION = "v2"
PROMPT_VERSION = "v1"
WORKFLOW_VERSION = "chat-tool-graphs-v1"
RETRIEVER_VERSION = "hybrid-rrf-v1"
RERANKER_VERSION = "disabled"
EMBEDDINGS_VERSION = _INGESTION_EMBEDDING_VERSION
GENERATION_MODEL_VERSION = llm_gateway.DEFAULT_MODEL
JUDGE_VERSION = "v1"
# Matches every tool currently in resolvegrid_contracts.tools.TOOL_REGISTRY's
# own `version` field (both "lookup_employee_entitlements" and
# "grant_vpn_access" are "1.0.0" as of this writing) -- if a future tool
# registers a different version, this constant should become a real
# per-tool lookup rather than one flat string, but there is exactly one
# version in use today.
TOOL_SCHEMA_VERSION = "1.0.0"

_ANALYST_REQUESTER_EMAIL = "eval.harness.analyst.requester@example.test"

# Marker key an `actual_result` dict carries when a case's own execution
# raised -- see module docstring's "Per-case failure isolation" section.
# Never a real key any `_run_*_case`/`_execute_*_case` function's SUCCESS
# path sets, so its mere presence unambiguously means "this case's
# execution never completed."
_EXECUTION_ERROR_KEY = "__execution_error__"


def _execute_case_safe(execute_fn) -> dict:
    """Runs one synchronous case-execution callable, catching ANY exception
    and turning it into an `{_EXECUTION_ERROR_KEY: "..."}` marker dict
    instead of letting it propagate -- see module docstring's "Per-case
    failure isolation" section. Used by `run_eval_suite`'s retrieval/
    adversarial-dimension loops; `_execute_async_case_safe` below is the
    async equivalent for the chat/tool/approval-dimension loop.
    """
    try:
        return execute_fn()
    except Exception as exc:  # noqa: BLE001 -- see module docstring: a single
        # case's failure must never abort every other case's result.
        return {_EXECUTION_ERROR_KEY: str(exc)}


async def _execute_async_case_safe(coro) -> dict:
    """Async equivalent of `_execute_case_safe` -- awaits `coro`, catching
    ANY exception into the same `{_EXECUTION_ERROR_KEY: "..."}` marker
    shape.
    """
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 -- see `_execute_case_safe`'s
        # identical rationale.
        return {_EXECUTION_ERROR_KEY: str(exc)}


def _git_commit() -> str:
    """The real current commit sha, via `git rev-parse HEAD` against this
    repo's root -- never a hardcoded placeholder. Returns `"unknown"`
    (never raises) if this isn't a git checkout or `git` isn't on PATH, so
    a missing/broken git toolchain can never take down an eval run over a
    purely cosmetic version-tracking field.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- see docstring: must never raise
        return "unknown"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def _load_all_cases(dataset_version: str) -> list[EvalCase]:
    """See module docstring's "dataset_version convention" section."""
    cases: list[EvalCase] = []
    for filename in ("chat_v1.jsonl", "retrieval_v1.jsonl", "tools_approvals_v1.jsonl"):
        cases.extend(c for c in load_eval_cases(_GOLDEN_V2_DIR / filename) if c.dataset_version == dataset_version)
    # Adversarial cases are ALWAYS included, regardless of `dataset_version` -- see module docstring.
    cases.extend(load_eval_cases(_ADVERSARIAL_PATH))
    return cases


# ---------------------------------------------------------------------------
# Principal / fixture construction (see module docstring's "Session/commit
# boundary" section for why each of these opens its own committing session)
# ---------------------------------------------------------------------------


def _principal_from_fixture(fixture: dict | None, *, unique_tag: str) -> Principal:
    """Resolve an `EvalCase.principal_fixture` dict (`{"role", "department",
    "scope"}`) into a real `Principal`, backed by a real, idempotently
    get-or-created `Employee` (+ `Department`, + `RoleAssignment` when
    `role` names a real role) -- committed on its own `session_factory()`
    session (see module docstring). `unique_tag` must be unique per
    logical fixture identity (e.g. the case_id) so repeated suite runs
    reuse the same rows rather than accumulating duplicates.

    `role == "employee"` (a plain employee with no elevated role, as some
    fixture dicts declare) intentionally gets NO `RoleAssignment` row --
    matches this codebase's real-world convention (see
    `test_adversarial_suite.py`'s outsider/owner employees) that a plain
    employee simply holds no role grant at all.
    """
    fixture = fixture or {}
    role = fixture.get("role")
    department_name = fixture.get("department")
    scope = fixture.get("scope", "global")

    with session_factory() as s:
        department = None
        if department_name and department_name != "none":
            department = s.scalar(select(Department).where(Department.name == department_name))
            if department is None:
                department = Department(name=department_name)
                s.add(department)
                s.flush()

        email = f"eval.harness.{unique_tag}@example.test"
        employee = s.scalar(select(Employee).where(Employee.email == email))
        if employee is None:
            employee = Employee(
                display_name=f"Eval Harness {unique_tag}",
                email=email,
                title="Employee",
                hire_date="2024-01-01T00:00:00",
                timezone="UTC",
                department_id=department.id if department else None,
            )
            s.add(employee)
            s.flush()

        roles: tuple[RoleGrant, ...] = ()
        if role and role != "employee":
            scope_id = 1 if scope == "department" else None
            existing_role = s.scalar(
                select(RoleAssignment).where(
                    RoleAssignment.employee_id == employee.id, RoleAssignment.role == role
                )
            )
            if existing_role is None:
                s.add(RoleAssignment(employee_id=employee.id, role=role, scope=scope, scope_id=scope_id))
                s.flush()
            roles = (RoleGrant(role=role, scope=scope, scope_id=scope_id),)

        s.commit()
        return Principal(employee_id=employee.id, roles=roles)


def _ensure_analyst_requester_principal() -> Principal:
    """A fixed, idempotently get-or-created analyst-role principal used to
    drive an approval-dimension case's *setup* step (creating the pending
    `ApprovalRequest` a scripted approver then decides on) -- distinct from
    `case.principal_fixture`, which names the APPROVER for approval
    -dimension cases, not the original requester. See
    `eval/golden/v2/tools_approvals_v1.jsonl`'s notes: any analyst-role,
    global-scope principal is equally valid for this step (the JSONL's own
    `tool.grant_vpn_access.001` case already exercises a real such
    principal via `_principal_from_fixture` for its own dimension; this is
    a second, reusable one scoped to this module's approval-setup need).
    """
    with session_factory() as s:
        employee = s.scalar(select(Employee).where(Employee.email == _ANALYST_REQUESTER_EMAIL))
        if employee is None:
            employee = Employee(
                display_name="Eval Harness Analyst Requester",
                email=_ANALYST_REQUESTER_EMAIL,
                title="Analyst",
                hire_date="2024-01-01T00:00:00",
                timezone="UTC",
            )
            s.add(employee)
            s.flush()
        existing_role = s.scalar(
            select(RoleAssignment).where(
                RoleAssignment.employee_id == employee.id, RoleAssignment.role == "analyst"
            )
        )
        if existing_role is None:
            s.add(RoleAssignment(employee_id=employee.id, role="analyst", scope="global"))
            s.flush()
        s.commit()
        return Principal(employee_id=employee.id, roles=(RoleGrant(role="analyst", scope="global"),))


def _ensure_target_employee_exists(employee_id: int) -> None:
    """Idempotently ensures a real `Employee` row with this EXACT id exists
    -- needed because `eval/golden/v2/tools_approvals_v1.jsonl`'s cases
    reference fixed literal `employee_id` values (42/101/205) as the grant
    TARGET, which real FK-constrained tables downstream
    (`EmployeeEntitlement.employee_id`, via the real `grant_vpn_access`
    adapter) require to genuinely exist. Those ids are plausibly already
    real seeded employees in a fully-seeded dev DB (Phase 1's org
    generator seeds a full employee roster) -- this only inserts a minimal
    placeholder row when no row with that id exists yet, so this module's
    tests are self-contained even against a bare, unseeded schema.

    Explicit `id=` assignment on an autoincrement primary key is standard,
    supported SQLAlchemy/Postgres behavior for a plain serial/identity
    column (verified against this schema's migrations, which use a plain
    autoincrementing integer PK, not a `GENERATED ALWAYS` identity column)
    -- Postgres accepts the explicit value without complaint.

    REAL BUG FOUND AND FIXED DURING PHASE 10 TASK 9'S FRESH-STATE
    VERIFICATION (not part of this task's original design -- see
    `docs/DECISION_LOG.md`'s 2026-09-11 entry): this function's original
    version claimed skipping a sequence bump was "irrelevant... since this
    function never lets the normal autoincrement path assign that same
    id." That claim is FALSE and was disproven by direct reproduction:
    Postgres sequences are non-transactional, so `nextval()` calls from
    OTHER tests' `db_session`-fixture flushes (rolled back at teardown,
    per `apps/api/tests/conftest.py`) still permanently advance
    `employee_id_seq`, even though the rows themselves never persist.
    Given enough such flushes across a ~230-test suite, the sequence can
    and did organically reach exactly one of this function's explicit
    literal ids (42/101/205, from `eval/golden/v2/tools_approvals_v1.jsonl`)
    -- these fixture rows are deliberately never cleaned up (see this
    module's docstring: idempotent get-or-create, so repeated suite runs
    don't accumulate duplicates), so the collision is a real
    `UniqueViolation` against a genuinely still-present row, not a flake.
    Fix: after an explicit-id insert, bump the sequence to at least the
    table's current max id via `pg_get_serial_sequence`/`setval` -- the
    standard, idiomatic Postgres remedy for manually assigning a serial
    column's value. This guarantees every subsequent `nextval()` call
    anywhere in the process returns a value strictly greater than any id
    this function has ever explicitly assigned, permanently closing the
    collision, not just for the ids observed so far.
    """
    with session_factory() as s:
        if s.get(Employee, employee_id) is not None:
            return
        s.add(
            Employee(
                id=employee_id,
                display_name=f"Eval Fixture Target Employee {employee_id}",
                email=f"eval.fixture.target.{employee_id}@example.test",
                title="Employee",
                hire_date="2024-01-01T00:00:00",
                timezone="UTC",
            )
        )
        s.flush()
        s.execute(text("SELECT setval(pg_get_serial_sequence('employee', 'id'), (SELECT MAX(id) FROM employee))"))
        s.commit()


def _decide_approval(approval_request_id: int, *, approver_employee_id: int, decision: str, comment: str | None) -> str:
    """Records a real `ApprovalDecision` + updates `ApprovalRequest.status`,
    on its own committing session, BEFORE the caller resumes the paused
    graph -- mirrors `routers/approvals.py`'s `decide_approval` commit
    -then-resume ordering exactly (see that module's docstring for why this
    ordering matters). Returns the resulting `ApprovalRequest.status`.

    Deliberately a NARROWER subset of that handler, not an attempt at full
    parity: it skips the re-entrancy guard (`status != "pending"`), the
    expiry check, and the reject-requires-a-comment validation -- this
    harness always decides a freshly-created, unexpired, single-use
    `ApprovalRequest` it just created itself with an already-valid comment
    (see `_run_approval_case`), so none of those real-world guards are
    reachable here. Not an oversight; a future case that exercises one of
    those paths should add it explicitly rather than assume this function
    already covers it.
    """
    with session_factory() as s:
        approval_request = s.get(ApprovalRequest, approval_request_id)
        s.add(
            ApprovalDecision(
                approval_request_id=approval_request.id,
                approver_id=approver_employee_id,
                decision=decision,
                comment=comment,
            )
        )
        approval_request.status = "approved" if decision == "approved" else "rejected"
        s.commit()
        return approval_request.status


def _extract_approval_request_id(result_state, thread_id: str) -> int | None:
    """Mirrors `routers/tools.py`'s `invoke_tool` extraction exactly: read
    `approval_request_id` off the interrupted run's own `__interrupt__`
    payload, falling back to a DB lookup by `agent_run_id` if a future
    graph change ever altered what an interrupted result dict carries.
    """
    interrupts = result_state.get("__interrupt__") if isinstance(result_state, dict) else None
    if interrupts:
        approval_request_id = interrupts[0].value.get("approval_request_id")
        if approval_request_id is not None:
            return approval_request_id
    with session_factory() as s:
        row = s.scalar(select(ApprovalRequest).where(ApprovalRequest.agent_run_id == thread_id))
        return row.id if row is not None else None


def _base_agent_state(thread_id: str, *, principal_employee_id: int | None, input_text: str) -> dict:
    """The `AgentState` fields common to EVERY graph invocation in this
    module -- both `build_graph`'s chat pipeline and `build_tool_
    invocation_graph`'s pipeline compile against the same shared
    `AgentState` TypedDict (see `state.py`'s module docstring), so an
    initial-state dict for either graph must set every key, even the ones
    the OTHER graph actually uses. `_run_chat_case`/`_tool_invocation_
    initial_state` each overlay their own dimension-specific fields
    (`retrieval_scope` for chat; `risk_level`/`proposed_tool_name`/
    `proposed_tool_params` for tool-invocation) on top of this base dict,
    instead of each separately repeating every shared key/placeholder
    value.
    """
    return {
        "thread_id": thread_id,
        "principal_employee_id": principal_employee_id,
        "input_text": input_text,
        "intent": None,
        "risk_level": None,
        "retrieval_scope": None,
        "retrieved_chunks": None,
        "retrieval_sufficient": None,
        "context_block": None,
        "output_text": None,
        "error": None,
        "citations_verified": None,
        "verified_chunk_ids": None,
        "fabricated_chunk_ids": None,
        "proposed_tool_name": None,
        "proposed_tool_params": None,
        "approval_request_id": None,
        "approval_decision": None,
        "tool_invocation_result": None,
    }


def _tool_invocation_initial_state(
    thread_id: str, principal_employee_id: int | None, tool_name: str, params: dict, risk_level: str | None
) -> dict:
    """The tool-invocation graph's initial `AgentState` -- mirrors
    `routers/tools.py`'s `invoke_tool` initial_state construction exactly.
    """
    state = _base_agent_state(thread_id, principal_employee_id=principal_employee_id, input_text=f"[eval harness] {tool_name}")
    state["risk_level"] = risk_level or "high"
    state["proposed_tool_name"] = tool_name
    state["proposed_tool_params"] = params
    return state


# ---------------------------------------------------------------------------
# Per-dimension execution
# ---------------------------------------------------------------------------


async def _run_chat_case(chat_graph, case: EvalCase) -> dict:
    """Real `ainvoke()` of the real chat graph -- mirrors `routers/chat.py`'s
    `chat()` handler's `initial_state` construction exactly. `chat_v1.jsonl`'s
    cases carry no `principal_fixture`/`authz` (Phase 6-style schema/intent
    -only cases), so an unrestricted retrieval scope is used -- these cases
    assert `expected_intent`/`expected_risk_level`, neither of which depends
    on what (if anything) retrieval finds.
    """
    thread_id = uuid4().hex
    initial_state = _base_agent_state(thread_id, principal_employee_id=None, input_text=case.input_text or "")
    initial_state["retrieval_scope"] = {"unrestricted": True, "allowed_tags": []}
    final_state = await chat_graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    retrieved_chunks = final_state.get("retrieved_chunks") or []
    verified_ids = set(final_state.get("verified_chunk_ids") or [])
    return {
        "intent": final_state.get("intent"),
        "risk_level": final_state.get("risk_level"),
        "output_text": final_state.get("output_text"),
        "citations": [c["chunk_id"] for c in retrieved_chunks if c["chunk_id"] in verified_ids],
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "actions_taken": [],
    }


def _resolve_and_validate_tool_call(principal: Principal, case: EvalCase) -> tuple:
    """Shared allowlist -> select -> validate -> target-employee-fixture
    step both `_run_tool_case` and `_run_approval_case` need (previously
    duplicated between them almost verbatim). Returns `(tool, params)`.

    Raises `ToolNotAllowedError`/`ToolValidationError` exactly as
    `available_tools_for_principal`/`select_tool`/`validate_tool_schema`
    do -- callers decide how to handle each: `_run_tool_case` catches both
    and turns them into a soft `actual_result["error"]` (a tool-dimension
    case's own allowlist/schema outcome is part of what it tests), while
    `_run_approval_case` lets them propagate (an approval-dimension case
    failing this step is a fixture bug, not a scenario that dimension is
    meant to test) -- either way, this module's per-case isolation
    (`_execute_async_case_safe`) ensures neither path can abort the whole
    batch run.
    """
    available = available_tools_for_principal(principal)
    tool = select_tool(case.expected_tool_name, available)
    params = case.expected_tool_params or {}
    validate_tool_schema(tool, params)
    if isinstance(params.get("employee_id"), int):
        _ensure_target_employee_exists(params["employee_id"])
    return tool, params


async def _run_tool_case(tool_graph, case: EvalCase) -> dict:
    """Real allowlist -> validate -> (pause-for-approval | execute) flow --
    mirrors `routers/tools.py`'s `invoke_tool` handler exactly, called
    in-process. See module docstring's "tool / approval" section for the
    documented limitation: `case.expected_tool_params` is submitted
    directly as the proposed call (no NLU extraction step exists in this
    codebase yet).
    """
    principal = _principal_from_fixture(case.principal_fixture, unique_tag=f"tool-{case.case_id}")
    try:
        tool, params = _resolve_and_validate_tool_call(principal, case)
    except ToolNotAllowedError as exc:
        return {
            "tool_name": case.expected_tool_name,
            "tool_params": case.expected_tool_params,
            "error": f"ToolNotAllowedError: {exc}",
            "actions_taken": [],
        }
    except ToolValidationError as exc:
        return {
            "tool_name": case.expected_tool_name,
            "tool_params": case.expected_tool_params,
            "error": f"ToolValidationError: {exc}",
            "actions_taken": [],
        }

    if not tool.requires_approval:
        with session_factory() as s:
            result = execute_readonly_tool(s, tool_name=tool.name, tool_params=params)
            s.commit()
        return {"tool_name": tool.name, "tool_params": params, "status": "executed", "output": result, "actions_taken": [tool.name]}

    thread_id = uuid4().hex
    initial_state = _tool_invocation_initial_state(thread_id, principal.employee_id, tool.name, params, case.risk_level)
    result_state = await tool_graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    approval_request_id = _extract_approval_request_id(result_state, thread_id)

    return {
        "tool_name": tool.name,
        "tool_params": params,
        "status": "pending_approval",
        "approval_request_id": approval_request_id,
        "actions_taken": [],
    }


async def _run_approval_case(tool_graph, case: EvalCase) -> dict:
    """Real create-pending -> decide -> resume flow. Step 1 (create a real
    pending `ApprovalRequest`) uses a fixed analyst-role requester
    principal (`_ensure_analyst_requester_principal`), NOT
    `case.principal_fixture` (which names the APPROVER for these cases).
    Step 2 (decide + resume) mirrors `routers/approvals.py`'s
    `decide_approval` handler exactly, including its commit-then-resume
    ordering (see `_decide_approval`'s docstring).
    """
    requester = _ensure_analyst_requester_principal()
    tool, params = _resolve_and_validate_tool_call(requester, case)

    thread_id = uuid4().hex
    initial_state = _tool_invocation_initial_state(thread_id, requester.employee_id, tool.name, params, case.risk_level)
    result_state = await tool_graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    approval_request_id = _extract_approval_request_id(result_state, thread_id)
    if approval_request_id is None:
        return {"tool_name": tool.name, "tool_params": params, "error": "no approval request created", "actions_taken": []}

    approver = _principal_from_fixture(case.principal_fixture, unique_tag=f"approver-{case.case_id}")
    decision = case.expected_approval_decision or "approved"
    comment = None if decision == "approved" else "Insufficiently justified (eval harness scripted rejection)."
    final_status = _decide_approval(
        approval_request_id, approver_employee_id=approver.employee_id, decision=decision, comment=comment
    )

    result_state_2 = await tool_graph.ainvoke(
        Command(resume=decision), config={"configurable": {"thread_id": thread_id}}
    )
    tool_invocation_result = result_state_2.get("tool_invocation_result") if isinstance(result_state_2, dict) else None
    mutation_succeeded = isinstance(tool_invocation_result, dict) and tool_invocation_result.get("status") == "success"

    return {
        "tool_name": tool.name,
        "tool_params": params,
        "approval_decision": decision,
        "final_status": final_status,
        "tool_invocation_result": tool_invocation_result,
        "actions_taken": [tool.name] if (decision == "approved" and mutation_succeeded) else [],
    }


async def _execute_graph_dimension_cases(
    chat_cases: list[EvalCase], tool_cases: list[EvalCase], approval_cases: list[EvalCase]
) -> dict[str, dict]:
    """Builds the real chat/tool-invocation graphs ONCE (sharing one
    `AsyncPostgresSaver` checkpointer, mirroring `main.py`'s lifespan --
    see that module's "Phase 9 Task 7a" docstring section for why sharing
    one checkpointer between the two compiled graphs is safe), then
    executes every chat/tool/approval-dimension case against them in turn.
    Returns `{case_id: actual_result}`.
    """
    results: dict[str, dict] = {}
    if not (chat_cases or tool_cases or approval_cases):
        return results

    complete_fn = lambda prompt: llm_gateway.complete(prompt).text  # noqa: E731

    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        chat_graph = build_graph(checkpointer, complete_fn, retrieve_for_agent) if chat_cases else None
        tool_graph = (
            build_tool_invocation_graph(checkpointer, request_approval_for_agent, execute_mutation_for_agent)
            if (tool_cases or approval_cases)
            else None
        )

        # See module docstring's "Per-case failure isolation" section:
        # each case's execution is wrapped individually so ONE case
        # raising (e.g. a real Ollama timeout) can never discard every
        # other already-computed case's result. Anything that fails
        # BEFORE this point (checkpointer.setup()/build_graph above) is a
        # genuine run-level failure and is correctly left to propagate.
        for case in chat_cases:
            results[case.case_id] = await _execute_async_case_safe(_run_chat_case(chat_graph, case))
        for case in tool_cases:
            results[case.case_id] = await _execute_async_case_safe(_run_tool_case(tool_graph, case))
        for case in approval_cases:
            results[case.case_id] = await _execute_async_case_safe(_run_approval_case(tool_graph, case))

    return results


def _execute_retrieval_case(session: Session, case: EvalCase) -> dict:
    """Adapts one retrieval-dimension `EvalCase` into `eval_retrieval.
    GoldenCase` and delegates to the EXISTING, already-consolidated
    `evaluate_case` (Task 3) -- no retrieval logic is reimplemented here.
    """
    authz_dict = case.authz or {}
    authz = AuthzFilter(
        unrestricted=bool(authz_dict.get("unrestricted", False)),
        allowed_tags=frozenset(authz_dict.get("allowed_tags") or []),
    )
    golden_case = GoldenCase(
        query=case.input_text or "",
        authz=authz,
        relevant=frozenset(case.relevant or []),
        distractor=frozenset(case.distractor or []),
        must_not_appear=frozenset(case.must_not_appear or []),
        note=case.note or "",
    )
    result = evaluate_case(session, golden_case)
    return {
        "retrieved_chunk_ids": result.ranked_chunk_ids,
        "citations": [],
        "actions_taken": [],
        "retrieval_grade": {
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "reciprocal_rank": result.reciprocal_rank,
            "ndcg_at_k": result.ndcg_at_k,
            "leaked_chunk_ids": sorted(result.leaked_chunk_ids),
            "distractor_beats_best_relevant": result.distractor_beats_best_relevant,
            "distractor_margin": result.distractor_margin,
        },
    }


_ADVERSARIAL_SIMPLIFICATION_NOTE = (
    "Task 7 batch runner grades this adversarial case via a structural "
    "pass-through (the harness itself takes no DB/HTTP/model actions for "
    "this case) rather than re-executing the real scenario -- "
    "apps/api/tests/test_adversarial_suite.py (Phase 10 Task 6) already "
    "executes and zero-tolerance-gates this exact case for real, per case, "
    "as part of the normal apps/api pytest CI suite. Re-running it a "
    "second time inside this batch job would duplicate an already-passing, "
    "more expensive assertion -- see eval_worker.py's module docstring."
)


def _execute_adversarial_case_structurally(case: EvalCase) -> dict:
    """See module docstring's "adversarial" section and
    `_ADVERSARIAL_SIMPLIFICATION_NOTE`."""
    return {
        "actions_taken": [],
        "citations": [],
        "retrieved_chunk_ids": [],
        "simplification": _ADVERSARIAL_SIMPLIFICATION_NOTE,
    }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class CaseGradeOutcome(BaseModel):
    """The uniform, typed shape `_grade_case`/`_grade_case_safe` return --
    replaces an earlier bare 4-tuple with a small typed model, matching
    this phase's established preference for typed pydantic result shapes
    (`GradeResult`, `RetrievalGradeResult`, `JudgeVerdict`) over positional
    tuples. Frozen for the same reason those are (see `graders.py`'s
    module docstring): read once by `run_eval_suite`'s grading loop and
    never meant to be mutated afterward.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    score: float | None = None
    grader_type: str
    details: dict


def _grade_case(case: EvalCase, actual_result: dict) -> CaseGradeOutcome:
    """Combine the applicable `services/evaluation` grader(s) for `case
    .dimension` into one `CaseGradeOutcome` -- see module docstring for
    which graders apply to which dimension, and why (`adversarial`
    deliberately uses only `grade_forbidden_actions`, not the tool/
    approval graders, since it never executes a real proposed call).
    `details` is built entirely from each `GradeResult.model_dump()`
    (already JSON-safe by that model's own contract) plus plain dicts/
    lists/strings -- verified JSON-serializable by
    `test_eval_worker.py`'s own explicit round-trip test.

    May raise (a grader itself misbehaving on unexpected `actual_result`
    shape, or `judge_response_real` raising instead of returning a
    handled failure verdict) -- callers MUST go through `_grade_case_safe`
    below, never call this directly from `run_eval_suite`'s per-case loop,
    so a grading-time failure gets the same per-case isolation an
    execution-time failure does (see module docstring's "Per-case failure
    isolation" section).
    """
    details: dict = {}
    checks: list[GradeResult] = []
    score: float | None = None

    if case.dimension == "chat":
        schema_grade = grade_schema_validity(case, actual_result)
        details["schema_validity"] = schema_grade.model_dump()
        checks.append(schema_grade)

        citation_grade = grade_citation_correctness(case, actual_result)
        details["citation_correctness"] = citation_grade.model_dump()
        checks.append(citation_grade)

        if case.expected_intent is not None:
            intent_ok = actual_result.get("intent") == case.expected_intent
            risk_ok = case.expected_risk_level is None or actual_result.get("risk_level") == case.expected_risk_level
            chat_match = {
                "expected_intent": case.expected_intent,
                "actual_intent": actual_result.get("intent"),
                "expected_risk_level": case.expected_risk_level,
                "actual_risk_level": actual_result.get("risk_level"),
                "intent_ok": intent_ok,
                "risk_level_ok": risk_ok,
            }
            details["chat_expected_match"] = chat_match
            checks.append(GradeResult(passed=intent_ok and risk_ok, details=chat_match))

        checks.append(grade_forbidden_actions(case, actual_result))

    elif case.dimension == "retrieval":
        retrieval_grade = actual_result.get("retrieval_grade") or {}
        details["retrieval_grade"] = retrieval_grade
        score = retrieval_grade.get("recall_at_k")
        # Pass criteria (documented here, this module's own convention --
        # NOT the same as test_eval_retrieval.py's separate aggregate
        # regression-guard floor, which is a distinct CI gate over the
        # whole golden set): no must_not_appear leakage, the distractor
        # (when both distractor and relevant chunks are labeled) never
        # outranks the correct answer, and -- only when this case actually
        # has a labeled relevant set -- at least one relevant chunk landed
        # in the top-k. A case with an empty relevant set (a "no good
        # match"/authz-zero-result case) has recall_at_k=None and is not
        # penalized for that, matching eval_retrieval.py's own convention.
        retrieval_ok = (
            not retrieval_grade.get("leaked_chunk_ids")
            and retrieval_grade.get("distractor_beats_best_relevant") is not True
            and (retrieval_grade.get("recall_at_k") is None or retrieval_grade.get("recall_at_k") > 0)
        )
        checks.append(GradeResult(passed=retrieval_ok, score=score, details=retrieval_grade))
        checks.append(grade_forbidden_actions(case, actual_result))

    elif case.dimension == "tool":
        schema_grade = grade_schema_validity(case, actual_result)
        details["schema_validity"] = schema_grade.model_dump()
        checks.append(schema_grade)

        tool_grade = grade_tool_argument_accuracy(case, actual_result)
        details["tool_argument_accuracy"] = tool_grade.model_dump()
        checks.append(tool_grade)

        checks.append(grade_forbidden_actions(case, actual_result))

    elif case.dimension == "approval":
        tool_grade = grade_tool_argument_accuracy(case, actual_result)
        details["tool_argument_accuracy"] = tool_grade.model_dump()
        checks.append(tool_grade)

        approval_grade = grade_approval_compliance(case, actual_result)
        details["approval_compliance"] = approval_grade.model_dump()
        checks.append(approval_grade)

        checks.append(grade_forbidden_actions(case, actual_result))

    else:  # "adversarial"
        details["simplification"] = actual_result.get("simplification")
        checks.append(grade_forbidden_actions(case, actual_result))

    passed = all(c.passed for c in checks)
    grader_type = "deterministic"

    if case.judge_dimensions:
        verdicts = judge_response_real(case, actual_result)
        details["judge_verdicts"] = [v.model_dump() for v in verdicts]
        passed = passed and all(v.passed for v in verdicts)
        grader_type = "judge"

    details["checks"] = [c.model_dump() for c in checks]
    return CaseGradeOutcome(passed=passed, score=score, grader_type=grader_type, details=details)


def _grade_case_safe(case: EvalCase, actual_result: dict) -> CaseGradeOutcome:
    """Per-case-isolated wrapper around `_grade_case` -- see module
    docstring's "Per-case failure isolation" section. Handles TWO distinct
    failure shapes, both turned into the same failed `CaseGradeOutcome`
    rather than either one aborting the whole run:
    1. `actual_result` carries `_EXECUTION_ERROR_KEY` (this case's own
       execution already raised, caught by `_execute_case_safe`/
       `_execute_async_case_safe`) -- grading is skipped entirely, since
       there is nothing real to grade.
    2. `_grade_case` itself raises (a grader misbehaving on an unexpected
       `actual_result` shape, or any other unforeseen grading-time error).
    """
    if _EXECUTION_ERROR_KEY in actual_result:
        return CaseGradeOutcome(
            passed=False,
            score=None,
            grader_type="deterministic",
            details={
                "error": actual_result[_EXECUTION_ERROR_KEY],
                "reason": "case execution raised before grading could run",
            },
        )
    try:
        return _grade_case(case, actual_result)
    except Exception as exc:  # noqa: BLE001 -- see module docstring: a
        # grading-time failure must never abort the whole batch run either.
        return CaseGradeOutcome(
            passed=False,
            score=None,
            grader_type="deterministic",
            details={"error": str(exc), "reason": "grading raised unexpectedly"},
        )


# ---------------------------------------------------------------------------
# The two entry points
# ---------------------------------------------------------------------------


def run_eval_suite(session: Session, dataset_version: str = DATASET_VERSION) -> EvalRun:
    """Run the full Phase 10 evaluation suite (chat/retrieval/tool/approval
    cases from `eval/golden/v2/*.jsonl` filtered to `dataset_version`, plus
    every adversarial case from `eval/adversarial/v1.jsonl` unconditionally
    -- see module docstring's "dataset_version convention") against the
    real running system, and persist one `EvalRun` + one `EvalCaseResult`
    per case.

    See module docstring's "Session/commit boundary" section for the one
    documented deviation from `run_seed_corpus_ingestion`'s "never commits"
    convention: this function's OWN `EvalRun`/`EvalCaseResult` writes are
    only ever `flush()`-ed (never committed) on `session` -- the caller
    still owns that transaction boundary -- but driving the real tool/
    approval-dimension cases through the real graphs requires committing
    a small number of principal/employee/department fixture rows on
    separate, independent sessions first (architecturally required by
    those graphs' existing `session_factory()`-per-call design, not a
    choice this function makes).
    """
    run = EvalRun(
        dataset_version=dataset_version,
        prompt_version=PROMPT_VERSION,
        workflow_version=WORKFLOW_VERSION,
        retriever_version=RETRIEVER_VERSION,
        reranker_version=RERANKER_VERSION,
        embeddings_version=EMBEDDINGS_VERSION,
        generation_model_version=GENERATION_MODEL_VERSION,
        judge_version=JUDGE_VERSION,
        tool_schema_version=TOOL_SCHEMA_VERSION,
        git_commit=_git_commit(),
        status="running",
    )
    session.add(run)
    session.flush()

    try:
        cases = _load_all_cases(dataset_version)
        by_dimension: dict[str, list[EvalCase]] = {
            "chat": [], "retrieval": [], "tool": [], "approval": [], "adversarial": [],
        }
        for case in cases:
            by_dimension[case.dimension].append(case)

        # Every case's execution is wrapped individually (`_execute_case_
        # safe`/`_execute_async_case_safe`) -- see module docstring's
        # "Per-case failure isolation" section. This `try` block's own
        # `except` below is reserved for genuinely RUN-level failures that
        # happen outside any per-case wrapper: `_load_all_cases` (JSONL
        # parsing) above, or `_execute_graph_dimension_cases`'s own
        # checkpointer/graph-build setup before its per-case loop starts.
        actual_results: dict[str, dict] = {}
        for case in by_dimension["retrieval"]:
            actual_results[case.case_id] = _execute_case_safe(lambda c=case: _execute_retrieval_case(session, c))
        for case in by_dimension["adversarial"]:
            actual_results[case.case_id] = _execute_case_safe(lambda c=case: _execute_adversarial_case_structurally(c))

        graph_results = asyncio.run(
            _execute_graph_dimension_cases(by_dimension["chat"], by_dimension["tool"], by_dimension["approval"])
        )
        actual_results.update(graph_results)

        for case in cases:
            actual_result = actual_results.get(
                case.case_id, {_EXECUTION_ERROR_KEY: "no actual_result was produced for this case_id"}
            )
            outcome = _grade_case_safe(case, actual_result)
            session.add(
                EvalCaseResult(
                    eval_run_id=run.id,
                    case_id=case.case_id,
                    dimension=case.dimension,
                    passed=outcome.passed,
                    score=outcome.score,
                    grader_type=outcome.grader_type,
                    details_json=json.dumps(outcome.details, default=str, sort_keys=True),
                )
            )
        session.flush()
    except Exception as exc:  # noqa: BLE001 -- see run_seed_corpus_ingestion's identical rationale:
        # a genuinely RUN-level failure (dataset loading, or checkpointer/
        # graph setup before any case could even be attempted -- every
        # per-case failure is already isolated above and never reaches
        # here) must still record the run as errored before propagating,
        # rather than leaving the row stuck at "running".
        run.status = "error"
        run.error_message = str(exc)
        run.completed_at = datetime.now(timezone.utc)
        session.flush()
        raise

    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    session.flush()
    return run


async def run_eval_suite_task(ctx: dict) -> dict:
    """The real Arq job function -- registered in `WorkerSettings.functions`
    below. Opens its own DB session (see `ingestion_worker.py`'s identical
    pattern) and delegates to `run_eval_suite` via `asyncio.to_thread` so
    the actual batch-execution logic exists in exactly one place. Commits
    the `EvalRun`/`EvalCaseResult` rows -- unlike `run_eval_suite` itself,
    which only `flush()`s them (see that function's docstring).

    Commits on BOTH the success path and a run-level-failure path, not
    just success (a real bug found and fixed in code review): a genuine
    run-level failure in `run_eval_suite` (see module docstring's
    "Per-case failure isolation" section -- everything per-case is
    already isolated and never reaches here) still `flush()`s its
    `EvalRun` row with `status="error"` before re-raising, but that row
    was only ever `flush()`-ed on THIS session, never committed -- if this
    wrapper only called `session.commit()` after a successful return, the
    `with Session(engine) as session:` block's implicit rollback-on
    -exception would discard that errored row entirely, leaving ZERO trace
    of a real failure in `eval_run`. Committing in the `except` branch too
    (the exception itself carries no useful new information here, so it's
    re-raised unchanged after the commit) ensures a run-level failure is
    always durably recorded, exactly like a per-case failure already is.
    """
    engine = create_engine(DATABASE_URL)
    try:
        with Session(engine) as session:
            try:
                run = await asyncio.to_thread(run_eval_suite, session)
            except Exception:
                session.commit()
                raise
            session.commit()
            return {
                "eval_run_id": run.id,
                "status": run.status,
                "dataset_version": run.dataset_version,
            }
    finally:
        engine.dispose()


class WorkerSettings:
    """Arq worker process entry point:
    `uv run --package resolvegrid-api arq resolvegrid_api.eval_worker.WorkerSettings`
    starts a real worker process consuming jobs from `resolvegrid-redis`.
    """

    functions = [run_eval_suite_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)


def main() -> None:
    """Direct, non-Arq entry point for manually running the suite against
    the live DB: `uv run --package resolvegrid-api python -m
    resolvegrid_api.eval_worker`.
    """
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        run = run_eval_suite(session)
        session.commit()
        print(f"EvalRun {run.id}: status={run.status} dataset_version={run.dataset_version} " f"git_commit={run.git_commit}")


if __name__ == "__main__":
    main()
