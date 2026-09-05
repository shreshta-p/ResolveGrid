"""Phase 9 Task 8: the restart-mid-approval cross-instance durability test --
the phase's primary exit-criteria proof artifact.

Why this test is shaped the way it is (methodology, not a shortcut): this is
the SAME property `test_checkpoint_restore.py` proves for a COMPLETED run,
applied instead to a run that is durably PAUSED at a real `interrupt()`. See
that file's module docstring for the full argument against a literal
OS-level process kill (flaky, platform-dependent) in favor of the actual
guaranteed property: state persisted to Postgres by one graph/checkpointer
instance is correctly recoverable and resumable by a completely separate,
independently-constructed graph/checkpointer instance -- no shared Python
object, no shared connection, no in-process cache, only the same underlying
Postgres tables. That is exactly what a real process restart would also
depend on.

1. Instance A: a fresh `AsyncPostgresSaver` + a fresh
   `build_tool_invocation_graph(checkpointer_a, request_approval_for_agent,
   execute_mutation_for_agent)` (the REAL Phase 9 Task 5/7a DI
   implementations, not mocks -- this test exercises the real idempotent
   upsert and the real mutation dispatch, exactly what the exit criteria
   requires). `ainvoke()`d with a real `grant_vpn_access` proposal for a
   real `Employee` row. This call is expected to hit `request_approval`'s
   `interrupt()` and return WITHOUT completing (a result dict carrying
   `"__interrupt__"`, per `routers/tools.py`'s own documented, verified-
   against-the-installed-langgraph behavior -- reused here rather than
   re-derived). Instance A is then fully torn down (its `async with` block
   exits, closing its connection) -- via `asyncio.run()` returning, so
   there is no possibility of any Python-level shared object, cache, or
   even event loop surviving into instance B.
2. A DIRECT DB query (via `raw_db_session`, not through either graph
   instance) confirms exactly one `ApprovalRequest` row exists with
   `status="pending"` for this run's `agent_run_id` (== `thread_id`).
3. Between A and B: writes the `ApprovalDecision` row and flips
   `ApprovalRequest.status` to `"approved"`, via a direct, real-committing
   DB write -- mirroring `routers/approvals.py`'s `decide_approval` exactly
   (see that router's "Commit-then-resume ordering" docstring section). A
   REAL finding this test caught while being written (not assumed): neither
   `request_approval` nor `execute_mutation` (`graph.py`'s nodes) ever
   writes `ApprovalRequest.status` themselves -- `request_approval_fn`
   inserts the row as `status="pending"` and never touches it again, and
   `execute_mutation`'s real implementation (`mutation_execution.
   execute_mutation`) only ever READS `row.status` to gate execution, it
   never sets `status="approved"`. That transition is `routers/approvals.py`'s
   job alone, performed as a separate, committed write strictly BEFORE the
   resume call. A first draft of this test skipped that write and called
   `Command(resume="approved")` directly against a still-`"pending"` row --
   `execute_mutation_for_agent` correctly refused with
   `ApprovalNotDecidedError` (`status='pending'` is not `"approved"`),
   proving the status check is real, not a formality, and that this test
   must reproduce the real caller's full write sequence, not just the
   `Command(resume=...)` call in isolation, to reach the success path this
   test is actually meant to verify.
4. Instance B: a brand-new `AsyncPostgresSaver` + a brand-new
   `build_tool_invocation_graph(...)`, same connection string / same
   Postgres tables, but a distinct Python object graph -- still wired to
   the SAME real `request_approval_for_agent`/`execute_mutation_for_agent`
   functions (unlike `test_checkpoint_restore.py`'s instance B, which must
   NEVER re-run anything: THIS test's instance B is expected to genuinely
   re-execute `request_approval` -- LangGraph re-runs an interrupted node's
   entire body from the top on every resume, per `graph.py`'s module
   docstring -- and drive the new `execute_mutation` node to completion for
   the first time. The whole point of Task 5's idempotent-upsert design is
   that this re-execution is safe, not that it doesn't happen). Resumed via
   `graph_b.ainvoke(Command(resume="approved"), config={"configurable":
   {"thread_id": thread_id}})` -- the exact resume-value shape confirmed by
   reading both `graph.py`'s `make_request_approval_node` (the resumed
   value is threaded directly into `state["approval_decision"]`, compared
   against the plain strings `"approved"`/`"rejected"` by
   `make_execute_mutation_node` -- no dict wrapper) and
   `routers/approvals.py`'s real caller (`Command(resume=payload.decision)`,
   `payload.decision` being that same plain string).
5. DIRECT DB assertions (again via `raw_db_session`, never through either
   graph instance) prove zero duplicate mutation side effects: exactly ONE
   `ApprovalRequest` row still exists (not two -- proves `request_approval`'s
   real re-execution-on-resume was correctly absorbed by the idempotent
   upsert rather than creating a duplicate), exactly ONE active
   `EmployeeEntitlement` grant for the target employee, exactly ONE
   `ToolCall` row with `status="success"` for this approval's idempotency
   key. This is the actual proof the phase's exit criteria requires.

Async-without-pytest-asyncio note, event-loop-policy note: identical to
`test_checkpoint_restore.py` -- see that file's module docstring for the
full explanation. This file imports `_CHECKPOINTER_DATABASE_URL` from the
same `resolvegrid_api.main` module for the same reason (reusing its
connection-string translation, and picking up its module-level Windows
event-loop-policy side effect for free).

Employee/seed-data note: this test creates its own `Employee` row via the
committing `raw_db_session` fixture (never the shared `seed_corpus`/
ingestion path -- see `docs/DECISION_LOG.md`'s 2026-08-28 empty-corpus
entry) with a fresh `uuid4()` suffix per run. `Employee`/`AuditLog` rows are
deliberately NOT deleted in cleanup -- `execute_mutation`'s success path
writes a permanent, append-only `AuditLog` row whose `actor_id` FK
references this employee (no cascade delete -- see `audit.py`'s module
docstring), matching `test_mutation_execution.py`'s
`test_execute_mutation_closes_the_concurrent_replay_race` and
`test_approvals_router.py`'s `requester_employee` fixture, both of which
already established this exact precedent.

Docker/Postgres required: this test needs the real `resolvegrid` Postgres
container running (same requirement as every sibling adversarial test in
this file's neighborhood).
"""

import asyncio
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy import delete, select

from resolvegrid_agent_orchestration import build_tool_invocation_graph
from resolvegrid_api.agent_mutation_execution import execute_mutation_for_agent
from resolvegrid_api.approval_service import request_approval_for_agent
from resolvegrid_api.main import _CHECKPOINTER_DATABASE_URL
from resolvegrid_api.models import ApprovalDecision, ApprovalRequest, ToolCall
from resolvegrid_api.models.org import Employee, EmployeeEntitlement

_ACTION_TYPE = "grant_vpn_access"
_JUSTIFICATION = "restart-mid-approval test: onboarding contractor"


def _initial_state(thread_id: str, employee_id: int) -> dict:
    """Mirrors `routers/tools.py`'s `POST /tools/{tool_name}/invoke` initial
    state construction verbatim (same field set, same placeholder values for
    fields this graph's two nodes never read) -- this test invokes
    `build_tool_invocation_graph` directly rather than through the HTTP
    endpoint, but the initial `AgentState` a real caller would seed is
    exactly this shape.
    """
    return {
        "thread_id": thread_id,
        "principal_employee_id": employee_id,
        "input_text": f"[tool invocation] {_ACTION_TYPE}",
        "intent": None,
        "risk_level": "high",
        "retrieval_scope": None,
        "retrieved_chunks": None,
        "retrieval_sufficient": None,
        "context_block": None,
        "output_text": None,
        "error": None,
        "citations_verified": None,
        "verified_chunk_ids": None,
        "fabricated_chunk_ids": None,
        "proposed_tool_name": _ACTION_TYPE,
        "proposed_tool_params": {"employee_id": employee_id, "justification": _JUSTIFICATION},
        "approval_request_id": None,
        "approval_decision": None,
        "tool_invocation_result": None,
    }


async def _run_instance_a_to_interrupt(thread_id: str, employee_id: int) -> dict:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer_a:
        await checkpointer_a.setup()
        graph_a = build_tool_invocation_graph(
            checkpointer_a, request_approval_for_agent, execute_mutation_for_agent
        )
        result = await graph_a.ainvoke(
            _initial_state(thread_id, employee_id),
            config={"configurable": {"thread_id": thread_id}},
        )
        return result
    # `async with` exits here -- instance A's connection is fully closed.
    # Nothing below this point can share any state with checkpointer_a.


async def _resume_via_instance_b(thread_id: str) -> dict:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer_b:
        # Deliberately NOT calling checkpointer_b.setup() again -- mirrors
        # test_checkpoint_restore.py's instance B: a real fresh-process
        # restart would also just connect and resume, not re-run migrations.
        graph_b = build_tool_invocation_graph(
            checkpointer_b, request_approval_for_agent, execute_mutation_for_agent
        )
        result = await graph_b.ainvoke(
            Command(resume="approved"),
            config={"configurable": {"thread_id": thread_id}},
        )
        return result


async def _delete_thread(thread_id: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer:
        await checkpointer.adelete_thread(thread_id)


def test_restart_mid_approval_cross_instance_resume_executes_mutation_exactly_once(raw_db_session):
    """The actual Phase 9 exit criterion: a `grant_vpn_access` run paused at
    `request_approval`'s `interrupt()` by graph instance A, whose connection
    is then fully closed, is correctly resumed to completion by a completely
    separate graph instance B -- with zero duplicate `ApprovalRequest` rows,
    zero duplicate `EmployeeEntitlement` grants, and zero duplicate
    `ToolCall` success rows, proven via direct DB row-count assertions, not
    by re-checking anything through either graph instance.
    """
    thread_id = uuid.uuid4().hex
    unique_suffix = uuid.uuid4().hex[:12]
    employee = Employee(
        display_name="Restart Mid Approval Employee",
        email=f"restart.mid.approval.{unique_suffix}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
    )
    raw_db_session.add(employee)
    raw_db_session.commit()
    employee_id = employee.id

    approval_request_id = None
    try:
        # --- Instance A: reach interrupt(), then fully tear down --------
        result_a = asyncio.run(_run_instance_a_to_interrupt(thread_id, employee_id))
        assert isinstance(result_a, dict)
        interrupts_a = result_a.get("__interrupt__")
        assert interrupts_a, (
            "expected instance A's ainvoke() to pause at request_approval's "
            f"interrupt(), got a result with no __interrupt__ key: {result_a!r}"
        )
        approval_request_id = interrupts_a[0].value["approval_request_id"]
        assert isinstance(approval_request_id, int)
        assert interrupts_a[0].value["status"] == "pending"
        assert interrupts_a[0].value["action_type"] == _ACTION_TYPE

        # Direct DB check -- NOT through either graph instance -- that
        # instance A's interrupted run left exactly one pending
        # ApprovalRequest row behind.
        pending_rows = (
            raw_db_session.execute(
                select(ApprovalRequest).where(ApprovalRequest.agent_run_id == thread_id)
            )
            .scalars()
            .all()
        )
        assert len(pending_rows) == 1
        assert pending_rows[0].id == approval_request_id
        assert pending_rows[0].status == "pending"

        # --- The approver-decide write: ApprovalDecision + status flip,
        # committed BEFORE the resume call -- mirrors routers/approvals.py's
        # decide_approval exactly (see module docstring's step 3 for the
        # real gap this test caught by first skipping this step).
        raw_db_session.add(
            ApprovalDecision(
                approval_request_id=approval_request_id,
                approver_id=employee_id,
                decision="approved",
                comment="restart-mid-approval test: approved",
            )
        )
        pending_row = raw_db_session.get(ApprovalRequest, approval_request_id)
        pending_row.status = "approved"
        raw_db_session.commit()

        # --- Instance B: brand-new checkpointer + brand-new compiled
        # graph, independent of instance A's already-closed connection.
        result_b = asyncio.run(_resume_via_instance_b(thread_id))
        assert not result_b.get("__interrupt__"), (
            "instance B's resume should run request_approval -> "
            f"execute_mutation to completion, not interrupt again: {result_b!r}"
        )
        assert result_b["approval_request_id"] == approval_request_id
        assert result_b["approval_decision"] == "approved"
        tool_result = result_b["tool_invocation_result"]
        assert tool_result is not None
        assert tool_result["status"] == "success", f"execute_mutation failed: {tool_result}"
        assert tool_result["error"] is None
        assert tool_result["output"]["employee_id"] == employee_id

        # --- Zero-duplicate-side-effects proof, via DIRECT DB queries ---
        approval_rows_after = (
            raw_db_session.execute(
                select(ApprovalRequest).where(ApprovalRequest.agent_run_id == thread_id)
            )
            .scalars()
            .all()
        )
        assert len(approval_rows_after) == 1, (
            "request_approval's real re-execution on resume should have been "
            "absorbed by Task 5's idempotent upsert, not created a second "
            f"ApprovalRequest row -- found {len(approval_rows_after)}"
        )
        assert approval_rows_after[0].status == "approved"

        grants = (
            raw_db_session.execute(
                select(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee_id)
            )
            .scalars()
            .all()
        )
        assert len(grants) == 1, (
            "expected exactly 1 EmployeeEntitlement grant after a cross-"
            f"instance restart-then-resume, found {len(grants)}"
        )

        idempotency_key = f"approval:{approval_request_id}"
        success_calls = (
            raw_db_session.execute(
                select(ToolCall).where(
                    ToolCall.idempotency_key == idempotency_key, ToolCall.status == "success"
                )
            )
            .scalars()
            .all()
        )
        assert len(success_calls) == 1, (
            f"expected exactly 1 success ToolCall row, found {len(success_calls)} -- "
            "execute_mutation_fn was invoked more than once across the "
            "instance-A-to-instance-B restart"
        )
        assert success_calls[0].tool_name == _ACTION_TYPE
        assert success_calls[0].approval_request_id == approval_request_id
    finally:
        asyncio.run(_delete_thread(thread_id))
        if approval_request_id is not None:
            raw_db_session.execute(
                delete(ToolCall).where(ToolCall.approval_request_id == approval_request_id)
            )
            raw_db_session.execute(
                delete(ApprovalDecision).where(ApprovalDecision.approval_request_id == approval_request_id)
            )
        raw_db_session.execute(
            delete(EmployeeEntitlement).where(EmployeeEntitlement.employee_id == employee_id)
        )
        if approval_request_id is not None:
            raw_db_session.execute(delete(ApprovalRequest).where(ApprovalRequest.id == approval_request_id))
        raw_db_session.commit()
        # Employee/AuditLog rows are deliberately NOT deleted -- see module
        # docstring's "Employee/seed-data note" for why.
