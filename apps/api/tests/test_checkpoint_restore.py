"""Checkpoint-restore test: proves LangGraph's `AsyncPostgresSaver` durably
persists agent-run state to real Postgres, recoverable by a completely
separate, independently-constructed graph/checkpointer instance.

Why this test is shaped the way it is (methodology, not a shortcut):
Phase 6's approved plan states the exit criterion as "kill process mid-run
at a node boundary, confirm resume from last checkpoint rather than
restart." A literal OS-level process kill mid-graph-execution is *possible*
to simulate, but it would be inherently flaky in an automated suite --
timing-dependent, platform-dependent process-termination mechanics. This
project already hit real pain with forceful process termination on Windows
during Phase 1's smoke-test work (see `scripts/smoke_test.sh`'s own
comments), so repeating that fragility here for a unit test would trade a
real property proof for a flaky one.

The actual property LangGraph's checkpointing exists to guarantee is:
**state persisted after a run completes survives being read by a
completely separate graph/checkpointer instance** -- no shared Python
object, no shared connection, no in-process cache, only the same
underlying Postgres tables. That is exactly what a real process restart
would also depend on (a fresh process has none of the old process's memory
either, only the database it reconnects to), so this test proves the same
property directly and deterministically instead of indirectly through a
flaky process kill:

1. Build graph instance A (a fresh `AsyncPostgresSaver` + fresh
   `build_graph(...)`), run it to completion for a fixed `thread_id` with a
   mocked `complete_fn` returning deterministic, distinguishable fake
   responses, then fully tear down instance A (exit its `async with`
   block, closing its connection) -- via `asyncio.run()` returning, so
   there is no possibility of any Python-level shared object, cache, or
   even event loop between A and B.
2. Build instance B: a brand-new `AsyncPostgresSaver` + brand-new
   `build_graph(...)`, same connection string / same Postgres tables, but
   a distinct Python object graph with a `complete_fn` that raises if
   called at all (instance B must never re-run the graph -- it only reads
   back what A already persisted).
3. Read back the thread's state via instance B's own compiled graph
   (`graph_b.aget_state(...)`) and assert it matches exactly what instance
   A produced.

This is real proof of durable, cross-instance persistence -- the same
guarantee a real process restart would depend on -- without the flakiness
of literal process termination.

Scope note: this test only exercises LangGraph's own checkpoint tables
(`checkpoints`, `checkpoint_writes`, etc., created by `checkpointer.setup()`)
under a fixed `thread_id`. It builds graphs directly rather than going
through the `/chat` HTTP endpoint, so it creates no `AgentRun`/`Span` rows
and no `Employee` FK dependency at all -- `initial_state`'s
`principal_employee_id` is `None` for exactly this reason, keeping this
test entirely out of `apps/api/src/resolvegrid_api/seed.py`'s
protected-employee-id concerns (see `test_chat_api.py`'s cleanup fixture
for why that concern exists at all -- it doesn't apply here since no
`Employee`/`AgentRun` row is ever created).

Mocking: only `complete_fn` is mocked (as a plain injected callable, per
`services/agent-orchestration`'s dependency-injection design -- see
`graph.py`'s module docstring). The checkpointer is real Postgres, not
mocked, since persistence is exactly the property under test.

Async-without-pytest-asyncio note: this repo has no `pytest-asyncio`
dependency (confirmed: not importable, not in `apps/api/pyproject.toml`),
and this task is scoped to adding one test file only, not adding a new
dependency. So each async flow here is driven by a synchronous test
function calling `asyncio.run(...)` directly, rather than an `async def`
test function -- no plugin required, and each `asyncio.run()` call gets
its own fresh event loop, which if anything makes the A/B separation
*stronger* (not even an event loop is shared between them).

Event-loop-policy note: `AsyncPostgresSaver` uses psycopg's async driver,
which raises `psycopg.InterfaceError` unconditionally under Windows'
default `ProactorEventLoop`. `resolvegrid_api.main` already sets
`asyncio.WindowsSelectorEventLoopPolicy()` at import time (see its own
comment block) to fix exactly this for its `AsyncPostgresSaver` usage.
This test imports `_CHECKPOINTER_DATABASE_URL` from that same module
specifically to reuse its connection-string-translation logic verbatim
(rather than duplicating the translation), and doing so has the side
effect of running that module's top-level `if sys.platform == "win32":
asyncio.set_event_loop_policy(...)` line too -- Python executes a
module's top-level code exactly once, at first import, which happens here
before either `asyncio.run()` call below creates an event loop. So no
separate, explicit event-loop-policy handling is needed in this file: the
import alone is sufficient, verified by this test actually passing on
Windows.
"""

import asyncio
import json
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from resolvegrid_agent_orchestration import build_graph
from resolvegrid_api.main import _CHECKPOINTER_DATABASE_URL

_CLASSIFICATION_RESPONSE = json.dumps({"intent": "general_question", "risk_level": "medium"})
_ANSWER_RESPONSE = "A VPN is a Virtual Private Network that encrypts your connection."


def _initial_state(thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        # Deliberately None: this test builds graphs directly (not through
        # /chat), so no AgentRun/Employee FK exists or is needed -- see
        # module docstring's "Scope note".
        "principal_employee_id": None,
        "input_text": "What is a VPN?",
        "intent": None,
        "risk_level": None,
        "output_text": None,
        "error": None,
    }


def _instance_a_complete_fn():
    """Deterministic, distinguishable fake responses: first call (from
    classify_intent) gets the classification JSON, second call (from
    compose_response) gets the answer text."""
    calls: list[str] = []

    def complete_fn(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return _CLASSIFICATION_RESPONSE
        return _ANSWER_RESPONSE

    return complete_fn


def _instance_b_complete_fn(prompt: str) -> str:
    # Instance B must never call the LLM at all -- it only reads back what
    # instance A already persisted to Postgres. If this is ever called, the
    # test is accidentally re-running the graph instead of proving
    # cross-instance persistence, so fail loudly.
    raise AssertionError(
        "instance B's complete_fn should never be invoked -- it only reads "
        "back already-persisted state via aget_state(), it must not re-run "
        "the graph"
    )


async def _run_instance_a_to_completion(thread_id: str) -> dict:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer_a:
        await checkpointer_a.setup()
        graph_a = build_graph(checkpointer_a, _instance_a_complete_fn())
        result = await graph_a.ainvoke(
            _initial_state(thread_id),
            config={"configurable": {"thread_id": thread_id}},
        )
        return result
    # `async with` exits here -- checkpointer_a's connection is fully closed.
    # Nothing below this point can share any state with checkpointer_a.


async def _read_state_via_instance_b(thread_id: str) -> dict:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer_b:
        # Deliberately NOT calling checkpointer_b.setup() again: Task 3
        # already established setup() is idempotent (safe to call more than
        # once), but skipping it here is the more informative choice for
        # THIS test -- it demonstrates that instance B needs no setup step
        # of its own to read data instance A already persisted, which is
        # exactly what a real fresh-process restart would also do (connect
        # and read, not re-run migrations).
        graph_b = build_graph(checkpointer_b, _instance_b_complete_fn)
        snapshot = await graph_b.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values


async def _delete_thread(thread_id: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer:
        await checkpointer.adelete_thread(thread_id)


def test_checkpoint_restore_cross_instance_persistence():
    """The actual Phase 6 exit criterion: state written by graph instance A
    is correctly recovered by a completely separate graph instance B, proven
    via B's own independent read (`CompiledStateGraph.aget_state`), not by
    re-checking anything through instance A.

    A fresh, random `thread_id` is used per run (not a fixed constant) and
    the thread's checkpoint rows are deleted afterward. Code review of an
    earlier version of this test found a real gap with a fixed thread_id
    and no cleanup: `aget_state()` only reads the LATEST checkpoint for a
    thread, so a stale row left over from an earlier, genuinely-passing run
    could mask a future regression in the write path (`aput`) -- the
    "instance B must never call complete_fn" trick only proves the graph
    didn't re-execute, not that THIS run's write actually succeeded, if an
    old row from a prior run were silently satisfying the read instead.
    """
    thread_id = uuid4().hex
    try:
        result_from_a = asyncio.run(_run_instance_a_to_completion(thread_id))
        assert result_from_a["output_text"] == _ANSWER_RESPONSE
        assert result_from_a["intent"] == "general_question"
        assert result_from_a["risk_level"] == "medium"
        assert result_from_a["error"] is None

        state_from_b = asyncio.run(_read_state_via_instance_b(thread_id))

        assert state_from_b["output_text"] == _ANSWER_RESPONSE
        assert state_from_b["intent"] == "general_question"
        assert state_from_b["risk_level"] == "medium"
        assert state_from_b["input_text"] == "What is a VPN?"
        assert state_from_b["error"] is None
    finally:
        asyncio.run(_delete_thread(thread_id))
