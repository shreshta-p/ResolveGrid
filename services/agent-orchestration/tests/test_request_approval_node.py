"""Unit tests for Phase 9 Task 5's `request_approval` node
(`make_request_approval_node`, `graph.py`).

`request_approval` calls LangGraph's real `interrupt()` -- unlike every
other node in `test_graph.py`, this cannot be exercised by calling the
node function directly (an uncaught `interrupt()` raises `GraphInterrupt`
outside a compiled graph). So each test here builds a small, ad hoc
`StateGraph` containing ONLY this node (mirroring the exact pattern
`langgraph.types.interrupt`'s own docstring example uses), compiled with a
real `InMemorySaver` -- proving `interrupt()`/`Command(resume=...)` works
correctly in isolation, without needing a real Postgres
`AsyncPostgresSaver` (that durability guarantee is what
`test_checkpoint_restore.py`-style tests exist for; this task explicitly
does not repeat a full real interrupt/restart/resume cycle -- that's
Task 8's job).

`request_approval_fn` itself is always a fake here -- proving the injected
callable receives the right payload shape and that the node correctly
threads the resumed decision into state. The real `apps/api`
implementation's actual idempotent-upsert-by-identity behavior against
real Postgres is proven separately in
`apps/api/tests/test_approval_service.py`.

Not wired into `build_graph` -- see graph.py's module docstring's "Phase 9
Task 5" section for why this node is standalone by deliberate design in
this task, so these tests build their own tiny graph instead of using
`build_graph`.
"""

import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from resolvegrid_agent_orchestration.graph import make_request_approval_node
from resolvegrid_agent_orchestration.state import AgentState

_BASE_STATE = {
    "thread_id": "t1",
    "principal_employee_id": None,
    "input_text": "placeholder",
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
}


def _state(**overrides) -> dict:
    return {**_BASE_STATE, **overrides}


def _build_standalone_graph():
    """A minimal graph containing only `request_approval` -- see module
    docstring for why this node can't be unit-tested by calling it as a
    bare function."""

    def _make(request_approval_fn):
        builder = StateGraph(AgentState)
        builder.add_node("request_approval", make_request_approval_node(request_approval_fn))
        builder.add_edge(START, "request_approval")
        return builder.compile(checkpointer=InMemorySaver())

    return _make


def test_request_approval_node_calls_fn_with_expected_payload_shape():
    captured_payloads = []

    def fake_request_approval_fn(payload):
        captured_payloads.append(payload)
        return {"approval_request_id": 99, "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(
            thread_id=thread_id,
            principal_employee_id=7,
            proposed_tool_name="grant_vpn_access",
            proposed_tool_params={"employee_id": 1, "justification": "new hire"},
            risk_level="medium",
        ),
        config,
    )

    assert len(captured_payloads) == 1
    assert captured_payloads[0] == {
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 1, "justification": "new hire"},
        "actor": 7,
        "evidence_refs": [],
        "risk_context": "medium",
        "agent_run_id": thread_id,
    }


def test_request_approval_node_defaults_missing_proposed_params_to_empty_dict():
    captured_payloads = []

    def fake_request_approval_fn(payload):
        captured_payloads.append(payload)
        return {"approval_request_id": 1, "status": "pending", "expires_at": "x"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(thread_id=thread_id, proposed_tool_name="grant_vpn_access", proposed_tool_params=None),
        config,
    )

    assert captured_payloads[0]["params"] == {}


def test_request_approval_node_pauses_via_real_interrupt_with_human_readable_payload():
    def fake_request_approval_fn(payload):
        return {"approval_request_id": 42, "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        _state(
            thread_id=thread_id,
            proposed_tool_name="grant_vpn_access",
            proposed_tool_params={"employee_id": 1, "justification": "new hire"},
            risk_level="high",
        ),
        config,
    )

    # A real interrupt() call durably pauses the graph -- .invoke() returns
    # the state as of the pause point, plus a `__interrupt__` key, instead
    # of running to completion (verified against the installed
    # langgraph==1.2.11's actual runtime behavior, not assumed).
    assert "__interrupt__" in result
    interrupt_obj = result["__interrupt__"][0]
    assert interrupt_obj.value == {
        "approval_request_id": 42,
        "action_type": "grant_vpn_access",
        "params": {"employee_id": 1, "justification": "new hire"},
        "risk_context": "high",
        "expires_at": "2026-09-05T00:00:00+00:00",
        "status": "pending",
    }
    # The node has not yet set approval_decision -- the graph run never
    # reached its `return` statement.
    assert result.get("approval_decision") is None


def test_request_approval_node_threads_resumed_decision_into_state_on_approval():
    def fake_request_approval_fn(payload):
        return {"approval_request_id": 42, "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(thread_id=thread_id, proposed_tool_name="grant_vpn_access", proposed_tool_params={}),
        config,
    )

    resumed = graph.invoke(Command(resume="approved"), config)

    assert resumed["approval_request_id"] == 42
    assert resumed["approval_decision"] == "approved"
    assert "__interrupt__" not in resumed


def test_request_approval_node_threads_resumed_decision_into_state_on_rejection():
    def fake_request_approval_fn(payload):
        return {"approval_request_id": 7, "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(thread_id=thread_id, proposed_tool_name="grant_vpn_access", proposed_tool_params={}),
        config,
    )
    resumed = graph.invoke(Command(resume="rejected"), config)

    assert resumed["approval_decision"] == "rejected"


def test_request_approval_node_re_executes_and_recalls_fn_on_resume():
    """Empirical proof of the re-execution semantics this task's design
    depends on (see graph.py's module docstring/`approval_service.py`'s
    "wall-clock idempotency hazard" note): `request_approval_fn` is called
    AGAIN, with an identical payload, when the node resumes -- it is not
    somehow skipped just because it ran once already. This is exactly why
    `request_approval_fn` itself must be idempotent; a real, non-idempotent
    fake here would visibly create two "records" for what should be one
    logical approval.
    """
    captured_payloads = []

    def fake_request_approval_fn(payload):
        captured_payloads.append(payload)
        return {"approval_request_id": 55, "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(
            thread_id=thread_id,
            proposed_tool_name="grant_vpn_access",
            proposed_tool_params={"employee_id": 1},
        ),
        config,
    )
    assert len(captured_payloads) == 1

    graph.invoke(Command(resume="approved"), config)

    assert len(captured_payloads) == 2
    assert captured_payloads[0] == captured_payloads[1]


def test_request_approval_node_idempotent_fake_returns_same_id_across_re_execution():
    """Node-level complement to `apps/api`'s real-DB idempotency test: a
    stateful fake that mimics an upsert-by-identity `request_approval_fn`
    (keyed on everything but a fictitious wall-clock field, matching
    `approval_service.py`'s real design) proves the node's resumed state
    ends up pointing at the SAME `approval_request_id` both times, never a
    second one -- exactly the property `apps/api/tests/
    test_approval_service.py` proves again against real Postgres.
    """
    store: dict[tuple, int] = {}
    next_id = [1]

    def fake_upserting_request_approval_fn(payload):
        key = (
            payload["agent_run_id"],
            payload["action_type"],
            tuple(sorted(payload["params"].items())),
            payload["actor"],
            payload["risk_context"],
        )
        if key not in store:
            store[key] = next_id[0]
            next_id[0] += 1
        return {"approval_request_id": store[key], "status": "pending", "expires_at": "2026-09-05T00:00:00+00:00"}

    graph = _build_standalone_graph()(fake_upserting_request_approval_fn)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        _state(
            thread_id=thread_id,
            proposed_tool_name="grant_vpn_access",
            proposed_tool_params={"employee_id": 1},
        ),
        config,
    )
    resumed = graph.invoke(Command(resume="approved"), config)

    assert resumed["approval_request_id"] == 1
    assert len(store) == 1  # only one logical ApprovalRequest was ever created
