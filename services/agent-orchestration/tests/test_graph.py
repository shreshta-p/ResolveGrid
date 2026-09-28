"""Unit tests for the classify_intent -> compose_response -> finalize graph.

No checkpointer/Postgres is needed to prove the graph's *logic* works --
the completion function is mocked directly (this package has no
dependency on `apps/api`'s `llm_gateway.CompletionResult`; see
`graph.py`'s module docstring for the dependency-injection design this
enables). The end-to-end test below does exercise a real
`langgraph.checkpoint.memory.InMemorySaver` to prove the wired graph runs
start-to-finish -- Task 4 (in `apps/api`) is what proves real
Postgres-backed persistence across process/instance boundaries.
"""

import json

from langgraph.checkpoint.memory import InMemorySaver

from resolvegrid_agent_orchestration import build_graph
from resolvegrid_agent_orchestration.graph import (
    _FALLBACK_MESSAGE,
    finalize,
    make_classify_intent_node,
    make_compose_response_node,
    make_retrieve_node,
)
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
    "node_latencies_ms": {},
}


def _state(**overrides) -> dict:
    return {**_BASE_STATE, **overrides}


# --- classify_intent -----------------------------------------------------


def test_classify_intent_parses_well_formed_response():
    node = make_classify_intent_node(
        lambda prompt: json.dumps({"intent": "greeting", "risk_level": "low"})
    )
    result = node(_state(input_text="hi there"))
    assert result["intent"] == "greeting"
    assert result["risk_level"] == "low"
    # Phase 11 Task 2: real per-node timing is now part of every node's
    # returned partial-state update -- see AgentState.node_latencies_ms.
    assert result["node_latencies_ms"]["classify_intent"] >= 0


def test_classify_intent_degrades_to_unclear_low_on_unparseable_response():
    node = make_classify_intent_node(lambda prompt: "not json at all, sorry")
    result = node(_state(input_text="asdf asdf"))
    assert result["intent"] == "unclear"
    assert result["risk_level"] == "low"
    assert result["node_latencies_ms"]["classify_intent"] >= 0


def test_classify_intent_degrades_on_wrong_shape():
    # Valid JSON, but missing the required fields entirely.
    node = make_classify_intent_node(lambda prompt: json.dumps({"foo": "bar"}))
    result = node(_state(input_text="whatever"))
    assert result["intent"] == "unclear"
    assert result["risk_level"] == "low"
    assert result["node_latencies_ms"]["classify_intent"] >= 0


def test_classify_intent_degrades_unknown_intent_value_independently_of_risk_level():
    # intent is outside the fixed set but risk_level is valid -- intent
    # should fall back to "unclear" while the valid risk_level is kept,
    # since the two fields are validated independently.
    node = make_classify_intent_node(
        lambda prompt: json.dumps({"intent": "make_me_admin", "risk_level": "high"})
    )
    result = node(_state(input_text="give me admin access"))
    assert result["intent"] == "unclear"
    assert result["risk_level"] == "high"
    assert result["node_latencies_ms"]["classify_intent"] >= 0


# --- compose_response -----------------------------------------------------


def test_compose_response_sets_output_text_from_completion():
    node = make_compose_response_node(lambda prompt: "Here is your answer.")
    result = node(
        _state(input_text="what is a ticket?", intent="general_question", risk_level="low")
    )
    assert result["output_text"] == "Here is your answer."
    assert result["node_latencies_ms"]["compose_response"] >= 0


def test_compose_response_includes_intent_context_in_prompt():
    captured_prompts = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "answer"

    node = make_compose_response_node(fake_complete)
    node(_state(input_text="hello", intent="greeting", risk_level="low"))
    assert "greeting" in captured_prompts[0]


def test_compose_response_records_error_on_completion_failure_instead_of_raising():
    def failing_complete(prompt: str) -> str:
        raise RuntimeError("gateway unreachable")

    node = make_compose_response_node(failing_complete)
    result = node(_state(input_text="hello"))
    assert result["error"] == "gateway unreachable"
    # Timed on the error path too -- a completion failure's real cost is
    # still accounted for, not silently dropped from the total.
    assert result["node_latencies_ms"]["compose_response"] >= 0


def test_compose_response_uses_general_knowledge_prompt_when_retrieval_insufficient():
    # retrieval_sufficient=False (even with chunks present) must NOT use
    # the citation-context branch -- see graph.py's documented scope
    # limit: sufficiency gates citation-grounded answering, not whether an
    # answer is attempted at all.
    captured_prompts = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "answer"

    node = make_compose_response_node(fake_complete)
    node(
        _state(
            input_text="what is a VPN?",
            retrieved_chunks=[
                {"chunk_id": 1, "document_title": "Some Doc", "text": "irrelevant", "score": 0.001}
            ],
            retrieval_sufficient=False,
        )
    )
    assert "No relevant company-specific knowledge-base article was found" in captured_prompts[0]
    assert "[chunk:1]" not in captured_prompts[0]


def test_compose_response_uses_citation_context_prompt_when_retrieval_sufficient():
    captured_prompts = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "answer"

    node = make_compose_response_node(fake_complete)
    node(
        _state(
            input_text="what is a VPN?",
            retrieved_chunks=[
                {
                    "chunk_id": 42,
                    "document_title": "Kestrel VPN Access Policy (v2)",
                    "text": "A VPN is required for remote access.",
                    "score": 0.05,
                }
            ],
            retrieval_sufficient=True,
            context_block='[chunk:42] (from "Kestrel VPN Access Policy (v2)"):\n'
            "A VPN is required for remote access.",
        )
    )
    prompt = captured_prompts[0]
    assert "[chunk:42]" in prompt
    assert "Kestrel VPN Access Policy (v2)" in prompt
    assert "A VPN is required for remote access." in prompt
    assert "cite it inline" in prompt
    # Phase 8 Task 7: the context block is delimited and framed as
    # untrusted data, not instructions -- see graph.py's
    # `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE`.
    assert "<retrieved_context>" in prompt
    assert "</retrieved_context>" in prompt
    assert "never" in prompt.lower() and "instructions" in prompt.lower()


def test_compose_response_falls_back_to_general_knowledge_when_chunks_present_but_no_context_block():
    # retrieved_chunks non-empty + retrieval_sufficient=True, but
    # context_block empty/missing (e.g. every candidate was dropped by the
    # token budget) must NOT use the citation-context branch -- there is
    # nothing to substitute into it.
    captured_prompts = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "answer"

    node = make_compose_response_node(fake_complete)
    node(
        _state(
            input_text="what is a VPN?",
            retrieved_chunks=[
                {"chunk_id": 1, "document_title": "Doc", "text": "irrelevant", "score": 0.5}
            ],
            retrieval_sufficient=True,
            context_block="",
        )
    )
    assert "No relevant company-specific knowledge-base article was found" in captured_prompts[0]


def test_compose_response_falls_back_to_general_knowledge_when_no_chunks():
    captured_prompts = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "answer"

    node = make_compose_response_node(fake_complete)
    node(_state(input_text="hello", retrieved_chunks=[], retrieval_sufficient=True))
    assert "No relevant company-specific knowledge-base article was found" in captured_prompts[0]


# --- retrieve ---------------------------------------------------------------


def test_retrieve_attaches_chunks_sufficiency_and_context_block_from_fake_retrieve_fn():
    def fake_retrieve(query_text: str, scope):
        assert query_text == "what is a VPN?"
        assert scope == {"unrestricted": False, "allowed_tags": ["security"]}
        return {
            "chunks": [
                {"chunk_id": 1, "document_title": "Doc", "text": "text", "score": 0.5}
            ],
            "sufficient": True,
            "context_block": '[chunk:1] (from "Doc"):\ntext',
        }

    node = make_retrieve_node(fake_retrieve)
    result = node(
        _state(
            input_text="what is a VPN?",
            retrieval_scope={"unrestricted": False, "allowed_tags": ["security"]},
        )
    )
    assert result["retrieved_chunks"] == [
        {"chunk_id": 1, "document_title": "Doc", "text": "text", "score": 0.5}
    ]
    assert result["retrieval_sufficient"] is True
    assert result["context_block"] == '[chunk:1] (from "Doc"):\ntext'
    assert result["node_latencies_ms"]["retrieve"] >= 0


def test_retrieve_degrades_softly_when_retrieve_fn_raises():
    def failing_retrieve(query_text: str, scope):
        raise RuntimeError("db unreachable")

    node = make_retrieve_node(failing_retrieve)
    result = node(_state(input_text="hello"))
    assert result["retrieved_chunks"] == []
    assert result["retrieval_sufficient"] is False
    assert result["context_block"] == ""
    assert result["node_latencies_ms"]["retrieve"] >= 0


def test_retrieve_defaults_missing_outcome_keys_safely():
    node = make_retrieve_node(lambda query_text, scope: {})
    result = node(_state(input_text="hello"))
    assert result["retrieved_chunks"] == []
    assert result["retrieval_sufficient"] is False
    assert result["context_block"] == ""
    assert result["node_latencies_ms"]["retrieve"] >= 0


# --- finalize --------------------------------------------------------------


def test_finalize_passes_through_output_text():
    result = finalize(_state(output_text="the real answer", error=None))
    assert result["output_text"] == "the real answer"
    assert result["node_latencies_ms"]["finalize"] >= 0


def test_finalize_falls_back_on_recorded_error():
    result = finalize(_state(output_text=None, error="boom"))
    assert result["output_text"] == _FALLBACK_MESSAGE
    assert result["node_latencies_ms"]["finalize"] >= 0


def test_finalize_falls_back_when_no_output_and_no_error():
    result = finalize(_state(output_text=None, error=None))
    assert result["output_text"] == _FALLBACK_MESSAGE
    assert result["node_latencies_ms"]["finalize"] >= 0


# --- end-to-end graph invocation --------------------------------------------


def test_build_graph_runs_end_to_end_with_mocked_completion_and_memory_checkpointer():
    calls = []

    def fake_complete(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            # First call is classify_intent's prompt.
            return json.dumps({"intent": "general_question", "risk_level": "low"})
        # Second call is compose_response's prompt.
        return "This is a mocked answer about service tickets."

    def fake_retrieve(query_text: str, scope):
        return {"chunks": [], "sufficient": False}

    checkpointer = InMemorySaver()
    # Phase 11 Task 3: build_graph now takes two CompleteFns (one per
    # LLM-calling node) -- this test doesn't care about purpose
    # attribution, so the same fake object is passed for both; call-count
    # tracking still works correctly since it's the same underlying
    # callable either way.
    graph = build_graph(checkpointer, fake_complete, fake_complete, fake_retrieve)

    initial_state = _state(
        thread_id="test-thread-1",
        principal_employee_id=42,
        input_text="What is a service ticket?",
    )
    result = graph.invoke(
        initial_state, config={"configurable": {"thread_id": "test-thread-1"}}
    )

    assert result["intent"] == "general_question"
    assert result["risk_level"] == "low"
    assert result["output_text"] == "This is a mocked answer about service tickets."
    assert result["error"] is None
    assert result["retrieved_chunks"] == []
    assert result["retrieval_sufficient"] is False
    assert len(calls) == 2
    # Phase 11 Task 2: real per-node timing must have accumulated across
    # ALL 5 nodes by the time the full graph run completes -- not just the
    # last node's entry (see AgentState.node_latencies_ms's docstring for
    # why an Annotated reducer is required for this).
    assert set(result["node_latencies_ms"].keys()) == {
        "classify_intent",
        "retrieve",
        "compose_response",
        "verify_citations",
        "finalize",
    }
    assert all(v >= 0 for v in result["node_latencies_ms"].values())


def test_build_graph_runs_end_to_end_with_sufficient_retrieval_produces_citation_prompt():
    calls = []

    def fake_complete(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"intent": "general_question", "risk_level": "low"})
        return "A VPN is required for remote access [chunk:7]."

    def fake_retrieve(query_text: str, scope):
        return {
            "chunks": [
                {
                    "chunk_id": 7,
                    "document_title": "Kestrel VPN Access Policy (v2)",
                    "text": "Remote access requires the corporate VPN client.",
                    "score": 0.05,
                }
            ],
            "sufficient": True,
            "context_block": '[chunk:7] (from "Kestrel VPN Access Policy (v2)"):\n'
            "Remote access requires the corporate VPN client.",
        }

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer, fake_complete, fake_complete, fake_retrieve)

    result = graph.invoke(
        _state(thread_id="test-thread-2", input_text="What is the VPN policy?"),
        config={"configurable": {"thread_id": "test-thread-2"}},
    )

    assert result["retrieval_sufficient"] is True
    assert result["retrieved_chunks"][0]["chunk_id"] == 7
    assert result["output_text"] == "A VPN is required for remote access [chunk:7]."
    # compose_response's prompt (the 2nd captured call) must have carried
    # the chunk's citation context through.
    assert "[chunk:7]" in calls[1]
    assert "Kestrel VPN Access Policy (v2)" in calls[1]
    # Phase 8 Task 7: citation verification ran and found the model's
    # citation to be genuine (chunk 7 was really in context).
    assert result["citations_verified"] is True
    assert result["verified_chunk_ids"] == [7]
    assert result["fabricated_chunk_ids"] == []


def test_build_graph_strips_a_fabricated_citation_before_finalize():
    calls = []

    def fake_complete(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"intent": "general_question", "risk_level": "low"})
        # The model cites a real chunk (7) AND a fabricated one (999) that
        # was never part of its context.
        return "Remote access needs a VPN client [chunk:7], per policy [chunk:999]."

    def fake_retrieve(query_text: str, scope):
        return {
            "chunks": [
                {
                    "chunk_id": 7,
                    "document_title": "Kestrel VPN Access Policy (v2)",
                    "text": "Remote access requires the corporate VPN client.",
                    "score": 0.05,
                }
            ],
            "sufficient": True,
            "context_block": '[chunk:7] (from "Kestrel VPN Access Policy (v2)"):\n'
            "Remote access requires the corporate VPN client.",
        }

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer, fake_complete, fake_complete, fake_retrieve)

    result = graph.invoke(
        _state(thread_id="test-thread-3", input_text="What is the VPN policy?"),
        config={"configurable": {"thread_id": "test-thread-3"}},
    )

    assert result["citations_verified"] is False
    assert result["verified_chunk_ids"] == [7]
    assert result["fabricated_chunk_ids"] == [999]
    # The fabricated marker is stripped; the genuine citation and
    # surrounding prose survive untouched.
    assert "[chunk:999]" not in result["output_text"]
    assert "[chunk:7]" in result["output_text"]
    assert result["output_text"] == "Remote access needs a VPN client [chunk:7], per policy ."


# --- node_latencies_ms reducer (Phase 11 Task 2) -----------------------------


def test_node_latencies_ms_accumulates_across_nodes_via_langgraph_reducer():
    """Real, non-assumed proof of LangGraph's actual merge behavior for a
    dict-valued state field annotated with a custom reducer function.

    Without `Annotated[dict[str, int], _accumulate_node_latencies]` on
    `AgentState.node_latencies_ms`, LangGraph's default channel semantics
    for a plain, un-annotated dict-typed TypedDict field is `LastValue`:
    the SECOND node's `{"node_latencies_ms": {"node_b": ...}}` return would
    silently REPLACE the whole dict, discarding the first node's
    `"node_a"` entry entirely -- exactly the "wrong reducer silently loses
    earlier nodes' timing data" failure mode this task's brief warned
    about. This test builds a minimal throwaway 2-node graph (independent
    of the real classify_intent/retrieve/etc. nodes, to isolate the
    reducer behavior itself from any node-specific logic) and asserts BOTH
    nodes' entries survive in the final merged state -- proving the
    installed `langgraph` package's real behavior, not just documenting an
    assumption about it.
    """
    from langgraph.graph import END, START, StateGraph

    def node_a(state: AgentState) -> dict:
        return {"node_latencies_ms": {"node_a": 11}}

    def node_b(state: AgentState) -> dict:
        return {"node_latencies_ms": {"node_b": 22}}

    builder = StateGraph(AgentState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    result = graph.invoke(
        _state(thread_id="latency-reducer-test"),
        config={"configurable": {"thread_id": "latency-reducer-test"}},
    )

    # If this were `== {"node_b": 22}` instead, the reducer would have
    # failed to accumulate (last-write-wins) -- this is the exact failure
    # mode this test exists to rule out.
    assert result["node_latencies_ms"] == {"node_a": 11, "node_b": 22}
