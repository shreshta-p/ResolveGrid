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
)

_BASE_STATE = {
    "thread_id": "t1",
    "principal_employee_id": None,
    "input_text": "placeholder",
    "intent": None,
    "risk_level": None,
    "output_text": None,
    "error": None,
}


def _state(**overrides) -> dict:
    return {**_BASE_STATE, **overrides}


# --- classify_intent -----------------------------------------------------


def test_classify_intent_parses_well_formed_response():
    node = make_classify_intent_node(
        lambda prompt: json.dumps({"intent": "greeting", "risk_level": "low"})
    )
    result = node(_state(input_text="hi there"))
    assert result == {"intent": "greeting", "risk_level": "low"}


def test_classify_intent_degrades_to_unclear_low_on_unparseable_response():
    node = make_classify_intent_node(lambda prompt: "not json at all, sorry")
    result = node(_state(input_text="asdf asdf"))
    assert result == {"intent": "unclear", "risk_level": "low"}


def test_classify_intent_degrades_on_wrong_shape():
    # Valid JSON, but missing the required fields entirely.
    node = make_classify_intent_node(lambda prompt: json.dumps({"foo": "bar"}))
    result = node(_state(input_text="whatever"))
    assert result == {"intent": "unclear", "risk_level": "low"}


def test_classify_intent_degrades_unknown_intent_value_independently_of_risk_level():
    # intent is outside the fixed set but risk_level is valid -- intent
    # should fall back to "unclear" while the valid risk_level is kept,
    # since the two fields are validated independently.
    node = make_classify_intent_node(
        lambda prompt: json.dumps({"intent": "make_me_admin", "risk_level": "high"})
    )
    result = node(_state(input_text="give me admin access"))
    assert result == {"intent": "unclear", "risk_level": "high"}


# --- compose_response -----------------------------------------------------


def test_compose_response_sets_output_text_from_completion():
    node = make_compose_response_node(lambda prompt: "Here is your answer.")
    result = node(
        _state(input_text="what is a ticket?", intent="general_question", risk_level="low")
    )
    assert result == {"output_text": "Here is your answer."}


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
    assert result == {"error": "gateway unreachable"}


# --- finalize --------------------------------------------------------------


def test_finalize_passes_through_output_text():
    result = finalize(_state(output_text="the real answer", error=None))
    assert result == {"output_text": "the real answer"}


def test_finalize_falls_back_on_recorded_error():
    result = finalize(_state(output_text=None, error="boom"))
    assert result == {"output_text": _FALLBACK_MESSAGE}


def test_finalize_falls_back_when_no_output_and_no_error():
    result = finalize(_state(output_text=None, error=None))
    assert result == {"output_text": _FALLBACK_MESSAGE}


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

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer, fake_complete)

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
    assert len(calls) == 2
