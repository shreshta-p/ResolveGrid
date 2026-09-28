"""LangGraph agent workflow orchestration for ResolveGrid.

Public entry point: `build_graph(checkpointer, classify_complete_fn,
compose_complete_fn, retrieve_fn)`, which wires up the classify_intent ->
retrieve -> compose_response -> finalize graph (Phase 6 stages
1-2-3-11-12-13; Phase 7 Task 7 adds retrieval between classify_intent and
compose_response). Phase 11 Task 3 splits the single original
`complete_fn` parameter into two -- one per LLM-calling node -- so each
node's real completion can be logged under its own `ModelCall.purpose`
(see `graph.py`'s `build_graph` docstring for the full rationale).

Phase 9 Task 7a adds a second public entry point,
`build_tool_invocation_graph(checkpointer, request_approval_fn,
execute_mutation_fn)`, for the separate `request_approval ->
execute_mutation` graph an explicit tool-invocation endpoint invokes (see
`graph.py`'s "Phase 9 Task 7a" docstring section for why this is a
separate compiled graph rather than new routing in `build_graph`).

Callers (e.g. `apps/api`) should only need `build_graph`/
`build_tool_invocation_graph`, `AgentState`, and the injected-callable
types (`CompleteFn`, `ComposeCompleteFn`, `RetrieveFn`, `RequestApprovalFn`,
`ExecuteMutationFn`) from this package -- `graph.py`'s node
factories/prompts are internal wiring, not part of the intended public
surface. Phase 11 Task 4 adds `ComposeCompleteFn` (`Callable[[str, str],
str]`, prompt + risk_level) specifically for `compose_response`'s real
routing-aware completion function -- see `graph.py`'s `ComposeCompleteFn`
docstring.
"""

from resolvegrid_agent_orchestration.graph import (
    ApprovalOutcome,
    CompleteFn,
    ComposeCompleteFn,
    ExecuteMutationFn,
    RequestApprovalFn,
    RetrievalOutcome,
    RetrieveFn,
    build_graph,
    build_tool_invocation_graph,
)
from resolvegrid_agent_orchestration.state import AgentState, RetrievedChunk

__all__ = [
    "AgentState",
    "ApprovalOutcome",
    "CompleteFn",
    "ComposeCompleteFn",
    "ExecuteMutationFn",
    "RequestApprovalFn",
    "RetrievalOutcome",
    "RetrieveFn",
    "RetrievedChunk",
    "build_graph",
    "build_tool_invocation_graph",
]
