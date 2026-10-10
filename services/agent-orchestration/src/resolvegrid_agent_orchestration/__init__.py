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
surface. Phase 11 Task 4 adds `ComposeCompleteFn` (a `Protocol` taking `prompt` and a
keyword-only `risk_level`) specifically for `compose_response`'s real
routing-aware completion function -- see `graph.py`'s `ComposeCompleteFn`
docstring. Phase 12 Task 3 adds one more: `PROMPT_TEMPLATE_VERSION`, a
plain string `apps/api`'s new `/chat` response cache folds into its cache
key so a future compose-prompt wording change organically busts every
previously-cached answer -- see `graph.py`'s own comment beside that
constant for the full rationale. Phase 12 Task 4 widens `ComposeCompleteFn`
by one more keyword-only, defaulted parameter, `cacheable_prefix` -- the
real stable instructional preamble of the with-context compose prompt,
handed over separately so `apps/api`'s real implementation can mark it with
Anthropic's provider-side `cache_control` annotation (never a Redis
application cache the way this phase's other caches are) -- see `graph.py`'s
`ComposeCompleteFn` docstring for the full rationale.
"""

from resolvegrid_agent_orchestration.graph import (
    PROMPT_TEMPLATE_VERSION,
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
    "PROMPT_TEMPLATE_VERSION",
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
