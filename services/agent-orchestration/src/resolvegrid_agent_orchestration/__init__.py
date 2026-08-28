"""LangGraph agent workflow orchestration for ResolveGrid.

Public entry point: `build_graph(checkpointer, complete_fn, retrieve_fn)`,
which wires up the classify_intent -> retrieve -> compose_response ->
finalize graph (Phase 6 stages 1-2-3-11-12-13; Phase 7 Task 7 adds
retrieval between classify_intent and compose_response; tools/approvals
land in later phases).

Callers (e.g. `apps/api`) should only need `build_graph`, `AgentState`,
and the two injected-callable types (`CompleteFn`, `RetrieveFn`) from this
package -- `graph.py`'s node factories/prompts are internal wiring, not
part of the intended public surface.
"""

from resolvegrid_agent_orchestration.graph import CompleteFn, RetrieveFn, RetrievalOutcome, build_graph
from resolvegrid_agent_orchestration.state import AgentState, RetrievedChunk

__all__ = [
    "AgentState",
    "CompleteFn",
    "RetrievalOutcome",
    "RetrieveFn",
    "RetrievedChunk",
    "build_graph",
]
