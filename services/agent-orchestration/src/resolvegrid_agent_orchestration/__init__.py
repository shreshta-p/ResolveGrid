"""LangGraph agent workflow orchestration for ResolveGrid.

Public entry point: `build_graph(checkpointer, complete_fn)`, which wires
up the classify_intent -> compose_response -> finalize graph (Phase 6
stages 1-2-3-11-12-13; retrieval/tools/approvals land in later phases).

Callers (e.g. `apps/api`) should only need `build_graph` and `AgentState`
from this package -- `graph.py`'s node factories/prompts are internal
wiring, not part of the intended public surface.
"""

from resolvegrid_agent_orchestration.graph import CompleteFn, build_graph
from resolvegrid_agent_orchestration.state import AgentState

__all__ = ["AgentState", "CompleteFn", "build_graph"]
