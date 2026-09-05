"""Phase 9 Task 7a: the real `ExecuteMutationFn` implementation injected into
`resolvegrid_agent_orchestration.build_tool_invocation_graph`'s
`execute_mutation` node (`services/agent-orchestration`'s `graph.py`).

Naming/placement note: mirrors `agent_retrieval.py`'s naming convention --
that module is the real `RetrieveFn` implementation for the agent graph;
this module is the real `ExecuteMutationFn` implementation for the same
graph package. It is deliberately its own file, NOT added to
`approval_service.py` (which the Task 7a dispatch prompt offered as one
option): `mutation_execution.py` already does `from resolvegrid_api.
approval_service import compute_snapshot_hash` at module scope, so adding
`import resolvegrid_api.mutation_execution` (needed here to call Task 6's
real `execute_mutation`) into `approval_service.py` would create a circular
import between the two modules. A new, small module avoids that entirely
without restructuring either existing file.

Session lifecycle note: mirrors `approval_service.request_approval_for_agent`'s
"closure built once at app startup (`main.py`'s lifespan), opens its own
short-lived `Session` per call via `resolvegrid_api.db.session_factory()`"
pattern, for the exact same reason -- `AgentState` is checkpointed to
Postgres after every graph superstep, so a live SQLAlchemy `Session` can
never be threaded through state (see agent-orchestration's `state.py`
module docstring), and `execute_mutation_fn` (like `request_approval_fn`)
is expected to be a closure built once, not a per-request dependency.

Commit/rollback ownership: unlike `mutation_execution.execute_mutation`
itself (which deliberately does NOT commit its caller-supplied `Session` --
see that module's docstring for why, a FastAPI approver-decide endpoint
owning a larger transaction), THIS module owns its own short-lived session
end-to-end (opened here, closed here), so it is the right place to actually
commit on success / roll back on a typed failure -- there is no larger
caller-owned transaction for this closure to defer to.

Error translation (the judgment call `graph.py`'s `ExecuteMutationFn`
comment documents): every one of Task 6's typed `MutationExecutionError`
subclasses (`ApprovalNotFoundError`, `ApprovalTamperError` and its
`ApprovalParamsMismatchError` subclass, `ApprovalNotDecidedError`,
`ApprovalExpiredError`, `UnknownMutationToolError`) is caught HERE, in
`apps/api`, and translated into the plain `{"status": "error", "output":
None, "error": "<ErrorClassName>: <message>"}` shape `services/
agent-orchestration`'s `execute_mutation` node expects -- never left to
propagate as a raised exception into that package, which has no import
visibility into `mutation_execution.py`'s error taxonomy at all (see this
codebase's established dependency-direction rule, restated in `graph.py`'s
module docstring). The `<ErrorClassName>` prefix keeps the underlying
error taxonomy code inspectable by whoever reads `tool_invocation_result`
later (e.g. a future Task 7b UI surfacing "why did this fail") without
that caller needing to import `mutation_execution.py` either.
"""

from resolvegrid_api import mutation_execution
from resolvegrid_api.db import session_factory


def execute_mutation_for_agent(payload: dict) -> dict:
    """The real `ExecuteMutationFn` implementation (see
    `resolvegrid_agent_orchestration.graph.ExecuteMutationFn`'s comment for
    the exact payload shape this expects and the return shape this must
    produce).

    Opens its own session, calls Task 6's real `mutation_execution.
    execute_mutation`, commits on success, rolls back and returns a
    structured error result (never raises) on any of Task 6's typed
    `MutationExecutionError` subclasses -- see this module's docstring for
    why that translation happens here rather than in the graph node.

    Only ever called by `execute_mutation`'s graph node when
    `state["approval_decision"] == "approved"` (see that node's docstring)
    -- there is deliberately no read-only-tool code path through this
    function; `execute_readonly_tool` is called directly by `apps/api`'s
    `POST /tools/{tool_name}/invoke` endpoint for non-mutating tools,
    entirely outside this graph (see `routers/tools.py`).
    """
    approval_request_id = payload.get("approval_request_id")
    tool_name = payload.get("tool_name")
    tool_params = payload.get("tool_params") or {}
    actor_employee_id = payload.get("actor_employee_id")

    with session_factory() as session:
        try:
            result = mutation_execution.execute_mutation(
                session,
                approval_request_id=approval_request_id,
                tool_name=tool_name,
                tool_params=tool_params,
                actor_employee_id=actor_employee_id,
            )
        except mutation_execution.MutationExecutionError as exc:
            session.rollback()
            return {
                "status": "error",
                "output": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        session.commit()
        return {
            "status": result["status"],
            "output": result.get("output"),
            "error": None,
        }
