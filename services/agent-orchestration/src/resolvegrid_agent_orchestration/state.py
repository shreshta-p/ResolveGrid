"""Graph state shape for the agent workflow.

Phase 6 shipped this with no retrieval-related fields ("adding them now
would be dead, untested surface area"). Phase 7 Task 7 adds them now that
a real knowledge base + retrieval pipeline exists (`apps/api`'s
`retrieval.py`/`retrieval_authz.py`).

Design note (read before touching retrieval wiring elsewhere): this
package must never import `resolvegrid_api`/SQLAlchemy (see graph.py's
module docstring for the full dependency-direction reasoning). That rules
out putting apps/api's real `AuthzFilter` dataclass or a `Chunk` ORM
object directly into this state -- `retrieval_scope` and `retrieved_chunks`
below are deliberately plain, JSON-shaped `dict`/`list` structures instead.

This isn't just a style preference: `AgentState` is checkpointed to
Postgres by `AsyncPostgresSaver` after every graph superstep. A live
SQLAlchemy `Session` (or anything holding one) must never end up in this
state -- it would not serialize safely, and even if it somehow pickled, a
live DB connection has no business being persisted as conversation state.
That's exactly why `RetrieveFn`'s actual database access is NOT threaded
through `AgentState` at all: `retrieve_fn` (see graph.py) is a closure
built once at app startup (mirroring `complete_fn`'s pattern) that opens
its own short-lived `Session` per call -- state only ever carries the
opaque, pre-built authz predicate (`retrieval_scope`) and the *results* of
retrieval (`retrieved_chunks`), both plain data.
"""

from typing import TypedDict


class RetrievedChunk(TypedDict):
    """One retrieved chunk, already resolved to citation-ready detail by
    the `retrieve` node -- see graph.py's `RetrieveFn`/`make_retrieve_node`.

    `chunk_id` is the real `resolvegrid_api.models.knowledge.Chunk.id`
    (this package has no `Chunk` model of its own -- it's just an int
    here), which `apps/api`'s `/chat` endpoint can resolve back to a real
    citation for the UI. `score` is the fused RRF score from
    `resolvegrid_api.retrieval.fuse_rrf`, carried through for
    debugging/telemetry -- not itself interpreted by this package.
    """

    chunk_id: int
    document_title: str
    text: str
    score: float


class AgentState(TypedDict):
    thread_id: str
    principal_employee_id: int | None
    input_text: str
    intent: str | None
    risk_level: str | None
    # Opaque, apps/api-built authz predicate consumed by the `retrieve`
    # node's `RetrieveFn` call -- a plain dict (e.g. {"unrestricted": bool,
    # "allowed_tags": [...]})  that this package never inspects beyond
    # passing it straight through to `retrieve_fn`. See this module's
    # docstring for why apps/api's real `AuthzFilter` dataclass is
    # translated to this shape *before* entering the graph, rather than
    # crossing the package boundary as a typed object.
    retrieval_scope: dict | None
    # Populated by the `retrieve` node; consumed by `compose_response` to
    # build citation-ready prompt context. `None` before `retrieve` runs;
    # an empty list after it runs if nothing was found/authorized.
    retrieved_chunks: list[RetrievedChunk] | None
    # The deterministic sufficiency verdict from
    # `resolvegrid_api.retrieval.assess_sufficiency`, threaded through
    # verbatim by the `retrieve` node. `compose_response` uses this (not
    # just "chunks non-empty") to decide whether to answer in
    # citation-grounded mode or fall back to general-knowledge mode -- see
    # graph.py's module docstring for the documented scope limit on how
    # far this drives abstention routing.
    retrieval_sufficient: bool | None
    # Phase 8 Task 7: the pre-assembled, token-budgeted context block text
    # (`resolvegrid_retrieval.context_budget.assemble_context(...).text`),
    # already reranked/status-adjusted/deduped/budgeted by `retrieve_fn`
    # (see `apps/api`'s `agent_retrieval.py` -- that's where all of this
    # runs, since it needs DB access `services/retrieval` doesn't have;
    # see graph.py's module docstring). `compose_response` substitutes
    # this directly into its prompt template's untrusted-data-delimited
    # slot instead of building its own unbounded join from
    # `retrieved_chunks` -- see graph.py's `_COMPOSE_PROMPT_WITH_CONTEXT_
    # TEMPLATE`. `retrieved_chunks` above is exactly the set of chunks
    # whose formatted entry appears in this text (`ContextBlock.chunk_ids`,
    # not the larger pre-budget candidate pool) -- the two must stay in
    # lockstep so citation verification's `valid_chunk_ids` (derived from
    # `retrieved_chunks`) matches what the model actually saw in `text`.
    context_block: str | None
    output_text: str | None
    error: str | None
    # Phase 8 Task 7: the deterministic citation-verification outcome for
    # `output_text`, populated by the `verify_citations` node (runs after
    # `compose_response`, before `finalize`). `citations_verified` is
    # `True` iff every `[chunk:<id>]` citation `output_text` contained (at
    # the time verification ran) resolved to a chunk id actually present
    # in `context_block`/`retrieved_chunks` -- vacuously `True` when the
    # answer cited nothing at all. `output_text` is REWRITTEN by that node
    # to strip any fabricated citation markers before `finalize` ever sees
    # it (see graph.py's `verify_citations_node` docstring for the exact
    # graph-level consequence chosen) -- so by the time a caller reads
    # `state["output_text"]` after a full graph run, it is already
    # citation-safe. `verified_chunk_ids`/`fabricated_chunk_ids` are the
    # unique ids in first-appearance order (mirrors `VerificationResult`'s
    # own shape) -- `apps/api`'s `/chat` uses `verified_chunk_ids` to build
    # the citation list it actually surfaces to the UI (only citations the
    # model actually made AND that verified survive), not the full
    # pre-citation `retrieved_chunks` list.
    citations_verified: bool | None
    verified_chunk_ids: list[int] | None
    fabricated_chunk_ids: list[int] | None
    # Phase 9 Task 5: the tool call a not-yet-built upstream node (Task 6's
    # select_tool/routing work) is expected to propose for approval. This
    # task's `request_approval` node (graph.py, currently standalone -- see
    # its module docstring's documented scope limit) reads these but does
    # NOT set them -- `None` unless some prior node in a given run has
    # populated them, which today only a test harness seeding initial state
    # directly does. `proposed_tool_name` is the `resolvegrid_contracts.
    # tools.ToolContract.name` of the tool being proposed (e.g.
    # "grant_vpn_access").
    proposed_tool_name: str | None
    # The concrete call arguments paired with `proposed_tool_name` above --
    # becomes `ApprovalRequest.action_params_json` (JSON-encoded with sorted
    # keys) via `request_approval_fn`'s real implementation (`apps/api`'s
    # `approval_service.py`); also one of the fields bound into the
    # snapshot hash that request re-verifies at execution time (Task 6).
    # Whatever node sets this (Task 6+) must keep it plain, JSON-safe data
    # (str/int/float/bool/None/list/dict only) -- like every other
    # `AgentState` field it is checkpointed to Postgres after every graph
    # superstep (see this module's docstring), and it is separately
    # `json.dumps(..., sort_keys=True)`-encoded by `approval_service.py`
    # for both `action_params_json` and the snapshot hash -- a
    # non-JSON-safe value (e.g. a raw `datetime`) would fail there.
    proposed_tool_params: dict | None
    # Populated by the `request_approval` node from its injected
    # `RequestApprovalFn`'s return value -- the real, durable
    # `ApprovalRequest.id` row this run's approval lives at (created, or
    # found via the idempotent upsert if this node re-executed after a
    # checkpoint restore -- see graph.py's `RequestApprovalFn`/
    # `make_request_approval_node` docstrings for why plain snapshot-hash
    # equality alone is not what makes that upsert idempotent). `None`
    # before `request_approval` runs, or on any run that never proposes a
    # tool requiring approval.
    approval_request_id: int | None
    # Populated by the `request_approval` node from the value LangGraph's
    # `interrupt()` returns on resume -- i.e. whatever an approver's
    # decision was passed as `Command(resume=<value>)`'s argument.
    # "approved" | "rejected" | `None`. Note `None` here does NOT mean "a
    # decision is pending" -- a genuinely pending decision means the graph
    # run itself is paused mid-`interrupt()` and `ainvoke()` has not
    # returned at all (observable via the compiled graph's own
    # `get_state(config).next`, not this field); by the time any caller
    # reads `state["approval_decision"]` from a *completed* `ainvoke()`
    # result, it is either a real decision string or this run never reached
    # `request_approval` in the first place.
    approval_decision: str | None
