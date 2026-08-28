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
    output_text: str | None
    error: str | None
