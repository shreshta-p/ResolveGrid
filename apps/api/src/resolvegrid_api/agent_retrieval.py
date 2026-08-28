"""Retrieval glue for the agent graph (Phase 7 Task 7): the concrete
`RetrieveFn` implementation injected into
`resolvegrid_agent_orchestration.build_graph`, wired up once in
`main.py`'s lifespan -- mirrors `complete_fn`'s "built once, closure over
apps/api internals" pattern (see
`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`'s
module docstring for that established convention).

Session lifecycle note: `AgentState` is checkpointed to Postgres after
every graph superstep (`AsyncPostgresSaver`), so a live SQLAlchemy
`Session` can never be threaded through state (see agent-orchestration's
`state.py` module docstring). `retrieve_fn` is built ONCE at app startup
(same as `complete_fn`), not per-request, so it also cannot close over a
request-scoped `Depends(get_db)` session. Instead this module opens its
own short-lived `Session` per call via `resolvegrid_api.db.session_factory()`
-- the same underlying idea `resolvegrid_api.ingestion_worker`'s Arq job
functions use for opening DB access outside FastAPI's request scope,
except this reuses the shared pooled engine (`session_factory()`) rather
than creating/disposing a brand new `Engine` every call, since `/chat` is
a hot path and ingestion is not.

Authz note: `retrieve_fn`'s second argument (`retrieval_scope`) is the
plain-dict shape `chat.py` builds from a real `AuthzFilter`
(`resolvegrid_api.retrieval_authz.build_authz_filter`) *before* the graph
is invoked -- see agent-orchestration's `state.py` module docstring for
why it crosses the package boundary as an opaque dict rather than the
typed `AuthzFilter` dataclass. `_authz_filter_from_scope` below
reconstructs the real `AuthzFilter` on this side of the boundary.
"""

from sqlalchemy import select

from resolvegrid_api.db import session_factory
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion
from resolvegrid_api.retrieval import assess_sufficiency, fuse_rrf, lexical_search, vector_search
from resolvegrid_api.retrieval_authz import AuthzFilter
from resolvegrid_retrieval.embedder import EmbeddingError, embed_texts

# Candidate pool size per signal (vector/lexical), pre-fusion -- matches
# `apps/api/tests/test_retrieval.py`'s own `limit=10` convention for
# exercising RRF with real overlap/depth to fuse over.
_SEARCH_LIMIT = 10

# How many fused chunks are actually attached to state / passed into
# compose_response's prompt. Smaller than `_SEARCH_LIMIT` on purpose:
# every chunk here goes verbatim into the LLM prompt as citation context,
# so this is a prompt-budget bound, not a relevance bound -- RRF has
# already ranked the full candidate pool; this just takes its top N.
_TOP_K_FOR_COMPOSE = 5


def _authz_filter_from_scope(scope: dict | None) -> AuthzFilter:
    """Reconstruct a real `AuthzFilter` from the opaque dict
    `chat.py` put on `state["retrieval_scope"]`.

    A missing/empty `scope` deliberately does NOT default to unrestricted
    -- an absent scope means "we don't know who's asking" (e.g. a caller
    that forgot to build one), which must fail closed (see all-empty
    `allowed_tags`), not silently show every document.
    """
    if not scope:
        return AuthzFilter(unrestricted=False, allowed_tags=frozenset())
    return AuthzFilter(
        unrestricted=bool(scope.get("unrestricted")),
        allowed_tags=frozenset(scope.get("allowed_tags") or []),
    )


def retrieve_for_agent(query_text: str, scope: dict | None) -> dict:
    """The `RetrieveFn` implementation: hybrid retrieval (vector + lexical
    + RRF fusion), restricted by `scope`'s authz predicate, resolved to
    citation-ready chunk detail (document title + chunk text), plus the
    deterministic sufficiency verdict.

    Returns `{"chunks": [...], "sufficient": bool}` -- the exact shape
    `resolvegrid_agent_orchestration.graph.RetrievalOutcome` expects.

    Degrades to lexical-only search (does not raise) if the embedding
    service is unavailable/the model isn't pulled (`EmbeddingError`) -- a
    partial result is more useful than none, and `assess_sufficiency`
    still applies the same bar to whatever comes back. Other failures (DB
    unreachable, etc.) propagate; the graph's `retrieve` node is what
    degrades softly on those (see its docstring).
    """
    authz_filter = _authz_filter_from_scope(scope)

    with session_factory() as session:
        try:
            query_vector = embed_texts([query_text])[0]
        except EmbeddingError:
            query_vector = None

        vector_results = (
            vector_search(session, query_vector, limit=_SEARCH_LIMIT, authz_filter=authz_filter)
            if query_vector is not None
            else []
        )
        lexical_results = lexical_search(
            session, query_text, limit=_SEARCH_LIMIT, authz_filter=authz_filter
        )
        fused = fuse_rrf(vector_results, lexical_results)
        sufficiency = assess_sufficiency(fused)

        top = fused[:_TOP_K_FOR_COMPOSE]
        chunk_ids = [chunk_id for chunk_id, _ in top]
        score_by_chunk_id = dict(top)

        rows = (
            session.execute(
                select(Chunk.id, Chunk.text, Document.title)
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(Chunk.id.in_(chunk_ids))
            ).all()
            if chunk_ids
            else []
        )
        row_by_id = {row.id: row for row in rows}

        chunks = [
            {
                "chunk_id": chunk_id,
                "document_title": row_by_id[chunk_id].title,
                "text": row_by_id[chunk_id].text,
                "score": score_by_chunk_id[chunk_id],
            }
            for chunk_id in chunk_ids
            if chunk_id in row_by_id
        ]

        return {"chunks": chunks, "sufficient": sufficiency.sufficient}
