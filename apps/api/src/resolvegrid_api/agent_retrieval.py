"""Retrieval glue for the agent graph (Phase 7 Task 7, extended by Phase 8
Task 7): the concrete `RetrieveFn` implementation injected into
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

Authz-then-rerank ordering, preserved (Phase 8 Task 7 Part D): `vector_
search`/`lexical_search` apply `authz_filter` INSIDE their SQL `WHERE`
clause (see `retrieval.py`'s module docstring) -- an unauthorized chunk
never becomes a Python row at all. Everything Task 7 adds below (rerank,
status-adjustment, dedup, budgeting) only ever reorders/filters the
already-authorized `fused` candidate set; none of it can add a chunk id
that wasn't already in `fused`, so it is architecturally impossible for
this pipeline to reintroduce a chunk the authz filter excluded. This
mirrors Phase 7 Task 5's established "filtered at the DB query, not
post-hoc in Python" guarantee.

Phase 8 Task 7 -- full pipeline (Parts B and C)
--------------------------------------------------------------------------
`retrieve_for_agent` now runs the complete rerank -> status-adjust ->
dedup -> token-budget pipeline (previously four separate, unwired
`services/retrieval` modules -- Tasks 1-3 plus this task's new
`status_adjustment.py`) before returning a `RetrievalOutcome`, in this
order:

  1. `fuse_rrf`'s full fused candidate list (NOT pre-truncated to a small
     top-N before reranking -- matches `eval_retrieval.py`'s own
     precedent of giving `rerank()` "the same full candidate pool a real
     pipeline would give it"; the old `_TOP_K_FOR_COMPOSE = 5` truncation
     that used to happen *before* any reranking/budgeting existed is
     removed, since `assemble_context`'s token budget is now the real,
     principled limiting factor instead of an arbitrary fixed top-N).
  2. One DB query resolves chunk text, document title, AND `Document.
     status` for every fused candidate id in one round trip (extends the
     query that already existed for title -- no new query shape, just
     one more selected column).
  3. `rerank()` (Task 1) against the real chunk texts. Degrades to the
     unreranked fused order (not to an empty result) if `RerankError` is
     raised -- see `_rerank_or_degrade`'s docstring for why: a broken/
     missing reranker dependency should not be able to zero out
     perfectly good fused retrieval results, mirroring this module's
     existing `EmbeddingError` degrade-to-lexical-only precedent below.
  4. `apply_status_adjustment` (Task 7 Part B, `resolvegrid_retrieval.
     status_adjustment`) deprioritizes any candidate whose `Document.
     status == "superseded"` -- see that module's docstring for the full
     mitigation rationale (the VPN v1/v2 distractor-flip regression
     `docs/EXPERIMENT_REGISTRY.md`'s Phase 8 Task 6 entry documented).
  5. `dedup()` (Task 2) removes near-duplicate survivors.
  6. `assemble_context` (Task 3) greedily fills the token budget from the
     deduped, best-first list, producing the final `ContextBlock`.

  The `RetrievalOutcome.chunks` returned is exactly `ContextBlock.
  chunk_ids`' corresponding chunk detail -- i.e. exactly the chunks whose
  formatted entry is present in `context_block`'s text, not the larger
  pre-budget candidate pool. This is required for citation verification
  (`verify_citations_node`, graph.py) to check citations against "what the
  model was actually shown," not an outdated larger set.
"""

from sqlalchemy import select

from resolvegrid_api.db import session_factory
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion
from resolvegrid_api.retrieval import assess_sufficiency, fuse_rrf, lexical_search, vector_search
from resolvegrid_api.retrieval_authz import AuthzFilter
from resolvegrid_retrieval.context_budget import assemble_context
from resolvegrid_retrieval.dedup import dedup
from resolvegrid_retrieval.embedder import EmbeddingError, embed_texts
from resolvegrid_retrieval.reranker import RerankError, rerank
from resolvegrid_retrieval.status_adjustment import apply_status_adjustment

# Candidate pool size per signal (vector/lexical), pre-fusion -- matches
# `apps/api/tests/test_retrieval.py`'s own `limit=10` convention for
# exercising RRF with real overlap/depth to fuse over, and
# `eval_retrieval.py`'s `DEFAULT_SEARCH_LIMIT`.
_SEARCH_LIMIT = 10


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


def _rerank_or_degrade(query_text: str, candidates: list[tuple[int, str]]) -> list[tuple[int, str, float]]:
    """`rerank()` (Task 1), degrading to the unreranked fused order (text
    re-attached, RRF rank position turned into a descending pseudo-score
    so the rest of the pipeline's "higher score wins" convention still
    holds) if `RerankError` is raised -- mirrors this module's existing
    `EmbeddingError` degrade-to-lexical-only precedent: a broken/missing
    optional reranker dependency must not be able to zero out perfectly
    good fused retrieval results.
    """
    if not candidates:
        return []
    try:
        reranked = rerank(query_text, candidates)
    except RerankError:
        text_by_id = dict(candidates)
        return [
            (chunk_id, text_by_id[chunk_id], float(len(candidates) - position))
            for position, (chunk_id, _text) in enumerate(candidates)
        ]
    text_by_id = dict(candidates)
    return [(chunk_id, text_by_id[chunk_id], score) for chunk_id, score in reranked]


def retrieve_for_agent(query_text: str, scope: dict | None) -> dict:
    """The `RetrieveFn` implementation: hybrid retrieval (vector + lexical
    + RRF fusion, authz-filtered at the SQL layer) -> rerank -> Document.
    status-aware deprioritization -> dedup -> token-budgeted context
    assembly. See this module's docstring for the full pipeline design
    and ordering rationale.

    Returns `{"chunks": [...], "sufficient": bool, "context_block": str}`
    -- the exact shape `resolvegrid_agent_orchestration.graph.
    RetrievalOutcome` expects. `chunks` is exactly the survivor set
    `context_block`'s text was built from (`ContextBlock.chunk_ids`), not
    the larger pre-budget fused/reranked candidate pool -- required so
    citation verification checks "what the model was actually shown."

    Degrades to lexical-only search (does not raise) if the embedding
    service is unavailable/the model isn't pulled (`EmbeddingError`) -- a
    partial result is more useful than none, and `assess_sufficiency`
    still applies the same bar to whatever comes back. Reranking degrades
    to the fused order, not an empty result, if the optional reranker
    dependency is unavailable (see `_rerank_or_degrade`). Other failures
    (DB unreachable, etc.) propagate; the graph's `retrieve` node is what
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

        fused_ids = [chunk_id for chunk_id, _score in fused]

        rows = (
            session.execute(
                select(Chunk.id, Chunk.text, Document.title, Document.status)
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(Chunk.id.in_(fused_ids))
            ).all()
            if fused_ids
            else []
        )
        row_by_id = {row.id: row for row in rows}

        # Only candidates that actually resolved to a real row survive
        # (matches the old code's "if chunk_id in row_by_id" guard) --
        # rerank() below is given exactly this authorized, resolvable set,
        # never the raw fused id list.
        candidates = [
            (chunk_id, row_by_id[chunk_id].text) for chunk_id in fused_ids if chunk_id in row_by_id
        ]
        superseded_chunk_ids = frozenset(
            chunk_id for chunk_id in fused_ids if chunk_id in row_by_id and row_by_id[chunk_id].status == "superseded"
        )

        reranked = _rerank_or_degrade(query_text, candidates)
        adjusted = apply_status_adjustment(reranked, superseded_chunk_ids=superseded_chunk_ids)
        deduped = dedup(adjusted)

        assembly_input = [
            (chunk_id, row_by_id[chunk_id].title, text) for chunk_id, text, _score in deduped
        ]
        context = assemble_context(assembly_input)

        score_by_id = {chunk_id: score for chunk_id, _text, score in deduped}
        chunks = [
            {
                "chunk_id": chunk_id,
                "document_title": row_by_id[chunk_id].title,
                "text": row_by_id[chunk_id].text,
                "score": score_by_id[chunk_id],
            }
            for chunk_id in context.chunk_ids
        ]

        return {
            "chunks": chunks,
            "sufficient": sufficiency.sufficient,
            "context_block": context.text,
        }
