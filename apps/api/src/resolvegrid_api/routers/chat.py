import time
from datetime import datetime, timezone
from uuid import uuid4

import redis
from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry import trace
from pydantic import BaseModel
from resolvegrid_agent_orchestration import PROMPT_TEMPLATE_VERSION
from sqlalchemy.orm import Session

from resolvegrid_api.cache import get_redis
from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.model_call_logging import log_cache_hit_model_call
from resolvegrid_api.models import AgentRun, Span
from resolvegrid_api.response_cache import lookup_cached_response, store_cached_response
from resolvegrid_api.retrieval_authz import build_authz_filter
from resolvegrid_api.retrieval_cache import compute_scope_sig, get_kb_version
from resolvegrid_authz import Principal

router = APIRouter(prefix="/chat", tags=["chat"])

# The global tracer provider is already configured once at app startup by
# main.py's lifespan hook (resolvegrid_telemetry.init_tracing) -- this module
# must NOT call init_tracing again, just bind a tracer to whatever provider is
# globally registered by the time a span is actually started (same pattern
# as tickets.py).
tracer = trace.get_tracer(__name__)

# The graph's 4 conceptual stages, in execution order -- used to write one
# DB `Span` row per stage. `retrieve` (Phase 7 Task 7) was inserted between
# `classify_intent` and `compose_response`, matching the real graph wiring
# in `services/agent-orchestration/.../graph.py`'s `build_graph`. See
# chat()'s docstring for why this is still 4 names (not 5, matching
# `build_graph`'s real node count) and for why their latency_ms values are
# now genuinely measured per-node, not a split placeholder.
_STAGE_NAMES = ("classify_intent", "retrieve", "compose_response", "finalize")

# Shown when retrieval ran but found nothing sufficient to cite -- a real
# knowledge base exists (Phase 7), so this is no longer "no KB yet" (that
# caption was accurate through Phase 6; it would now be a false statement).
_NO_KB_MATCH_CAPTION = (
    "General-knowledge answer — no matching company knowledge-base article "
    "was found for this question."
)


class ChatRequest(BaseModel):
    message: str


@router.post("")
async def chat(
    payload: ChatRequest,
    request: Request,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    redis_client: redis.Redis = Depends(get_redis),
) -> dict:
    """Single-turn chat: run the classify_intent -> retrieve -> compose_response
    -> finalize graph (via `request.app.state.agent_graph`, wired up in
    main.py's lifespan with AsyncPostgresSaver checkpointing) for one fresh
    thread_id.

    Phase 6 doesn't expose multi-turn conversation resumption to the user
    yet -- every call here starts a brand new thread_id, so the checkpointer
    persists state for potential future resumption/audit, but this endpoint
    itself is stateless across calls (a new conversation every time).

    This handler is `async def` (a deliberate, justified exception to this
    codebase's otherwise-universal "plain `def`, let FastAPI's threadpool
    handle blocking I/O" convention): `AsyncPostgresSaver`-backed graphs must
    be invoked via `ainvoke()` inside a running event loop -- there is no
    synchronous equivalent being used here, so this is architecturally
    correct, not an inconsistency with the rest of this router's modules.

    Authz note (Phase 7 Task 7): `build_authz_filter(principal, session)` is
    resolved here, per-request, using this handler's own `Depends(get_db)`
    session -- then translated into the plain-dict `retrieval_scope` that
    crosses into the graph's `AgentState` (see agent-orchestration's
    `state.py` module docstring for why it's a dict, not the typed
    `AuthzFilter`). This is genuinely per-request (unlike `complete_fn`,
    which has no per-request auth concept at all), because a different
    caller can have a different authorized scope for the same graph.

    Span/timing note (Phase 11 Task 2 -- real per-node timing, replacing the
    old placeholder): each node in `services/agent-orchestration/.../
    graph.py`'s `build_graph` chain now wraps its own real work in
    `time.monotonic()` and appends `{node_name: elapsed_ms}` into
    `AgentState.node_latencies_ms` as part of its own returned partial-state
    update (see that field's docstring in `state.py` for why an `Annotated`
    custom reducer is required for these per-node entries to accumulate
    across the run instead of each node's return overwriting the whole
    dict -- verified against the installed `langgraph` package with a real
    test, not assumed). This handler reads `final_state["node_latencies_ms"]`
    after `ainvoke()` returns and writes those REAL measured values as each
    stage's `Span.latency_ms` below -- no evenly-split placeholder remains.

    `graph.py`'s real chain has 5 nodes end to end (`classify_intent` ->
    `retrieve` -> `compose_response` -> `verify_citations` -> `finalize`),
    but this table's established shape is 4 `Span` rows per run, one per
    name in `_STAGE_NAMES` -- a shape an existing test
    (`test_chat_success_writes_agent_run_and_four_success_spans` in
    `apps/api/tests/test_chat_api.py`) asserts exactly, predating
    `verify_citations` ever being its own separately-instrumented node.
    Rather than adding a 5th row and breaking that established shape, this
    handler folds `verify_citations`'s real measured time into the
    `"finalize"` row (`node_latencies_ms["verify_citations"] +
    node_latencies_ms["finalize"]`) -- both nodes are cheap, non-LLM,
    in-process work (deterministic citation-marker stripping and a few dict
    lookups, respectively), so grouping them under one conceptual
    "finalize" stage does not hide any meaningfully-sized real cost the way
    folding, say, a real LLM call into another stage would. This is a
    deliberate, documented choice, not an oversight -- every real
    millisecond `graph.py` measures is still accounted for somewhere in the
    4 rows below, none of it silently dropped.

    Also note: `build_graph` has no conditional routing at all (every edge
    is a plain `add_edge`, verified by reading it) -- all 5 nodes execute on
    every real chat turn, so there is currently no "node skipped this run"
    case to omit a `Span` row for; if a future phase adds real conditional
    routing, whoever does that should revisit whether every `_STAGE_NAMES`
    entry can still be assumed present in `node_latencies_ms`.

    Exact-response cache note (Phase 12 Task 3, `resolvegrid_api.
    response_cache`): right after `authz_filter`/`scope_sig`/`kb_version`
    are resolved below, this handler performs a real cache lookup keyed on
    `(kb_version, scope_sig, PROMPT_TEMPLATE_VERSION, payload.message)`.
    On a real hit, `request.app.state.agent_graph.ainvoke(...)` is NEVER
    called -- the entire graph run (classify_intent/retrieve/
    compose_response/verify_citations/finalize) is skipped -- and this
    handler instead writes a real `AgentRun` row (`status="completed"`,
    `cache_hit=True`, real near-zero latency for the cache lookup itself)
    plus the same real 4-row `_STAGE_NAMES` `Span` set and a real `$0`
    `ModelCall` row (`purpose="chat.cache_hit"`), then returns the
    identical response SHAPE (`answer`/`thread_id`/`citations`/`caption`)
    the real graph path returns -- `thread_id` is still a fresh
    `uuid4().hex` per call even on a hit, since it identifies THIS call's
    `AgentRun`, not the cached content. On a miss (including a real Redis
    outage, which degrades to a miss rather than ever failing the request),
    this handler falls through to the exact pre-existing real-graph flow
    below, then stores the freshly computed response into the cache right
    before returning it.
    """
    thread_id = uuid4().hex

    agent_run = AgentRun(
        status="running",
        thread_id=thread_id,
        principal_employee_id=principal.employee_id,
        input_text=payload.message,
    )
    session.add(agent_run)
    session.commit()

    authz_filter = build_authz_filter(principal, session)
    retrieval_scope = {
        "unrestricted": authz_filter.unrestricted,
        "allowed_tags": sorted(authz_filter.allowed_tags),
    }

    # Real cache lookup, per this task's own corrected key shape (see
    # response_cache.py's module docstring and docs/DECISION_LOG.md's
    # 2026-09-30 entry): `scope_sig`/`kb_version` are the SAME real,
    # already-established functions Task 2's retrieval cache uses (reused
    # directly, never reimplemented), computed from THIS request's own
    # resolved `authz_filter`/`session` -- never a raw employee id alone.
    scope_sig = compute_scope_sig(authz_filter)
    kb_version = get_kb_version(session)

    with tracer.start_as_current_span("chat.cache_lookup") as cache_span:
        cache_span.set_attribute("resolvegrid.thread_id", thread_id)
        cache_lookup_start = time.monotonic()
        cached_response = lookup_cached_response(
            message_text=payload.message,
            kb_version=kb_version,
            scope_sig=scope_sig,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            redis_client=redis_client,
        )
        cache_lookup_latency_ms = int((time.monotonic() - cache_lookup_start) * 1000)
        cache_span.set_attribute("resolvegrid.cache_hit", cached_response is not None)

    if cached_response is not None:
        # CACHE HIT: the real graph never runs. Every field below is a
        # genuine, honestly-shaped row -- not a fabricated placeholder --
        # per this task's own requirement that a cache hit still produce a
        # real, queryable AgentRun/Span/ModelCall row set.
        agent_run.status = "completed"
        agent_run.cache_hit = True
        agent_run.output_text = cached_response.get("answer") or ""
        agent_run.completed_at = datetime.now(timezone.utc)

        # The real 4-row `_STAGE_NAMES` shape is preserved even though no
        # graph node actually ran: the one real cost this call incurred --
        # the cache lookup itself -- is attributed in full to the first
        # stage (the conceptual point where real work would otherwise have
        # begun); the other three stages honestly record 0ms, since they
        # genuinely did not execute (0 is the true measured value for
        # "this stage did not run", not a fabricated placeholder the way
        # an evenly-split total would be).
        for index, stage_name in enumerate(_STAGE_NAMES):
            session.add(
                Span(
                    agent_run_id=agent_run.id,
                    stage_name=stage_name,
                    status="success",
                    latency_ms=cache_lookup_latency_ms if index == 0 else 0,
                )
            )

        # Real $0 ModelCall row so cost/telemetry dashboards don't silently
        # under-count this real traffic (see model_call_logging.py's
        # log_cache_hit_model_call docstring for why `model`/`provider` are
        # derived from the cached risk_level via the same real routing
        # policy a genuine call would have used).
        log_cache_hit_model_call(
            session,
            tracer,
            purpose="chat.cache_hit",
            risk_level=cached_response.get("risk_level") or "low",
        )
        session.commit()

        return {
            "answer": agent_run.output_text,
            "thread_id": thread_id,
            "citations": cached_response.get("citations") or [],
            "caption": cached_response.get("caption") or "",
        }

    initial_state = {
        "thread_id": thread_id,
        "principal_employee_id": principal.employee_id,
        "input_text": payload.message,
        "intent": None,
        "risk_level": None,
        "retrieval_scope": retrieval_scope,
        "retrieved_chunks": None,
        "retrieval_sufficient": None,
        "context_block": None,
        "output_text": None,
        "error": None,
        "citations_verified": None,
        "verified_chunk_ids": None,
        "fabricated_chunk_ids": None,
        "node_latencies_ms": {},
    }

    with tracer.start_as_current_span("chat.graph_run") as span:
        span.set_attribute("resolvegrid.thread_id", thread_id)
        start = time.monotonic()
        try:
            final_state = await request.app.state.agent_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            # Broader than summarize_ticket's `except llm_gateway.LLMGatewayError`:
            # a graph run can fail for reasons beyond the LLM gateway itself
            # (checkpointer/DB errors, a node raising for any other reason,
            # etc.), and this endpoint's contract is "the agent run failed,
            # cleanly" regardless of which layer raised -- narrowing this to
            # one exception type would leave other real failure modes
            # uncaught and turn into unhandled 500s instead of the same
            # clean 502 contract callers already get for LLM-gateway errors.
            span.set_attribute("error.type", type(exc).__name__)
            agent_run.status = "error"
            agent_run.error_message = str(exc)
            session.commit()
            raise HTTPException(status_code=502, detail=f"agent run failed: {exc}") from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        span.set_attribute("resolvegrid.latency_ms", latency_ms)

    output_text = final_state.get("output_text") or ""
    retrieved_chunks = final_state.get("retrieved_chunks") or []
    retrieval_sufficient = bool(final_state.get("retrieval_sufficient"))
    verified_chunk_ids = set(final_state.get("verified_chunk_ids") or [])

    agent_run.status = "completed"
    agent_run.output_text = output_text
    agent_run.completed_at = datetime.now(timezone.utc)

    # See docstring above: real per-node latencies, not a fabricated even
    # split. `verify_citations` has no dedicated row in this table's
    # established 4-row shape -- its real measured time is folded into
    # "finalize" instead (see docstring for why that's safe/documented).
    node_latencies_ms = final_state.get("node_latencies_ms") or {}
    # Keys derived from `_STAGE_NAMES` directly (not re-typed as separate
    # string literals) so this dict can never silently drift out of sync
    # with that tuple.
    stage_latency_ms = {stage_name: node_latencies_ms.get(stage_name, 0) for stage_name in _STAGE_NAMES}
    stage_latency_ms["finalize"] += node_latencies_ms.get("verify_citations", 0)
    for stage_name in _STAGE_NAMES:
        session.add(
            Span(
                agent_run_id=agent_run.id,
                stage_name=stage_name,
                status="success",
                latency_ms=stage_latency_ms[stage_name],
            )
        )
    session.commit()

    # Citation/caption logic (Phase 7 Task 7, refined Phase 8 Task 7): only
    # surface citations when `compose_response` actually used the
    # citation-grounded prompt branch (retrieval_sufficient AND non-empty
    # chunks -- mirrors graph.py's own branch condition exactly, so the UI
    # never claims a grounded answer the model wasn't actually given KB
    # context for). Otherwise this is the same general-knowledge-answer
    # case Phase 6 always produced -- with an updated caption, since "no
    # ticket or company-specific knowledge base yet" is no longer true now
    # that a real KB exists; see graph.py's module docstring for why an
    # insufficient/empty retrieval result still gets a best-effort
    # general-knowledge answer rather than a hard abstention.
    #
    # Phase 8 Task 7: citations are further filtered down to
    # `verified_chunk_ids` -- the graph's `verify_citations` node (running
    # after `compose_response`, before `finalize`) already rewrote
    # `output_text` to strip any fabricated `[chunk:<id>]` marker before
    # this handler ever sees it (see state.py's `citations_verified`
    # docstring); this filter keeps the separately-returned `citations`
    # list in lockstep with that same guarantee -- only citations the
    # model actually made AND that verified against `context_block` are
    # ever surfaced to the UI, never the full pre-verification
    # `retrieved_chunks` set (which may include chunks retrieved but never
    # actually cited in the answer).
    if retrieval_sufficient and retrieved_chunks:
        citations = [
            {"chunk_id": chunk["chunk_id"], "document_title": chunk["document_title"]}
            for chunk in retrieved_chunks
            if chunk["chunk_id"] in verified_chunk_ids
        ]
        if citations:
            noun = "citation" if len(citations) == 1 else "citations"
            caption = (
                f"Answer grounded in {len(citations)} knowledge-base {noun} from the "
                "Kestrel knowledge base — see citations below."
            )
        else:
            # Retrieval was sufficient and chunks were shown to the model,
            # but the answer either cited nothing or cited only fabricated
            # ids that got stripped -- there is nothing verified to point
            # to, so this degrades to the same caption as "no KB match"
            # rather than falsely implying grounded citations exist.
            caption = _NO_KB_MATCH_CAPTION
    else:
        citations = []
        caption = _NO_KB_MATCH_CAPTION

    # Real graph run succeeded -- populate the exact-response cache for a
    # future identical (kb_version, scope_sig, message) lookup. Includes
    # the classified `intent`/`risk_level` in the stored VALUE (per this
    # task's key-shape correction -- see response_cache.py's module
    # docstring): a future cache HIT reads them back from here instead of
    # ever needing to re-run classify_intent to know them.
    store_cached_response(
        message_text=payload.message,
        kb_version=kb_version,
        scope_sig=scope_sig,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        redis_client=redis_client,
        value={
            "answer": output_text,
            "citations": citations,
            "caption": caption,
            "intent": final_state.get("intent") or "unclear",
            "risk_level": final_state.get("risk_level") or "low",
        },
    )

    return {
        "answer": output_text,
        "thread_id": thread_id,
        "citations": citations,
        "caption": caption,
    }
