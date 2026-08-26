import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry import trace
from pydantic import BaseModel
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models import AgentRun, Span
from resolvegrid_authz import Principal

router = APIRouter(prefix="/chat", tags=["chat"])

# The global tracer provider is already configured once at app startup by
# main.py's lifespan hook (resolvegrid_telemetry.init_tracing) -- this module
# must NOT call init_tracing again, just bind a tracer to whatever provider is
# globally registered by the time a span is actually started (same pattern
# as tickets.py).
tracer = trace.get_tracer(__name__)

# The graph's 3 conceptual stages, in execution order -- used to write one
# DB `Span` row per stage. See chat()'s docstring for why their latency_ms
# values are a documented simplification, not real per-node timing.
_STAGE_NAMES = ("classify_intent", "compose_response", "finalize")


class ChatRequest(BaseModel):
    message: str


@router.post("")
async def chat(
    payload: ChatRequest,
    request: Request,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """Single-turn chat: run the classify_intent -> compose_response ->
    finalize graph (via `request.app.state.agent_graph`, wired up in
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

    Span/timing note: this task doesn't have genuine per-node timing
    instrumentation wired through LangGraph's own internals yet -- only the
    single `ainvoke()` call's wall-clock time is actually measured. The 3
    `Span` rows below (one per conceptual stage) split that single measured
    duration evenly across the 3 stages as a documented, honest
    simplification -- this is NOT real per-node timing and must not be read
    as such; a future phase that wants genuine per-node latency would need
    to instrument inside the graph nodes themselves (see
    services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py)
    or hook LangGraph's own streaming/callback API.
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

    initial_state = {
        "thread_id": thread_id,
        "principal_employee_id": principal.employee_id,
        "input_text": payload.message,
        "intent": None,
        "risk_level": None,
        "output_text": None,
        "error": None,
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

    agent_run.status = "completed"
    agent_run.output_text = output_text
    agent_run.completed_at = datetime.now(timezone.utc)

    # See docstring above: latency is only measured for the whole ainvoke()
    # call, not per node -- split evenly across the 3 stages as an honest
    # placeholder, not a claim of real per-node instrumentation.
    per_stage_latency_ms = latency_ms // len(_STAGE_NAMES)
    for stage_name in _STAGE_NAMES:
        session.add(
            Span(
                agent_run_id=agent_run.id,
                stage_name=stage_name,
                status="success",
                latency_ms=per_stage_latency_ms,
            )
        )
    session.commit()

    return {"answer": output_text, "thread_id": thread_id}
