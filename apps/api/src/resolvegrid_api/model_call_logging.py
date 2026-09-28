"""Shared LLM-completion + telemetry logging path.

Extracted from `routers/tickets.py`'s `summarize_ticket` (Phase 11 Task 1),
which was, until now, the ONLY call site in the codebase that actually wrote
a `ModelCall` row. As later Phase 11 tasks wire real logging into the chat
graph, judge, and eval worker, this is the one place that owns:

- the GenAI-semconv tracer span around a real `llm_gateway.complete()` call
- the real OTel trace-id capture that makes a `ModelCall` row correlatable to
  a real trace (Langfuse, Task 5)
- the `PricingVersion` lookup + `estimated_cost_usd` math
- the `ModelCall` row itself, success or error shaped

Commit-boundary convention: like `audit.py`'s `record_audit_event`, this
function `session.add()`s and does NOT call `session.commit()` -- the caller
owns the transaction boundary (e.g. `summarize_ticket` still commits once,
after its own subsequent authz/state work, exactly as it did before this
extraction).

Span-exception-attribution note (real, silent difference from the
pre-extraction code, documented per code review): pre-refactor,
`summarize_ticket` raised its `HTTPException` INSIDE the same `llm.*` span,
so that span recorded `HTTPException` (with `LLMGatewayError` as its cause)
as the exception that ended it. Post-extraction, `log_completion`'s span
closes on the original `LLMGatewayError` itself -- arguably more correct,
since that's the exception this module actually raises -- and any
`HTTPException` a caller (e.g. `summarize_ticket`) wraps it in afterward is
raised outside any span entirely. No test depends on either shape; this is
a deliberate outcome of the extraction, not an oversight.

`purpose` string convention (breadcrumb added per code review, before a 7th
purpose string gets added ad hoc): `<domain>.<action>`, with an `eval.`
prefix reserved for eval-harness-driven traffic so it's never confused with
real user/production traffic sharing the same underlying code path. Every
real value in use as of Phase 11 Task 3: `"ticket.summarize"` (Task 1,
`routers/tickets.py`), `"chat.classify_intent"` / `"chat.compose_response"`
(`main.py`'s real chat-graph closures), `"eval.chat.classify_intent"` /
`"eval.chat.compose_response"` (`eval_worker.py`'s own, separately-built
chat-graph closures), and `"eval.judge"` (`eval_judge.py`'s real judge
closure).
"""

from typing import Callable

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Tracer
from resolvegrid_agent_orchestration import ComposeCompleteFn
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.models import ModelCall, PricingVersion
from resolvegrid_api.routing import select_model_for_risk_level


def current_pricing_version(session: Session, provider: str, model: str) -> PricingVersion | None:
    # Moved here from routers/tickets.py (its only prior call site) and made
    # PUBLIC (no leading underscore), not left private-and-duplicated:
    # log_completion() below needs it to compute estimated_cost_usd for the
    # ModelCall row it writes, and routers/tickets.py's summarize_ticket
    # ALSO needs it -- log_completion() only returns a CompletionResult (its
    # signature is pinned by this task's spec), which has no cost field, but
    # summarize_ticket's HTTP response body has always included
    # estimated_cost_usd. Rather than growing log_completion's return type
    # (out of scope) or duplicating this lookup verbatim in two modules, the
    # single lookup function is shared: summarize_ticket calls it a second
    # time, purely to shape its own response body, after log_completion has
    # already used it once internally to price the ModelCall row it wrote.
    # One extra cheap indexed SELECT against a tiny table, versus a second
    # copy of the lookup logic living in two files -- reuse wins.
    #
    # No `effective_at <= now()` filter -- harmless today (one seeded $0 row
    # for ollama/local-qwen3), but once a real paid provider adds a
    # future-dated PricingVersion row (a scheduled rate change), this would
    # pick it up early and misprice calls made before that date. Fix before
    # Phase 5 adds real provider pricing.
    return session.scalar(
        select(PricingVersion)
        .where(PricingVersion.provider == provider, PricingVersion.model == model)
        .order_by(PricingVersion.effective_at.desc())
    )


def log_completion(
    session: Session,
    tracer: Tracer,
    *,
    purpose: str,
    prompt: str,
    model: str = llm_gateway.DEFAULT_MODEL,
    routing_reason: str | None = None,
) -> llm_gateway.CompletionResult:
    """Run a real LLM completion under a real tracer span, log a `ModelCall`
    row (success or error shaped), and return the `CompletionResult`.

    `purpose` drives both the span name (`llm.<purpose>`) and the
    `ModelCall.purpose` column -- e.g. `purpose="ticket.summarize"` yields a
    span named `llm.ticket.summarize`. This is a deliberate, small, NOT
    behavior-identical change from the pre-extraction span name
    (`llm.summarize_ticket`): reusing one string for both the span name and
    the DB column is what lets every future call site (chat graph, judge,
    eval worker -- Task 3) get a sensible span name for free just by picking
    a `purpose`, with zero risk of the two drifting apart. No test asserts
    the literal span name string (only `ModelCall.purpose`, which is
    unchanged), so this is safe; documented here per the plan's
    self-review requirement.

    On `LLMGatewayError`, writes an error-shaped `ModelCall` row (matching
    `summarize_ticket`'s original error path exactly: zero tokens/latency/
    cost, no pricing_version_id, `error_message=str(exc)`) and re-raises the
    SAME exception unchanged -- callers still decide how to surface it (e.g.
    `summarize_ticket` turns it into an HTTP 502).

    Does NOT call `session.commit()` in either the success or error path --
    the caller owns the transaction boundary (same convention as `audit.py`'s
    `record_audit_event`) and MUST commit (or otherwise flush/persist) the
    session itself, or the `ModelCall` row this just added is silently lost.
    """
    with tracer.start_as_current_span(f"llm.{purpose}") as span:
        # Real OTel trace-id capture: read the actual current trace id from
        # the active span context, formatted as the standard 32-hex-char
        # lowercase string via the OTel API's own formatter (verified
        # against the installed opentelemetry-api==1.44.0:
        # `trace.format_trace_id` returns exactly this format). Captured
        # up front so both the success and error `ModelCall` rows below can
        # carry it.
        span_context = otel_trace.get_current_span().get_span_context()
        trace_id = otel_trace.format_trace_id(span_context.trace_id)

        # Phase 11 Task 4 fix (flagged by name in the comment this replaces):
        # gen_ai.system/the error-row's `provider` used to be hardcoded
        # "ollama" unconditionally, correct only because no non-local model
        # was ever actually requested before this task. Now that real
        # routing (routing.py) can request `cloud-primary`/`cloud-fallback`,
        # deriving the requested model's real provider via
        # `llm_gateway.provider_for_model` -- so a failed cloud call's error
        # row is never mislabeled "ollama".
        requested_provider = llm_gateway.provider_for_model(model)
        span.set_attribute("gen_ai.system", requested_provider)
        span.set_attribute("gen_ai.request.model", model)
        try:
            result = llm_gateway.complete(prompt, model=model)
        except llm_gateway.LLMGatewayError as exc:
            span.set_attribute("error.type", type(exc).__name__)
            session.add(
                ModelCall(
                    purpose=purpose, provider=requested_provider, model=model,
                    pricing_version_id=None, input_tokens=0, output_tokens=0, latency_ms=0,
                    estimated_cost_usd=0.0, status="error", error_message=str(exc),
                    trace_id=trace_id, routing_reason=routing_reason,
                )
            )
            raise

        span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
        # "gen_ai.response.model" is the currently-installed OTel GenAI
        # semconv name for the model that actually served the response (see
        # opentelemetry.semconv._incubating.attributes.gen_ai_attributes.
        # GEN_AI_RESPONSE_MODEL) -- flagged deprecated-in-favor-of-the-
        # standalone genai-semconv-repo in that module's docstring, but it's
        # still the only "response model" constant this installed package
        # ships, so it's the correct name to emit today. Only set when a
        # fallback (or any distinct serving model group) actually occurred;
        # a non-fallback call already has the request model on this span.
        if result.serving_model_group:
            span.set_attribute("gen_ai.response.model", result.serving_model_group)

    pricing = current_pricing_version(session, result.provider, result.model)
    if pricing is not None:
        estimated_cost_usd = (
            result.input_tokens / 1000 * pricing.input_cost_per_1k_tokens_usd
            + result.output_tokens / 1000 * pricing.output_cost_per_1k_tokens_usd
        )
    else:
        estimated_cost_usd = 0.0

    session.add(
        ModelCall(
            purpose=purpose, provider=result.provider, model=result.model,
            pricing_version_id=pricing.id if pricing is not None else None,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_ms=result.latency_ms, estimated_cost_usd=estimated_cost_usd, status="success",
            fallback_occurred=result.fallback_occurred, serving_model_group=result.serving_model_group,
            trace_id=trace_id, routing_reason=routing_reason,
        )
    )
    return result


def make_logging_complete_fn(
    session_factory_fn: Callable[[], Session],
    tracer: Tracer,
    *,
    purpose: str,
    model: str = llm_gateway.DEFAULT_MODEL,
) -> Callable[[str], str]:
    """Build a plain `Callable[[str], str]` ("CompleteFn"-shaped, matching
    `resolvegrid_agent_orchestration.graph.CompleteFn` and every other
    `CompleteFn`-consuming seam in this codebase, e.g. `eval_judge.py`'s
    real judge closure) that opens its OWN short-lived session via
    `session_factory_fn()`, runs a real completion through `log_completion`
    under it, commits, and returns just the completion text.

    Phase 11 Task 3: this is the one shared shape every real "closure that
    needs to log a `ModelCall` row but is built once, outside any
    per-request `Depends(get_db)` scope" call site uses --
    `apps/api/main.py`'s chat-graph `classify_intent`/`compose_response`
    closures, `eval_worker.py`'s own (separately built, per that module's
    docstring) chat-graph closures, and `eval_judge.py`'s real judge
    closure -- rather than duplicating the same open-session/log/commit
    /return-text shape three times. Mirrors `agent_retrieval.py`'s
    already-established `retrieve_for_agent` pattern exactly (see that
    module's docstring for the full "why session_factory(), not a shared
    request session" rationale): a closure built once at app/worker
    startup cannot close over a live per-request `Session` (there isn't
    one yet, and `AgentState` can never carry one across a checkpointed
    graph run -- see `state.py`'s module docstring), so it opens its own
    per-call session instead, commits immediately (this closure -- unlike
    `log_completion` itself -- DOES own its own commit boundary, since
    there is no enclosing request/case handler positioned to do it for
    it), and hands back a live-DB-session-free `str`, the only thing
    `CompleteFn`'s contract requires.

    Not a "new global/module-level session": `session_factory_fn` is
    itself passed in as ordinary dependency injection (real call sites
    pass `resolvegrid_api.db.session_factory` -- itself just a factory
    function, not a shared session object) and a genuinely fresh `Session`
    is opened and closed on every single call.
    """

    def complete_fn(prompt: str) -> str:
        with session_factory_fn() as session:
            # Mirrors routers/tickets.py's summarize_ticket commit-on-both
            # -paths precedent exactly (code-review fix: narrowed to the
            # SAME exception type that precedent actually catches, not a
            # bare `except Exception` -- the broader catch previously here
            # overclaimed "exact" parity it didn't have). On
            # `LLMGatewayError` specifically, log_completion() has already
            # session.add()'d an error-shaped ModelCall row and re-raises --
            # that row must still be committed here before the exception
            # propagates, or `Session.__exit__`'s implicit
            # rollback-on-uncommitted-work would silently discard it,
            # leaving no trace of a real failure. No other exception
            # log_completion can raise (a non-LLMGatewayError failure
            # inside `llm_gateway.complete()`, or inside
            # `current_pricing_version`) reaches session.add() first --
            # verified by reading log_completion's body -- so there is
            # nothing pending to lose on any other exception path; letting
            # those propagate through this `with` block's ordinary
            # rollback-on-close is correct, not a gap.
            try:
                result = log_completion(session, tracer, purpose=purpose, prompt=prompt, model=model)
            except llm_gateway.LLMGatewayError:
                session.commit()
                raise
            session.commit()
            return result.text

    return complete_fn


def make_compose_routing_complete_fn(
    session_factory_fn: Callable[[], Session],
    tracer: Tracer,
    *,
    purpose: str,
) -> ComposeCompleteFn:
    """Build the real, `ModelCall`-logging `ComposeCompleteFn` (Phase 11
    Task 4) `main.py` wires into `compose_response` -- the one node with a
    real, risk-aware routing policy (see `routing.py`'s module docstring).

    Same "open own short-lived session per call, commit on both success and
    `LLMGatewayError` paths" shape as `make_logging_complete_fn` above (see
    that function's docstring for the full rationale) -- the ONE real
    difference is the extra `risk_level` argument `graph.py`'s
    `make_compose_response_node` now passes (see that module's
    `ComposeCompleteFn` docstring for why `compose_response` specifically
    needs a second, real call argument rather than the narrower, prompt
    -text-only `CompleteFn` every other node still uses): this closure
    derives the real `model=` to request via `select_model_for_risk_level`,
    and records WHICH risk_level drove that choice on the `ModelCall` row's
    `routing_reason` column, e.g. `"risk_level=high"` -- see
    `models/telemetry.py`'s `ModelCall.routing_reason` docstring for why
    that's judged worth a real column rather than being left implicit in
    `model`/`provider` alone.

    Deliberately NOT reused by `eval_worker.py`'s own compose closure --
    that module wraps `make_logging_complete_fn` directly instead (see its
    own docstring for why real eval-harness traffic stays on the zero-cost
    local model regardless of a case's classified risk_level, rather than
    incurring real per-run cloud spend on every automated eval suite
    execution).
    """

    def complete_fn(prompt: str, risk_level: str) -> str:
        model = select_model_for_risk_level(risk_level)
        routing_reason = f"risk_level={risk_level}"
        with session_factory_fn() as session:
            # See make_logging_complete_fn's docstring/inline comment above
            # for why LLMGatewayError specifically must still be committed
            # before re-raising (the error-shaped ModelCall row
            # log_completion already session.add()'d would otherwise be
            # silently discarded by Session.__exit__'s implicit rollback).
            try:
                result = log_completion(
                    session, tracer, purpose=purpose, prompt=prompt, model=model,
                    routing_reason=routing_reason,
                )
            except llm_gateway.LLMGatewayError:
                session.commit()
                raise
            session.commit()
            return result.text

    return complete_fn
