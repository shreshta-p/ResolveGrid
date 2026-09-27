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
"""

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Tracer
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.models import ModelCall, PricingVersion


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

        # gen_ai.system is hardcoded "ollama" here, matching
        # summarize_ticket's exact pre-extraction behavior (it never varied
        # by provider since only DEFAULT_MODEL was ever requested). This is
        # a known simplification, not a general correctness fix -- Task 4's
        # real model-routing work will need to derive this from the
        # actually-targeted provider once calls other than local-qwen3 are
        # routed through here. Out of scope for Task 1's behavior-preserving
        # refactor.
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", model)
        try:
            result = llm_gateway.complete(prompt, model=model)
        except llm_gateway.LLMGatewayError as exc:
            span.set_attribute("error.type", type(exc).__name__)
            session.add(
                ModelCall(
                    purpose=purpose, provider="ollama", model=model,
                    pricing_version_id=None, input_tokens=0, output_tokens=0, latency_ms=0,
                    estimated_cost_usd=0.0, status="error", error_message=str(exc),
                    trace_id=trace_id,
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
            trace_id=trace_id,
        )
    )
    return result
