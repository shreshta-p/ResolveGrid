# Telemetry & Cost

## Schema (Phase 4)

`ModelCall` and `PricingVersion` (defined in [`apps/api/src/resolvegrid_api/models/telemetry.py`](../apps/api/src/resolvegrid_api/models/telemetry.py) — the single source of truth for field names/types, not duplicated here) record every LLM call ResolveGrid makes, whether it succeeds or fails.

- **`PricingVersion`** is a versioned, never-mutated snapshot of per-1k-token pricing for one `provider`+`model` pair. Multiple rows can exist for the same provider/model over time (e.g. a provider's published rate changes); a `ModelCall` always points at the specific `PricingVersion` row that was current when the call was made, so historical cost never silently shifts when a newer price is added later.
- **`ModelCall`** records one call: `purpose` (what it was for, e.g. `"ticket.summarize"`), `provider`/`model`, token counts, latency, the computed `estimated_cost_usd`, and `status` (`"success"` or `"error"`, with `error_message` set on failure). A row is written on **both** outcomes — a failed call is not silently dropped, matching the approved architecture's error-taxonomy principle that failures are tracked, not hidden.

## Cost computation

```
estimated_cost_usd = (input_tokens / 1000 * pricing.input_cost_per_1k_tokens_usd)
                    + (output_tokens / 1000 * pricing.output_cost_per_1k_tokens_usd)
```

computed in [`apps/api/src/resolvegrid_api/routers/tickets.py`](../apps/api/src/resolvegrid_api/routers/tickets.py)'s `summarize_ticket` endpoint (the first caller). If no matching `PricingVersion` row is found for the call's provider/model, `pricing_version_id` is left `NULL` and cost is treated as `0.0` rather than raising — this only happens for models with no seeded pricing row, which should not occur in practice once a model is actually wired up.

## The $0.00 Ollama convention

Local Ollama inference has zero marginal cost (self-hosted, already-paid-for GPU), but still gets a real `PricingVersion` row (`provider="ollama", model="local-qwen3", $0/$0` — seeded by migration `0006`, corrected in `0007` to use the LiteLLM-facing model alias rather than the raw underlying Ollama tag, since that's the only identifier a `ModelCall` row's `model` field ever actually carries) rather than special-casing "no pricing row means free" in application code. This keeps `ModelCall`'s cost field meaningful and consistently populated regardless of which provider served a given call.

## Cloud provider routing (Phase 5)

`infra/litellm/config.yaml` defines `cloud-primary` (Anthropic, `claude-haiku-4-5-20251001`) and `cloud-fallback` (OpenAI, `gpt-4o-mini`), wired via `router_settings.fallbacks` so a `cloud-primary` failure automatically retries against `cloud-fallback`. Real API keys live only in a local, gitignored `.env` — read via `os.environ/...` in the LiteLLM config, never hardcoded or committed.

**What's actually observable about a fallback, and why the schema is shaped the way it is:** empirically verified against real, live Anthropic/OpenAI calls, LiteLLM signals a fallback via response **headers**, never the response body — `x-litellm-attempted-fallbacks` (a count) and `x-litellm-model-group` (which model group actually served the request). The specific reason a fallback occurred (the upstream provider's error text) is not available anywhere in a successful fallback response; LiteLLM swallows it internally after a successful retry. This is why `ModelCall` has `fallback_occurred: bool` and `serving_model_group: str | None` — not a `routing_reason`/`fallback_reason` free-text field, since no genuine text is ever available to populate one honestly. `llm_gateway.complete()` reads these two headers and threads them through to `ModelCall` on every successful call.

## OpenTelemetry

Every LLM call is wrapped in an OTel span using GenAI semantic-convention attribute names (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`), emitted via the tracer provider `resolvegrid_telemetry.init_tracing` configures once at app startup. As of Phase 4, the OTel Collector still only logs received spans to stdout (`infra/otel-collector/config.yaml`'s `debug` exporter) — nothing is persisted beyond ResolveGrid's own `ModelCall` table. Standing up Langfuse (trace/eval/cost UI) and Prometheus/Grafana (infra metrics) is deliberately deferred to a later phase — see `docs/DECISION_LOG.md`'s 2026-08-24 entry for the rationale. ResolveGrid's own `ModelCall`/`PricingVersion` tables are the system of record regardless of when/whether Langfuse is added; per the approved architecture, Langfuse is always a secondary deep-debugging surface, never the sole source of truth.
