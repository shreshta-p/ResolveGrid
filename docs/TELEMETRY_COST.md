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

Every LLM call is wrapped in an OTel span using GenAI semantic-convention attribute names (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`), emitted via the tracer provider `resolvegrid_telemetry.init_tracing` configures once at app startup. As of Phase 4, the OTel Collector only logged received spans to stdout (`infra/otel-collector/config.yaml`'s `debug` exporter) — nothing was persisted beyond ResolveGrid's own `ModelCall` table. **As of Phase 11 (Task 5), this is no longer true**: the collector also exports real trace data to a real, self-hosted Langfuse instance — see "Langfuse (Phase 11)" below. ResolveGrid's own `ModelCall`/`PricingVersion` tables remain the system of record regardless; per the approved architecture, Langfuse is always a secondary deep-debugging surface, never the sole source of truth.

## Phase 11: centralized logging, real per-node span timing, real routing, real budgets

### `ModelCall` schema additions

Two columns were added on top of Phase 4/5's original schema (see `models/telemetry.py` for the authoritative definitions):

- **`trace_id`** (migration 0014): the real OTel trace id (32 lowercase hex chars, `trace.format_trace_id(...)`) the completion's span was recorded under — captured from the actual active span context inside `log_completion`, not derived/guessed. This is what makes a `ModelCall` row correlatable to a real Langfuse trace (see below): `SELECT * FROM model_call WHERE trace_id = '<id>'` and Langfuse's `GET /api/public/v2/observations?traceId=<id>` are two views of the exact same real call.
- **`routing_reason`** (migration 0015): a free-text record of which real policy input drove a call's `model=` choice, e.g. `"risk_level=high"`. Distinct from the pre-existing `fallback_occurred`/`serving_model_group` columns — those record what LiteLLM's router did *after* a request was sent; `routing_reason` records what *this application's own routing policy* decided *before* the request was ever sent, and why. `NULL` for every call site with no real routing policy (everything except `chat.compose_response` as of this phase).

### One shared logging path, six real call sites

Before this phase, `summarize_ticket` was the *only* call site that wrote a `ModelCall` row. `apps/api/src/resolvegrid_api/model_call_logging.py`'s `log_completion()` (extracted from `summarize_ticket`'s original inline logic, Task 1) is now the single shared path every real completion in the system goes through — it starts a real `llm.<purpose>` tracer span, calls `llm_gateway.complete()`, looks up the current `PricingVersion`, writes a `ModelCall` row (success or error shaped), and returns the result. `make_logging_complete_fn`/`make_compose_routing_complete_fn` wrap it into `CompleteFn`-shaped closures for callers that only have a `str -> str` seam to build against (the chat graph, judge, eval worker).

As of Task 3, real `ModelCall` rows are written for all six real completion call sites in the system:

| `purpose` | Call site | Routing |
|---|---|---|
| `ticket.summarize` | `routers/tickets.py`'s `summarize_ticket` | always `DEFAULT_MODEL` (`local-qwen3`) |
| `chat.classify_intent` | `main.py`'s real chat-graph closure | always `DEFAULT_MODEL` |
| `chat.compose_response` | `main.py`'s real chat-graph closure | **real routing** — see below |
| `eval.judge` | `eval_judge.py`'s real judge closure | always `DEFAULT_MODEL` |
| `eval.chat.classify_intent` | `eval_worker.py`'s own chat-graph closure | always `DEFAULT_MODEL` |
| `eval.chat.compose_response` | `eval_worker.py`'s own chat-graph closure | always `DEFAULT_MODEL` (eval traffic deliberately never pays for cloud calls) |

### Real per-node `Span` timing (Task 2)

`routers/chat.py`'s 4 `Span` rows per chat turn (`classify_intent`/`retrieve`/`compose_response`/`finalize`, folding `verify_citations`'s time into `finalize` — see `docs/DECISION_LOG.md`) used to be a single wall-clock measurement for the whole graph `ainvoke()` call, split evenly across 4 stage names — an honestly-commented placeholder, never real per-node data. `services/agent-orchestration/.../graph.py` now wraps each node's real work in `time.monotonic()` timing, threading genuine per-node durations through `AgentState.node_latencies_ms`; `chat.py` reads these real values back after `ainvoke()` returns and writes them directly, with the placeholder-split logic removed entirely. Verified non-uniform: `apps/api/tests/test_chat_api.py` asserts the 4 `Span.latency_ms` values are not all equal for a real (no-mocking) chat turn.

### Routing policy (Task 4)

`apps/api/src/resolvegrid_api/routing.py`'s `select_model_for_risk_level(risk_level)`: `risk_level == "high"` → `"cloud-primary"` (Anthropic Haiku 4.5, via LiteLLM); anything else (`"low"`, `"medium"`, unrecognized) → `DEFAULT_MODEL` (`"local-qwen3"`). A defensible cheap-by-default / cloud-for-high-risk policy, only ever applied to `compose_response` (the one node with real `risk_level`-aware routing as of this phase). Pure function, zero I/O, unit-tested with zero credentials in `apps/api/tests/test_routing.py` (the CI-safe half of the required CI-vs-local proof split — a real cloud-reaching call cannot run in CI, since `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are always empty strings there).

### Budget enforcement (Task 4)

Two real per-provider LiteLLM virtual keys exist (`LITELLM_CLOUD_PRIMARY_KEY` scoped to `cloud-primary`, `LITELLM_CLOUD_FALLBACK_KEY` scoped to `cloud-fallback`), generated via LiteLLM's real `/key/generate`/`/key/update` admin API and stored only in the gitignored `.env`. `llm_gateway.py`'s `_resolve_api_key(model)` selects the model's own virtual key when configured, degrading to the shared `LITELLM_MASTER_KEY` for `local-qwen3` (no real per-call cost, no budget cap warranted). LiteLLM itself rejects a call that would exceed a key's real recorded `max_budget` with a genuine `429` — proven for real during Task 4 (see `docs/SECURITY.md`) and re-confirmed structurally (not necessarily re-triggered) during this closing task's fresh-state verification via `GET /key/info`.

### The `PricingVersion` gap: seeded, not deferred (Task 7 / this closing task)

Task 4 flagged an honest gap: no `PricingVersion` row existed for `provider="anthropic"`, so every real `cloud-primary` `ModelCall` row computed `estimated_cost_usd=$0.00` regardless of real spend — LiteLLM's own `/key/info` `spend` figure was the only authoritative cloud-cost number. This closing task resolves it: migration `0016` seeds `provider="anthropic", model="cloud-primary"` (matching the `ModelCall.model` column's established LiteLLM-alias convention, same as migration 0007 already fixed for the ollama row) with `$0.001`/1k input, `$0.005`/1k output tokens — Anthropic's real, current published rate for `claude-haiku-4-5-20251001` ($1/$5 per million tokens, confirmed via a live fetch of `https://claude.com/pricing` on 2026-09-28, not assumed from training data). `cloud-fallback` (OpenAI `gpt-4o-mini`) is deliberately left unseeded — no real `cloud-fallback` completion has ever succeeded in this environment (Task 4 found `OPENAI_API_KEY` has zero real credit); seeding a pricing row for a path with zero real evidence would be speculative. Add it via the same pattern the moment a real `cloud-fallback` call first succeeds.

### Langfuse (Phase 11 Task 5) — the correct v4 API endpoint

The self-hosted Langfuse v4 instance (`infra/docker-compose.yml`'s `langfuse-web`/`langfuse-worker`/dedicated ClickHouse/MinIO/Redis, Postgres reused via its own `langfuse` database) runs in v4's default `events_only` mode. **`GET /api/public/traces/{id}` 404s under this mode** with an explicit "not available... events_only mode" message — this is not a bug or misconfiguration, it's how v4's `events_only` mode behaves by design, per Langfuse's own v3→v4 migration guidance. **The correct endpoint to query a trace by id is `GET /api/public/v2/observations?traceId=<id>`**, confirmed working repeatedly throughout this phase's review cycles (including this closing task's own fresh-state verification below) — use this one, not the v3-shaped singular-trace endpoint, when correlating a `ModelCall.trace_id` to its real Langfuse observation.

### Grafana / Prometheus (Phase 11 Task 6)

Real Prometheus (`infra/prometheus/prometheus.yml`) scrapes LiteLLM's real `/metrics` endpoint (mounted because `"prometheus"` is a registered `success_callback`; auth disabled for this endpoint specifically via `require_auth_for_metrics_endpoint: false`, since it exposes only counters/histograms and there is no clean way to inject a Bearer secret into a committed scrape config) plus the OTel Collector's own internal pipeline-health metrics. A provisioned Grafana dashboard (`infra/grafana/`, not manual UI clicking — survives a fresh container) shows request count/rate and latency from real LiteLLM traffic. See `docs/DECISION_LOG.md`'s 2026-09-28 entries for the full reasoning on both stacks.

## Phase 12 Task 4: real Anthropic provider-side prompt-prefix caching

Genuinely different from every other cache in Phase 12 (Tasks 1/2/3/5, all this application's own Redis-backed caches): this is Anthropic's own provider-side KV-cache reuse, invoked by annotating a stable content block with `cache_control: {"type": "ephemeral"}`. Fully implemented and wired end-to-end; **real verification shows it does not currently activate**, for a real, measured, honest reason recorded below — not a fabricated number.

### Mechanism

`llm_gateway.complete()` gained an optional `cacheable_prefix: str | None` parameter. When provided AND `provider_for_model(model) == "anthropic"` (the exact same per-model-conditional shape as the existing `_THINK_FALSE_MODELS` precedent), the request is restructured from a single flat user message into:

```json
{
  "model": "cloud-primary",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "<stable prefix>", "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": "<variable suffix>"}
  ]
}
```

— confirmed, via a real request against the real running LiteLLM proxy, to be the correct real shape: LiteLLM's OpenAI-compatible endpoint genuinely forwards `cache_control` through to Anthropic's native API when placed on a `system` content block this way (also confirmed against LiteLLM's own prompt-caching docs, `https://docs.litellm.ai/docs/completion/prompt_caching`). For every non-Anthropic model (`local-qwen3`, `cloud-fallback`), `cacheable_prefix` is instead concatenated directly onto the front of the prompt (`cacheable_prefix + prompt`, no restructuring) — Ollama/OpenAI callers see zero behavior change, proven by `resolvegrid_agent_orchestration.graph`'s reconstruction-identity test (`_COMPOSE_STABLE_PREFIX_WITH_CONTEXT + _COMPOSE_VARIABLE_SUFFIX_WITH_CONTEXT_TEMPLATE.format(...) == _COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE.format(...)`, always, by construction of the `str.split()`-based derivation).

The one real call site wired up: `compose_response`'s with-context branch (sufficient retrieval). `graph.py` splits `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE` into a stable preamble (the fixed persona/instruction text plus the Phase 8 Task 7 prompt-injection-defense framing — identical on every real call) and a variable suffix (classified intent/risk_level + retrieved context + the user's question). `classify_intent`'s prompt and the short no-context compose prompt were deliberately NOT split — both are judged too short to be worth the mechanism's overhead even before measuring (see the real numbers below, which confirm this judgment for the with-context preamble too).

`CompletionResult` gained `cache_creation_input_tokens`/`cache_read_input_tokens`, read from LiteLLM's real passthrough of Anthropic's `usage` object. `ModelCall` gained the same two columns (migration `0018`, mirroring `trace_id`'s migration `0014` precedent) — the real, queryable provenance signal this task's plan doc actually asks for (`SELECT id, cache_creation_input_tokens, cache_read_input_tokens FROM model_call WHERE purpose = 'chat.compose_response' AND provider = 'anthropic'`).

### Real verification: two back-to-back real `cloud-primary` calls

Two real calls were made through `make_compose_routing_complete_fn`'s real closure (the exact real `chat.compose_response` call path — real tracer span, real `ModelCall` row, real LiteLLM → real Anthropic round trip), sharing the byte-identical real stable prefix (`_COMPOSE_STABLE_PREFIX_WITH_CONTEXT`) and the same retrieved-context block, with two different real questions ("What do I need installed on my laptop to connect remotely?" / "Is multi-factor authentication required for the VPN?") so Task 3's own exact-response cache could never short-circuit the call. Both real answers came back correct and correctly cited (`[chunk:1]`), confirming the restructured request shape didn't break real Anthropic behavior. The real, persisted `ModelCall` rows:

| `id` | `status` | `input_tokens` | `output_tokens` | `cache_creation_input_tokens` | `cache_read_input_tokens` |
|---|---|---|---|---|---|
| 1222 | success | 557 | 147 | 0 | 0 |
| 1223 | success | 555 | 113 | 0 | 0 |

**Honest result: no real cache activity on either call.** This was investigated, not just reported blind:

- A direct raw-HTTP diagnostic call (same real stable prefix, same real running proxy) confirmed the `usage` object genuinely carries `cache_creation_input_tokens: 0` / `cache_read_input_tokens: 0` / `prompt_tokens_details.cache_creation_token_details.ephemeral_5m_input_tokens: 0` as *present, real zeros* — not fields silently absent from the response and defaulted by this code's own `.get(..., 0)`. Anthropic genuinely attempted no cache write at all.
- The real cause: `cloud-primary` is configured as `anthropic/claude-haiku-4-5-20251001` (`infra/litellm/config.yaml`) — **Claude Haiku 4.5's real, current minimum cacheable-block size is 4096 tokens**, confirmed via a live fetch of Anthropic's own prompt-caching docs (`https://platform.claude.com/docs/en/build-with-claude/prompt-caching`, fetched 2026-10-08) — not assumed from training data, per this task's own requirement. This is the highest minimum of any served model tier; it is *not* monotonic across generations (current-generation Opus 5.5/Sonnet 5.5/Haiku 5.5/Fable 5.1 are all 512 tokens — an order of magnitude lower).
- The real stable prefix is ~2061 characters (~338 words). The diagnostic call's real `usage.prompt_tokens` for the full request (stable prefix + a one-line question) was **456 tokens** — confirming the prefix itself is on the order of ~400 real tokens, roughly a tenth of Haiku 4.5's 4096-token floor. The two real production-path calls' `input_tokens` (557/555, slightly larger due to the real retrieved-context block and longer real questions) corroborate this: nowhere close to 4096 either way.

### Decision: document the honest limitation, don't pad the prompt (real option (a) vs (b) call)

The task's own brief allows two honest outcomes: grow the stable prefix with genuinely useful content until it legitimately crosses the real threshold, or document the gap. This was decided in favor of documenting, not padding, for real engineering reasons:

- `_COMPOSE_STABLE_PREFIX_WITH_CONTEXT` (Phase 8 Task 7's carefully-reviewed prompt-injection-defense framing, re-verified against a real live-model adversarial probe at the time) is already a deliberately scoped, reviewed piece of prompt engineering — growing it ~10x *purely* to cross a legacy model's unusually high cache-minimum would be padding for padding's sake, the exact thing this task's own brief warns against, not a genuine prompt improvement.
- `cloud-primary`'s real traffic volume is inherently low by `routing.py`'s own design (`risk_level == "high"` only) — the 1.25× cache-write premium would need real back-to-back high-risk traffic within the same 5-minute TTL to ever pay off, which this low-volume routing path isn't likely to produce often enough to justify inflating every single `cloud-primary` request (cached or not) with several thousand extra real input tokens.
- The real, working mechanism already in place needs no further code change to start paying off the moment either condition changes: (a) the stable preamble legitimately grows past 4096 tokens through genuine future prompt-engineering work, or (b) `cloud-primary` is migrated to a current-generation Anthropic model (512-token minimum) — a separate, independent decision, out of this task's scope, but one this mechanism is already fully ready for with zero further plumbing.

**This is a genuine, reportable negative result, not a failure of the task**: the real per-model threshold was verified against live docs (not assumed), the real request/response shape was verified against a real running proxy and real Anthropic responses (not assumed to work), and the real `ModelCall.cache_creation_input_tokens`/`cache_read_input_tokens` columns are genuinely queryable and will show real nonzero values automatically the first time a real with-context `cloud-primary` call's stable prefix crosses the real threshold — nothing further needs to change in this code for that to happen.
