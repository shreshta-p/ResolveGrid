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

## Phase 12: caching layer architecture, as-built (Task 6)

Five real caches, two shared invalidation mechanisms, and one real telemetry counter they all emit to. Full task-by-task narrative lives in `docs/superpowers/plans/2026-09-29-phase12-caching.md`; this section is the as-built reference a future reader should start from.

### The two shared mechanisms every authz-touching cache keys on

- **`scope_sig`** (`retrieval_cache.compute_scope_sig`, imported unchanged by `response_cache.py`/`semantic_cache.py`): `"unrestricted"` if the resolved `AuthzFilter` is unrestricted, else `"tags:" + ",".join(sorted(allowed_tags))`. Two employees with the identical resolved scope share a cache entry (the actual efficiency point of role/department-scoped caching); two employees with different scopes never do. The embedding cache (Task 1) has no `scope_sig` at all — an embedding of fixed text has no authz dimension.
- **`kb_version`** (`retrieval_cache.get_kb_version`): `select(max(IngestionRun.id)).where(status == "completed")`, 0 if nothing has ever completed. Re-ingesting the corpus creates a new completed `IngestionRun` row, which organically invalidates every previously-cached retrieval/response/semantic entry — not via an explicit delete pass, but because a new `kb_version` simply never looks the old key up again (it keeps existing in Redis/Postgres until its own TTL naturally expires). `apps/api/tests/test_cache_invalidation_wiring.py` is the real, non-mocked proof of this for retrieval/response (a real new completed `IngestionRun` row is created, `get_kb_version()` is shown to advance, and the identical query/message under the new version is shown to be a genuine miss while the old entry is still a live hit under its own original key); semantic cache's identical property was already proven in `test_semantic_cache.py`'s `test_kb_version_isolation_identical_embedding_and_scope_but_old_kb_version` (Task 5), cited rather than duplicated.

### The 5 caches

| Cache | Module | Backend | Key shape | TTL | Authz dimension |
|---|---|---|---|---|---|
| Embedding | `cache.py` | Redis | `cache:embedding:{model}:{sha256(text)}` | 7 days | None (unscoped, globally shareable) |
| Retrieval-result | `retrieval_cache.py` | Redis | `cache:retrieval:{kb_version}:{scope_sig}:{sha256(query_text)}` | 2 hours | `scope_sig` |
| Exact-response | `response_cache.py` | Redis | `cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:{sha256(message_text)}` | 15 minutes | `scope_sig` |
| Semantic-response | `semantic_cache.py` | Postgres/pgvector (`semantic_response_cache` table) | `(scope_sig, kb_version)` real columns + nearest-neighbor `<=>` cosine distance on `query_embedding`, threshold `0.15` | 15 minutes (applied as both a lookup-match cutoff and a real purge-deletion cutoff — see below) | `scope_sig` |
| Provider prompt-prefix (Task 4) | `llm_gateway.py` | Anthropic's own provider-side KV cache (not this app's store) | N/A (provider-internal) | Anthropic-managed (~5 min ephemeral) | N/A — content-level, not this app's authz concern |

Why each cache is a Redis `SET...EX` except semantic (Postgres): Redis is this phase's default (already the shared `resolvegrid-redis` container every Arq job also uses; namespaced keys carry no confidentiality requirement stricter than what's already in Postgres). Semantic cache is the one exception because its lookup is a similarity SEARCH, not an exact key `GET` — pgvector is this codebase's already-established vector-search mechanism (`Embedding`/`retrieval.py`), so extending it was preferred over standing up a second, parallel vector index inside Redis (see `semantic_cache.py`'s own module docstring and the plan doc's Task 5 section for the full comparison).

### Why semantic cache alone needs an active purge (Task 6)

Redis's native `EX` TTL makes "stale entries age out" a property Redis itself enforces for the other 4 caches — an expired key simply stops existing, no code needed. Postgres has no equivalent primitive: `semantic_response_cache` rows are plain rows, so `lookup_semantic_cache`'s own `created_at > now() - interval` filter only stops a stale row from ever MATCHING again, it does not remove it — left alone, every entry ever written stays a live, scanned candidate row forever, growing the table unbounded and slowing every future similarity search. `semantic_cache.purge_expired_semantic_cache_entries(session)` (built and tested in Task 5) is the real `DELETE ... WHERE created_at < cutoff` that closes this; Task 6 is what actually schedules it, for real, via a new Arq cron job (`cache_maintenance_worker.py`, `WorkerSettings.cron_jobs`, firing hourly plus once at worker startup) — see `docs/DECISION_LOG.md`'s 2026-10-20 entry for the full reasoning (why Arq cron specifically, why a new dedicated worker module, and the honestly-stated operational caveat that no Arq worker in this codebase runs as a persistent container yet).

### Cache telemetry: `cache_lookups_total{cache, result}`

Every cache module calls the one shared OTel counter `cache.get_cache_lookups_counter()` (`cache` in `{embedding, retrieval, response, semantic}`, `result` in `{hit, miss, error}` — `error` covers a real Redis/Postgres outage degrade, which every cache treats as "fall through to the real computation," never a hard failure). Exposed at `apps/api`'s own `/metrics` endpoint, scraped by the already-provisioned `resolvegrid-api` Prometheus job (`infra/prometheus/prometheus.yml`), confirmed to export under the EXACT name `cache_lookups_total` with no suffix mangling (checked directly against a real scrape, unlike Phase 11 Task 6's `otelcol_*` naming gotcha — not assumed this time either).

A new Grafana dashboard, `infra/grafana/dashboards/cache-overview.json` (provisioned the same file-based way as Phase 11 Task 6's `litellm-overview.json`, not hand-clicked), shows: total lookups, overall hit rate, a 5m lookup-rate stat, a real-outage counter, per-cache-kind hit-rate-over-time (`sum(rate(cache_lookups_total{result="hit"}[5m])) by (cache) / sum(rate(cache_lookups_total{result=~"hit|miss"}[5m])) by (cache)`), per-cache-kind/result raw lookup rate, and stacked total volume by cache kind. Verified against real traffic (not just JSON-valid): two real `/chat` calls with the identical message ("What is a VPN?") produced a real response-cache hit, and a third, differently-worded-but-equivalent message ("Can you explain what a VPN is?") produced a real semantic-cache hit — all three returned instantly with the identical cached answer. Grafana's own datasource-proxy API (`GET /api/datasources/proxy/uid/prometheus/api/v1/query`, the same verification method as Phase 11 Task 6) confirmed every new panel's query returns real, non-empty series from this traffic (e.g. `sum(cache_lookups_total)` = 13, overall hit rate ≈ 0.385, 6 real `(cache, result)` series including `embedding`/`hit` and `semantic`/`hit`), not "No data."

### Latency-saved: documented SQL, not a live panel (and why)

`infra/grafana/provisioning/datasources/` provisions exactly one datasource, Prometheus — no Postgres datasource exists for Grafana to query (checked before deciding, per this task's own instructions). The real "cache-hit vs cache-miss latency" comparison needs `Span`/`AgentRun` rows, which live in Postgres, not Prometheus — `cache_lookups_total` only tells you a lookup happened, not how long the SURROUNDING request took. Rather than standing up a new Grafana Postgres datasource for one comparison (a real but disproportionate infra change for this task), the comparison is documented here as a real, directly-runnable SQL query against `apps/api`'s own database:

```sql
-- Total per-chat-turn latency (sum of all 4 Span rows per AgentRun),
-- cache hit vs cache miss. A cache-hit AgentRun's 4 Spans carry the real
-- (near-zero) cache-lookup time on the first stage and honest 0ms on the
-- other three (routers/chat.py never runs those stages on a hit); a
-- cache-miss AgentRun's 4 Spans carry the real per-node graph timing.
SELECT
    agent_run.cache_hit,
    agent_run.cache_hit_kind,
    count(*) AS runs,
    avg(turn_latency.total_latency_ms) AS avg_latency_ms,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY turn_latency.total_latency_ms) AS p50_latency_ms
FROM agent_run
JOIN (
    SELECT agent_run_id, sum(latency_ms) AS total_latency_ms
    FROM span
    GROUP BY agent_run_id
) AS turn_latency ON turn_latency.agent_run_id = agent_run.id
GROUP BY agent_run.cache_hit, agent_run.cache_hit_kind
ORDER BY agent_run.cache_hit, agent_run.cache_hit_kind;
```

Run for real against the dev DB as part of this task's own verification (which includes this phase's accumulated real `/chat` traffic, not only this task's own 3 calls), this returned:

| `cache_hit` | `cache_hit_kind` | `runs` | `avg_latency_ms` | `p50_latency_ms` |
|---|---|---|---|---|
| `false` | `NULL` | 11 | 41,574.9 | 15,837.0 |
| `true` | `semantic` | 4 | 185.5 | 165.5 |
| `true` | `NULL` (exact-cache hit; `cache_hit_kind` is only ever set to `"exact"`/`"semantic"` on rows written after Task 5 added that column — see `models/agent.py`'s `cache_hit_kind` docstring for why a `NULL` kind alongside `cache_hit=true` is a valid, older-row state, not a bug) | 1 | 0.0 | 0.0 |

A real cache-miss `/chat` turn (full graph: `classify_intent` + retrieval + a real LLM `compose_response` call + `verify_citations`) costs ~41.6 seconds on average on this dev machine's local Ollama; a real cache hit (semantic or exact) costs under 200ms — roughly **two orders of magnitude faster**, the real, measured latency-saved proof this task's own brief asks for, not an estimate. If a Postgres datasource is ever provisioned for Grafana in a future phase, this exact query (as a `table`-type panel) is the natural first candidate to promote into a live panel.
