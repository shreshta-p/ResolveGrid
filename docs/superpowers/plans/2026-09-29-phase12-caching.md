# Phase 12: Caching Layer

**Renumbering note** (see `docs/DECISION_LOG.md`'s 2026-09-28 entry and
`docs/PROGRESS.md`): this is the approved master plan's original Phase 11
("Caching"), renumbered Phase 12 when Phase 11 was inserted ahead of it for
observability/routing gap-closing. `plan.md` itself is left untouched as the
historical spec; this doc is the as-built plan for what `plan.md`'s
"## Caching" section (lines 117-121) describes. Depends on Phase 11
(Verified, `docs/PROGRESS.md`).

**Standing directive governing this phase, verbatim from the human operator
after Phase 11 closed**: "nothing mocked, this is going to be a properly
working project." Every cache mechanism below must be proven against real,
running infrastructure (real Redis reads/writes, real pgvector similarity
search, real Anthropic API responses showing real `cache_read_input_tokens`)
— never a code path that merely "looks like" caching without a cited,
reproduced measurement proving it actually reduces real work.

**Credential-handling constraint (standing, zero-tolerance, from
`feedback_credential_exposure_never_read_env.md`)**: no task in this phase
touches new secrets, but if any task needs to inspect `.env`, it must do so
exclusively via `grep`/shell-variable-expansion, never the Read tool, never
a command whose output could print a resolved secret.

**CI-vs-local proof split (established Phase 11, restated here)**:
`.github/workflows/ci.yml` passes empty `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`
— CI never has real cloud credentials. Task 4 (real Anthropic prompt
caching) cannot be proven inside the automated CI suite; it must be proven
via a real local verification step with cited evidence (actual
`cache_creation_input_tokens`/`cache_read_input_tokens` values from real
Anthropic responses), exactly as Phase 11's routing/budget work was.

## Scope decisions (deliberate, not oversights)

The master plan's "Caching" section names seven cache *kinds*: exact-response,
semantic-response, retrieval, embedding, reranker, safe read-only tool, and
provider prompt-prefix. This phase builds the five that are real, distinct,
and valuable given this codebase's actual architecture and traffic patterns:
embedding, retrieval, exact-response, provider prompt-prefix, and
semantic-response. **Reranker caching and tool-result caching are deliberately
deferred**, for reasons to record in `docs/DECISION_LOG.md` at Task 6:
- The reranker (`BAAI/bge-reranker-base`) only ever scores the handful of
  chunks a single retrieval call already fetched — there is no cross-call
  reuse opportunity distinct from the retrieval cache itself (Task 2) already
  covers, since caching the *retrieval result* makes the reranker call
  disappear entirely on a cache hit.
- The one real mutating tool (`grant_vpn_access`) must never be cached (a
  mutation cache-hit would silently skip a real side effect); the one
  read-only tool (`lookup_employee_entitlements`) is a single indexed SQL
  query already fast enough (sub-millisecond) that a cache layer would add
  authz-safety risk (same cross-user-leakage surface as every other cache
  here) for no measurable latency win. If a future phase adds a slower
  read-only tool, this decision should be revisited.

## Cross-cutting design (read before Task 1)

**Cache backend**: the existing shared `redis` service (`infra/docker-compose.yml`,
port 6380, already used for Arq job queues) — not a new Redis instance. This
mirrors the Langfuse-reuse-vs-new-instance decision from Phase 11 Task 5 in
reverse: here reuse is correct because cache keys are namespaced
(`cache:embedding:`, `cache:retrieval:`, `cache:response:`, `cache:semantic:`)
and cached data has no confidentiality requirement stricter than what's
already in Postgres, unlike Langfuse's need for dedicated storage/eviction
tuning. A new `redis.asyncio`/`redis` Python client dependency will need
adding to `apps/api`'s `pyproject.toml` (currently no direct Redis client is
used from `apps/api` itself — Arq wraps its own connection).

**Authorization-scope signature (the security-critical piece — read
carefully)**: the master plan requires "cache keys and validation must
account for authorization scope" and "test cross-user leakage." Per
`retrieval_authz.py`'s `AuthzFilter(unrestricted: bool, allowed_tags: set[str])`,
define a canonical, deterministic **scope signature**:

```
scope_sig = "unrestricted" if authz_filter.unrestricted
            else "tags:" + ",".join(sorted(authz_filter.allowed_tags))
```

This is what the master plan calls "role-scoped, department-scoped" sharing:
two different employees with the identical resolved `AuthzFilter` (e.g. two
analysts in the same department) MUST hit the same cache entry (that's the
whole efficiency point of role/department-scoped caching, not a bug); two
employees with different resolved scopes MUST NEVER share an entry. Every
cache key in this phase that touches retrieval-authorized content (Tasks 2,
3, 5) includes `scope_sig`, never the raw `principal.employee_id` alone
(that would be correct but defeat the sharing the master plan asks for) and
never omit it (that would leak across authz boundaries). The employee-level
cache (Task 1, embeddings of raw text) has no authz dimension at all — an
embedding vector of a fixed string is the same for every caller and reveals
nothing an unauthorized party couldn't already produce themselves by
embedding the same public string, so it is globally shareable, unscoped.

**KB version stamp (real event-invalidation trigger)**: `IngestionRun` already
exists and gets a new completed row every time the corpus changes (Phase 7).
Reuse `select(func.max(IngestionRun.id)).where(IngestionRun.status ==
"completed")` as `kb_version` — a real, already-existing signal, not a new
schema column. Every retrieval/response/semantic cache key (Tasks 2, 3, 5)
includes `kb_version`, so re-ingesting the corpus organically invalidates
every prior entry (old keys simply stop being looked up; they expire via TTL
rather than requiring an explicit delete pass, which is a legitimate,
documented "stale handling" strategy the master plan explicitly allows for).

**Entitlement/role-change staleness (documented, bounded gap)**: a scope
signature is only recomputed when `build_authz_filter` runs again per-request
— if an admin revokes a role grant, any cache entry keyed by that employee's
*old* scope signature remains servable to them until it naturally expires by
TTL. Real event-driven invalidation on every entitlement/role mutation
(`grant_vpn_access`, future admin role-assignment endpoints) is out of scope
for this phase — record this honestly in `docs/DECISION_LOG.md` at Task 6
with the chosen TTL as the bound (see Task 3), matching this project's
established pattern of shipping a real, bounded mechanism plus an honest gap
note rather than either silently ignoring it or over-building a full
event bus for a single mutation path.

**Cache telemetry (real, not fabricated)**: every cache lookup (hit or miss)
increments a real Prometheus counter via the existing `otel-collector` ->
Prometheus pipeline (Phase 11 Task 6 precedent) — `cache_lookups_total{cache,
result}` (`cache` in `{embedding, retrieval, response, semantic}`, `result` in
`{hit, miss}`). This is the same OTel-metrics pattern already proven working
for LiteLLM's own metrics; a new Grafana panel set is added in Task 6.

## Tasks

### Task 1: Embedding cache

Add `apps/api/src/resolvegrid_api/cache.py` (new module) with a real
Redis-backed `get_or_compute_embedding(text: str, model: str, redis_client) ->
list[float]` — key `cache:embedding:{model}:{sha256(text)}`, value the
JSON-serialized embedding vector, real TTL (define one, e.g. 7 days — the
embedding of stable text doesn't go stale the way retrieved KB content does,
so a long TTL is correct, not arbitrary). Wire this into
`services/retrieval/src/resolvegrid_retrieval/embedder.py`'s real call sites
(both query-time embedding in `retrieval.py` and ingestion-time embedding in
`ingestion_worker.py`) — respect the existing dependency-direction rule:
`services/retrieval` must not import `resolvegrid_api`/Redis directly, so the
cache lookup itself lives in `apps/api` (mirroring how `services/retrieval`
already takes a plain injected embed function, if it does — check
`embedder.py`'s actual signature first; if it's a bare function `apps/api`
calls directly rather than a DI closure, wrap it at the `apps/api` call site
instead of threading a new parameter through the library).

A real FastAPI dependency provides the Redis client (`redis.Redis.from_url`,
matching `REDIS_URL` from `.env.example`), following the existing
`Depends(get_db)` pattern's shape for a new `Depends(get_redis)`.

**Real verification required**: call the same query text twice against a
running stack; the second call must be measurably faster (embedding call
skipped entirely) AND `cache_lookups_total{cache="embedding",result="hit"}`
must have incremented by exactly 1 — checked via a real Prometheus query,
not inferred from application logs.

### Task 2: Retrieval-result cache + cross-user leakage test

Cache the full hybrid-search-plus-rerank-plus-dedup-plus-budget result
(whatever `retrieve_for_agent` in `apps/api` currently returns to the graph —
read that function first) keyed by
`cache:retrieval:{kb_version}:{scope_sig}:{sha256(query_text)}`, real TTL
(shorter than the embedding cache's — retrieved *rankings* can shift as
status/rerank logic evolves even without a KB version bump; a TTL on the
order of hours, not days, is the honest choice — document the reasoning,
don't just pick a number).

**Mandatory security test (master plan: "test cross-user leakage")**: a real,
non-mocked test that seeds two employees with genuinely different resolved
`AuthzFilter`s (e.g. different departments, matching the existing
Phase 7/8 walkthrough precedent of using two real differently-scoped
employees), issues the identical query text from both, and asserts the
second employee's retrieval result is INDEPENDENTLY computed (or at minimum
never contains chunks the first employee's scope wouldn't have allowed) —
this is the single most important test in this phase and must not be
satisfied by mocking the cache away.

### Task 3: Exact-response cache for `/chat`

**Amendment (2026-09-30, made during Task 3's own implementation — see
`docs/DECISION_LOG.md`'s entry of the same date for the formal record)**:
this section originally specified the key shape
`cache:response:{kb_version}:{scope_sig}:{risk_level}:
{prompt_template_version}:{sha256(message_text)}`. That is wrong as written
and was corrected before implementation, not followed blindly:
`risk_level` is not known until AFTER `classify_intent` runs INSIDE the
graph — it is an *output* of classification, not an input the caller has
before the graph starts. Including it in the cache KEY would make a cache
lookup depend on having already run the one node a cache hit is supposed to
let this endpoint skip entirely, directly contradicting this section's own
very next requirement ("a cache hit must skip the ENTIRE graph run").

**Corrected key shape** (what was actually implemented):
`cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:
{sha256(message_text)}` — `risk_level` removed from the key. The classified
`intent`/`risk_level` are instead stored INSIDE the cached VALUE payload
(alongside `answer`/`citations`/`caption`), so a cache hit can still
honestly report/log them (e.g. on the cache-hit `ModelCall` row's derived
`model`/`routing_reason`) without ever having needed them to perform the
lookup. This is safe: `risk_level` only affects which model TIER answers a
NEW question (a routing decision), never what content is exposed to whom
(that is `scope_sig`'s job, unchanged) — a cached answer's content is
already fixed regardless of which model tier originally produced it, and
re-classifying the identical `(kb_version, scope_sig, message_text)` inputs
would deterministically-enough reach the same classification anyway (any
residual classifier variance here is an already-accepted, pre-existing
characteristic of this system — see Phase 6's `classify_intent` docstring —
not a new risk this cache introduces).

Read `chat.py`/`graph.py` first to confirm exactly what varies the compose
prompt and add a real `PROMPT_TEMPLATE_VERSION` constant next to
`_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE` (or wherever the compose prompt
template actually lives) so a future prompt-wording change organically
busts this cache instead of silently serving stale-worded answers forever.
Real TTL — this is the entitlement-staleness bound named in the cross-cutting
section above, so choose it deliberately short (e.g. 15-30 minutes) and say
so explicitly in this task's commit/docs.

A cache hit must skip the ENTIRE graph run (no `classify_intent`/`retrieve`/
`compose_response`/`verify_citations` node executes) but must still write a
real `AgentRun`/`Span` row set so telemetry/cost dashboards don't silently
under-count real traffic — a cache-hit `Span` row's `latency_ms` reflects the
real (near-zero) cache-lookup time, not a fabricated placeholder, and its
`ModelCall` accounting must show `$0` cost with a routing_reason/purpose that
honestly says "cache_hit", not omit the row entirely (the master plan asks
for "cache telemetry" and "provenance" — a vanished row would be the opposite
of that).

**Real verification required**: the identical `/chat` message from the
identical principal twice in a row — second call must be dramatically faster
and produce zero new `ModelCall` rows (cost $0), while still producing a real
new `AgentRun`/`Span` row set marked as a cache hit.

### Task 4: Real Anthropic prompt-prefix caching

This is a genuinely different mechanism from Tasks 1-3/5 (provider-level
KV-cache reuse inside Anthropic's own infrastructure, not this application's
own cache store) and the one most literally matching "prompt caching" as an
AI-engineering term of art. Read `llm_gateway.py`'s `complete()` (single flat
`prompt: str` today, sent as one `user` message) before starting — this task
must restructure it to support an optional stable, cacheable prefix (e.g. a
`system` message carrying `compose_response`'s fixed instruction/persona text,
identical across every real call) separately from the variable
question/context suffix, and mark the stable block with Anthropic's real
`cache_control: {"type": "ephemeral"}` content-block annotation (only ever
sent when `provider_for_model(model) == "anthropic"` — Ollama/OpenAI callers
must be completely unaffected, mirroring the existing `_THINK_FALSE_MODELS`
per-model-conditional precedent exactly).

Capture the real `cache_creation_input_tokens`/`cache_read_input_tokens`
fields LiteLLM passes through from Anthropic's response `usage` object into
`CompletionResult` (new fields) and from there into `ModelCall` (new
migration, mirroring `trace_id`'s Phase 11 Task 1 precedent) so real cache
utilization is queryable telemetry, not a one-off manual observation.

**Real verification required, and required to be reported HONESTLY either
way**: make two real, back-to-back `cloud-primary` calls sharing the
identical stable prefix and confirm via the real Anthropic response whether
`cache_read_input_tokens > 0` on the second call. Anthropic enforces a real
minimum token count per cacheable block (verify the actual current number
against Anthropic's real API docs/response, do not assume a number) — if
this codebase's real stable prefix text is too short to cross that
threshold, that is real, reportable information: either grow the stable
prefix with genuinely useful additional real instruction content until it
legitimately crosses the threshold, or document the honest limitation in
`docs/TELEMETRY_COST.md` exactly as Phase 11's honest CI-hardware finding was
documented — never fabricate a passing number.

### Task 5: Semantic-response cache

A response cache hit not just on exact text match (Task 3) but on semantic
similarity to a previously-cached question within the same `scope_sig` +
`kb_version`. Use the same `embedding` infrastructure Phase 7 already
established (pgvector, `nomic-embed-text`) — add a new small table (new
migration) storing `(scope_sig, kb_version, query_embedding vector,
cached_response_id/text, created_at)` rows, or reuse Task 3's Redis cache
entries with a parallel pgvector index of their query embeddings (choose
whichever is the smaller real diff against this codebase's existing
patterns — Postgres+pgvector already exists and is this project's
established vector-search mechanism, so prefer extending it over introducing
vector search inside Redis).

Define a real, justified similarity threshold (cosine distance) — do not
pick an arbitrary round number without checking it against a few real
example near-duplicate questions (e.g. "What is a firewall?" vs. "Can you
explain what a firewall is?") and a few real example DIFFERENT questions
that must NOT collide, exactly the same rigor Phase 7's retrieval eval
harness already applied to its own recall/precision tuning.

**Real verification required**: two genuinely differently-worded but
semantically equivalent real questions from the same principal — second
must hit the semantic cache (real telemetry counter increments, zero new
`ModelCall` rows); a real semantically-different question must NOT hit it.

### Task 6: Invalidation wiring, cache telemetry dashboard, decision-log entries

- Confirm (with a real test, not just reasoning) that re-running
  `ingestion_worker` bumps `kb_version` and that a previously-cached
  retrieval/response/semantic entry keyed by the old `kb_version` is no
  longer looked up (still present in Redis until its TTL, per the
  stale-handling strategy above — that's fine, it's simply never read again
  under the new key).
- Add a new Grafana dashboard panel set (new JSON, following Task 6 of
  Phase 11's `litellm-overview.json` precedent) — cache hit-rate per cache
  kind, real latency saved (compare cache-hit vs cache-miss `Span`/
  `ModelCall` latencies already recorded).
- `docs/DECISION_LOG.md`: record the reranker/tool-cache deferral (see Scope
  decisions above), the entitlement-staleness TTL-bound gap, and the exact
  chosen TTLs with rationale for each cache kind.
- `docs/ARCHITECTURE.md`/`docs/TELEMETRY_COST.md`: document the full caching
  architecture as-built (scope-signature mechanism, kb_version mechanism,
  each cache's key shape and TTL) — the master plan explicitly requires
  "provenance" and this is where a future reader finds it.

### Task 7: Fresh-state verification, push, CI confirmation

Same closing-task shape as every prior phase: `docker compose ... down -v`
(confirm exactly which named volumes exist before teardown, matching Phase
10/11's precedent of checking nothing outside this project's own volumes is
touched), full migration chain from empty, full test suite BEFORE any
seeding/ingestion (per the established empty-corpus-first constraint), then
seed/ingest, then a real manual walkthrough (Playwright MCP) proving at least
one real cache hit end-to-end through the actual browser UI (not just via
direct API/script calls), plus the cross-user-leakage test's own real
two-employee proof re-run against this fresh stack. Push to `origin/master`,
poll real CI via `gh run view` directly (do not trust a subagent's
"waiting for notification" claim — run it as a blocking foreground call, per
this session's established recovery pattern), fix any real CI failure with a
root-caused fix (not a weakened assertion for its own sake), confirm all jobs
green, then update `docs/PROGRESS.md` with the same evidentiary density as
every prior phase's row.
