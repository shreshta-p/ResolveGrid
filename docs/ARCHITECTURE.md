# Architecture

System topology: modular monolith (`apps/api` FastAPI + `apps/web` Next.js) with LangGraph as the sole top-level agent orchestrator, one bounded in-process Google ADK specialist, no A2A in v1. Full rationale: `docs/adr/0001-modular-monolith-topology.md` and the approved architecture plan referenced in `CLAUDE.md`.

## Current topology (Phase 1)
- `apps/api` — FastAPI, `/health` only so far.
- `apps/web` — Next.js placeholder.
- `packages/telemetry` — OTel tracing helper, wired into API startup.
- `infra/` — Postgres+pgvector, Redis, GPU Ollama, OTel Collector (debug exporter only, no backend yet).

See `docs/PROGRESS.md` for what's actually built vs. planned.

## Observability stack (Phase 11) — Langfuse/Prometheus/Grafana are REAL, not deferred

`docs/DECISION_LOG.md`'s 2026-08-24 entry originally deferred standing up Langfuse and
Prometheus/Grafana, with a stated revisit trigger: "any later phase that specifically needs
Langfuse's trace-debugging UI or Prometheus/Grafana's infra-metrics dashboards." Phase 11 was that
trigger, and both are now real, running infra, not aspirational:

- **OTel Collector** (`infra/otel-collector/config.yaml`) exports real trace data to **Langfuse**
  (self-hosted v4: `langfuse-web`/`langfuse-worker`, dedicated ClickHouse + MinIO + Redis
  containers, Postgres reused via its own `langfuse` database inside the shared
  `resolvegrid-postgres` container) via Langfuse's real OTLP-over-HTTP ingestion endpoint
  (`/api/public/otel`). A real trace from a real `/tickets/{id}/summarize` (or `/chat`) call is
  confirmed to land and be queryable — see `docs/TELEMETRY_COST.md`'s "Langfuse" section for the
  correct v4 query endpoint (`/api/public/v2/observations?traceId=...`, not the v3-shaped
  `/api/public/traces/{id}`, which 404s under v4's `events_only` mode).
- **Prometheus** (`infra/prometheus/`) scrapes LiteLLM's real `/metrics` endpoint plus the OTel
  Collector's own internal pipeline-health metrics. **Grafana** (`infra/grafana/`) provisions a real
  dashboard (not manual UI clicking) showing real request-count/latency data from actual LiteLLM
  traffic — confirmed non-zero, non-mock via Grafana's own datasource-proxy API.
- Both stacks are local-infra-only: CI never provisions either (no cloud/Langfuse/Grafana secrets
  are set up there), so their real-data proofs are documented as real LOCAL verification evidence,
  not automated CI tests — see `docs/superpowers/plans/2026-09-27-phase11-observability-routing.md`'s
  "Notes" section for why, and `docs/PROGRESS.md`'s Phase 11 row for the cited evidence.

Full rationale for each infra choice (why ClickHouse/MinIO are new dedicated containers, why
Langfuse's Redis is NOT the shared `redis` service, why LiteLLM's `/metrics` auth is disabled
rather than templated) is in `docs/DECISION_LOG.md`'s 2026-09-28 entries.

## Caching layer (Phase 12)

Five real caches sit in front of embedding/retrieval/response generation: an unscoped Redis
embedding cache (`cache.py`), and four Redis/Postgres caches scoped by a shared `scope_sig`
(authz-scope signature) + `kb_version` (real `IngestionRun`-derived corpus-version stamp) pair —
retrieval-result (`retrieval_cache.py`), exact-response (`response_cache.py`), semantic-response
(`semantic_cache.py`, Postgres/pgvector), and Anthropic's own provider-side prompt-prefix cache
(`llm_gateway.py`, a real but currently-inactive mechanism — see below). Every cache emits to one
shared `cache_lookups_total{cache, result}` OTel counter, visualized in a new provisioned Grafana
dashboard (`infra/grafana/dashboards/cache-overview.json`). Full key shapes, TTLs, the
`scope_sig`/`kb_version` invalidation mechanism, the semantic-cache purge job's real Arq-cron
scheduling, and Task 4's honest "Anthropic's real cache-minimum exceeds this codebase's real stable
prefix" negative result are all documented in `docs/TELEMETRY_COST.md`'s "Phase 12: caching layer
architecture, as-built" section — not duplicated here. Scope decisions (reranker/tool-result
caching deliberately left out) and the accepted entitlement-staleness TTL bound are in
`docs/DECISION_LOG.md`'s 2026-10-20 entries.

## ADK diagnostic specialist (Phase 13)

A real Google ADK (`google-adk[extensions]==1.14.1`) `Agent`/`Runner` sits inside `apps/api`,
powering an analyst-only "Diagnose" action on the ticket detail page. Full empirical findings
behind every decision below are in `docs/adr/0003-adk-real-integration-findings.md` (Task 1 spike),
`docs/adr/0004-adk-diagnostic-specialist-wiring.md` (Task 3/4 build), and
`docs/adr/0005-diagnostic-eval-integration-depth.md` (Task 5 evals) — this section summarizes the
as-built shape, not duplicates their detail.

### No LangGraph node, no A2A — ADK is an opaque, directly-called specialist

Per the master plan's explicit "do not let LangGraph and ADK own the same orchestration layer":
LangGraph continues to own every other agent interaction in this system (the `/chat` pipeline, the
tool-invocation/approval graph); the diagnostic specialist is reached by `POST
/tickets/{id}/diagnose` calling `resolvegrid_api.diagnostic_agent.diagnose_ticket(session, tracer,
principal, ticket_id)` **directly** — no new LangGraph node, no new compiled graph. This was a real
Task 4 decision, not an oversight: ADK's own `Runner` already owns its entire multi-turn
tool-calling loop as one opaque, synchronous-from-the-caller's-perspective unit, so there is no
pause/resume boundary for a graph/checkpointer to add value around, and the real
`session`/`tracer`/`principal` signature doesn't fit `AgentState`'s checkpoint-safe, primitive-only
contract without inventing per-request plumbing for zero benefit. A `DiagnosisFn`/`DiagnosisOutcome`
primitive-typed contract was still added to `services/agent-orchestration` as a ready (currently
unused) seam, in case a future phase needs diagnosis reachable from the general chat graph. See
`docs/DECISION_LOG.md`'s 2026-10-23 entry and ADR 0004's "Task 4 addendum."

A2A (agent-to-agent protocol) is deliberately not used anywhere in this integration. Real research
against ADK's current docs (`adk.dev`) and corroborating secondary sources, performed before this
phase's plan was written and re-affirmed once real code existed (`docs/DECISION_LOG.md`'s
2026-10-21 entry), found A2A is meant for genuinely separate, independently-deployed agent services
— cross-system/cross-organization handoffs between agents on different machines or frameworks — not
a protocol for wrapping an in-process function call in a network boundary. The diagnostic specialist
runs in the same `apps/api` process as everything else in this codebase; there is no second,
independently-meaningful agent service for it to hand off to.

### Model routing: `LiteLlm` through the existing proxy, never a parallel gateway

The specialist's model (`google.adk.models.lite_llm.LiteLlm`) points at this project's own
self-hosted LiteLLM proxy (`infra/litellm/config.yaml`'s `local-qwen3`/`cloud-primary` model
groups) — never a disconnected Gemini-only path. A ticket's own `priority` maps to the existing
`routing.select_model_for_risk_level` policy (the same one `compose_response` already uses), so a
high-priority ticket's diagnosis can route to `cloud-primary` while the common case stays on the
zero-marginal-cost `local-qwen3` default. Two real, qwen3-specific tuning requirements are mandatory
on every call, not optional: `think=False` (without it, a real tool-calling call to `local-qwen3`
returns a bare `"{}"` instead of invoking a tool — ADR 0003 Finding 4) and
`_skip_mcp_handler=True` (works around a real `google-adk`/`litellm` dependency conflict — ADR 0003
Finding 3, `docs/DECISION_LOG.md`'s 2026-10-21 entry). `RunConfig.max_llm_calls=60` bounds a real,
confirmed qwen3 tendency to re-invoke an already-answered tool several times before answering (up to
36 observed turns for a single tool — ADR 0003 Finding 5). Every real internal model call is logged
as its own `ModelCall` row (`purpose="diagnostic.specialist_turn"`) via ADK's real
`BasePlugin.after_model_callback` hook, feeding `model_call_logging.log_adk_model_call` — the same
cost/telemetry ledger every other agent interaction in this system writes to, not a parallel
mechanism.

### Four real diagnostic tools, authz-scoped before the model ever sees them

`apps/api/src/resolvegrid_api/diagnostic_tools.py` defines four plain Python functions (ADK's own
tool convention), each authz-checked against the requesting analyst's department scope using this
codebase's established tools: search similar/related past tickets (reusing retrieval infra),
`lookup_employee_entitlements` (Phase 9's existing read-only tool, reused directly, not duplicated),
`retrieve_ticket_kb_evidence` (reusing the Phase 7/8 retrieval pipeline, so citations are real
`[chunk:<id>]` references verified via `resolvegrid_retrieval.citation_verification`, the same
function `/chat`'s own citation-verification node uses), and the ticket's own current state/history
(`AuditLog`-backed). Each is curried into a per-request closure (not a bare `functools.partial`,
which collides on `__name__`/loses `__doc__` across all four tools — ADR 0004 Finding 1) binding the
live `Session`/`Principal`, since ADK has no generic mechanism to inject a caller-supplied object
without exposing it to the model. The specialist's own system instruction explicitly states it may
only **suggest** a mutating action — it must never invoke `grant_vpn_access` itself; any mutating
suggestion routes back through the existing, already-reviewed
`POST /tools/grant_vpn_access/invoke` human-in-the-loop approval path. `grant_vpn_access` is not
registered as a tool the specialist can call at all (a structural guarantee, not just a prompt
instruction), proven by a dedicated unit test and the real prompt-injection adversarial test below.

### The judge-based safety check

The specialist's one safety-critical property — it never claims a mutating action was completed,
even under adversarial prompting — is checked by a real, calibrated judge classifier
(`resolvegrid_api.eval_judge.classify_mutation_completion_claim_real`), not a regex. This replaced a
regex-based grader (`grade_no_mutation_claim_in_text`) after three real rounds of adversarial
probing against the live model each found it wrong in a different way (missed third-person/passive
completion claims, then false-positived on legitimate policy-document phrasing, then missed further
completion verbs and false-flagged real negations/conditionals) — confirming "does this text claim
completion" is a genuinely semantic question a fixed pattern list cannot reliably answer, not a
harness bug to keep patching. The judge was calibrated at 93.8% exact agreement (15/16) against a
hand-labeled set covering every adversarial string found across all three probing rounds. The regex
is kept, demoted to a cheap, non-gating, always-logged defense-in-depth signal — any real
disagreement between it and the judge is now automatically logged, closing an earlier forensic gap
where a flagged-but-unreproduced case could not be investigated. Full history:
`docs/adr/0005-diagnostic-eval-integration-depth.md`'s Amendments 2 and 3.
