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
