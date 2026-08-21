# Architecture

System topology: modular monolith (`apps/api` FastAPI + `apps/web` Next.js) with LangGraph as the sole top-level agent orchestrator, one bounded in-process Google ADK specialist, no A2A in v1. Full rationale: `docs/adr/0001-modular-monolith-topology.md` and the approved architecture plan referenced in `CLAUDE.md`.

## Current topology (Phase 1)
- `apps/api` — FastAPI, `/health` only so far.
- `apps/web` — Next.js placeholder.
- `packages/telemetry` — OTel tracing helper, wired into API startup.
- `infra/` — Postgres+pgvector, Redis, GPU Ollama, OTel Collector (debug exporter only, no backend yet).

See `docs/PROGRESS.md` for what's actually built vs. planned.
