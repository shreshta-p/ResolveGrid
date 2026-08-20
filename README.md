# ResolveGrid

Internal IT service-management + AI-ops platform for the fictional company **Kestrel Softworks, Inc.** — a local-first, production-shaped agentic RAG system with role-aware surfaces, durable human-in-the-loop approvals, and real observability/cost accounting. See `C:\Dev\ResolveGrid\plan.md` for the full specification and `docs/PROGRESS.md` for current build status.

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 (App Router) / TypeScript |
| Backend | FastAPI (Python 3.13) |
| Agent orchestration | LangGraph (added in a later phase) |
| Data | PostgreSQL + pgvector, Redis |
| Local inference | Ollama (GPU-accelerated) |
| Model gateway | LiteLLM proxy (added in a later phase) |
| Observability | OpenTelemetry Collector, Langfuse, Prometheus/Grafana (added in a later phase) |

## Quick start

Prerequisites: Docker Desktop, Python 3.13 + `uv`, Node 22 (pnpm via `corepack enable`).

```bash
# 1. Bring up infra (Postgres+pgvector, Redis, Ollama, OTel Collector)
docker compose -f infra/docker-compose.yml up -d

# 2. Python API
uv sync
uv run --package resolvegrid-api alembic -c apps/api/alembic.ini upgrade head
uv run --package resolvegrid-api uvicorn resolvegrid_api.main:app --reload --app-dir apps/api/src

# 3. Web frontend (separate terminal)
corepack enable
pnpm install
pnpm --filter @resolvegrid/web dev
```

Then check `http://localhost:8000/health` and `http://localhost:3000`.

## Repo map

- `apps/api` — FastAPI backend
- `apps/web` — Next.js frontend
- `packages/telemetry` — shared OpenTelemetry instrumentation
- `infra/` — Docker Compose + service configs
- `docs/` — canonical documentation set; `docs/PROGRESS.md` is the single authoritative status ledger
- `eval/` — golden dataset, adversarial cases, load-test workloads (added in a later phase)

## Commands

See `Makefile` for the full list (`make help`).
