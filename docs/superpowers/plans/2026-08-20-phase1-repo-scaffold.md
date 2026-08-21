# Phase 1 — Repo Scaffold & Local Infra Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the empty-but-wired ResolveGrid monorepo — Python (uv) and Node (pnpm) workspaces, a FastAPI health endpoint, a Next.js placeholder page, Docker Compose infra (Postgres+pgvector, Redis, GPU-enabled Ollama, OTel Collector), a baseline Alembic migration, CI skeleton, and the canonical docs/progress-ledger skeleton — with zero product features.

**Architecture:** Modular monolith per the approved architecture plan (`C:\Users\shres\.claude\plans\follow-the-instructions-in-snazzy-sutton.md`, §3 Decision A/B): `apps/api` (FastAPI) and `apps/web` (Next.js) as the two deployables, `packages/telemetry` as the first shared library (OTel instrumentation), everything else deferred to later phases. uv workspace for Python, pnpm workspace for Node, plain `Makefile` for cross-cutting commands (no `just` — not installed on this machine, and the architecture plan already names Makefile as the accepted alternative).

**Tech Stack:** Python 3.13 + uv 0.8, FastAPI + Uvicorn, OpenTelemetry SDK/OTLP-gRPC exporter, Alembic + psycopg, Node 22 + pnpm (via corepack) + Next.js 16 (App Router) + React 19, Docker Compose v2 (pgvector/pgvector:pg16, redis:7-alpine, ollama/ollama with NVIDIA GPU reservation, otel/opentelemetry-collector-contrib), GitHub Actions CI skeleton.

**Environment facts verified on this machine before writing this plan (do not re-verify, just use):**
- Docker 28.1.1 / Compose v2.35.1, `docker run --gpus all` confirmed working against the host's RTX 4080 (WDDM driver 592.00, CUDA 13.1) — GPU passthrough to containers works, so Ollama can run containerized with a GPU reservation.
- Node v22.18.0, npm 10.9.3, **pnpm is NOT globally installed** but `corepack` v0.33.0 is available (ships with Node) — enable pnpm via corepack, do not `npm install -g pnpm`.
- Python 3.13.7, `uv` 0.8.13 already installed globally.
- `just` is NOT installed — use `Makefile` (GNU Make 3.81 is available via Git Bash).
- git 2.51.0.windows.1 available. `C:\Dev\ResolveGrid` currently has no `.git` — this plan performs the initial `git init`.
- **Host ports 5432 (Postgres) and 6379 (Redis) are already bound by an unrelated running project** (`stocksense-implementation`) on this machine. ResolveGrid's Compose file maps Postgres to host port **5433** and Redis to host port **6380** instead (container-internal ports stay 5432/6379 — only the host-side mapping changes), so both projects can run simultaneously. `.env.example`, `migrations/env.py`'s default fallback, and every documented command below use 5433/6380.
- Current npm registry versions at plan-writing time: `next@16.3.1`, `react@19.2.8` (use caret ranges anchored to these, let pnpm resolve exact current patch at install time).
- All commands below are written for the Bash tool (Git Bash on Windows), matching how this session already operates. Run every command from `C:\Dev\ResolveGrid` unless a step says otherwise.

---

## File Structure

```
C:\Dev\ResolveGrid\
├── .gitignore
├── .env.example
├── README.md
├── CLAUDE.md
├── Makefile
├── pyproject.toml                       # uv workspace root (virtual, package = false)
├── package.json                         # pnpm workspace root
├── pnpm-workspace.yaml
├── apps\
│   ├── api\
│   │   ├── pyproject.toml
│   │   ├── alembic.ini
│   │   ├── migrations\
│   │   │   ├── env.py
│   │   │   ├── script.py.mako
│   │   │   └── versions\
│   │   │       └── 0001_baseline_health_check.py
│   │   ├── src\resolvegrid_api\
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   └── tests\
│   │       ├── __init__.py
│   │       └── test_health.py
│   └── web\
│       ├── package.json
│       ├── tsconfig.json
│       ├── next.config.ts
│       ├── next-env.d.ts
│       ├── .eslintrc.json
│       └── app\
│           ├── layout.tsx
│           └── page.tsx
├── packages\
│   └── telemetry\
│       ├── pyproject.toml
│       ├── src\resolvegrid_telemetry\
│       │   ├── __init__.py
│       │   └── tracing.py
│       └── tests\
│           ├── __init__.py
│           └── test_tracing.py
├── infra\
│   ├── docker-compose.yml
│   └── otel-collector\
│       └── config.yaml
├── scripts\
│   └── smoke_test.sh
├── docs\
│   ├── ARCHITECTURE.md
│   ├── DATA_MODEL.md
│   ├── WORKFLOWS_TOOLS.md
│   ├── RAG_INGESTION.md
│   ├── EVALUATIONS.md
│   ├── TELEMETRY_COST.md
│   ├── SECURITY.md
│   ├── API_CONTRACTS.md
│   ├── DESIGN_SYSTEM.md
│   ├── EXPERIMENT_REGISTRY.md
│   ├── RUNBOOKS.md
│   ├── DECISION_LOG.md
│   ├── PROGRESS.md
│   ├── adr\
│   │   └── 0001-modular-monolith-topology.md
│   └── superpowers\plans\2026-08-20-phase1-repo-scaffold.md   # this file (already exists)
└── .github\workflows\ci.yml
```

Each file has one job: `apps/api` is the only FastAPI process; `apps/web` is the only Next.js process; `packages/telemetry` is a pure library with no side effects at import time (instrumentation is opt-in via `init_tracing()`); `infra/` holds only infrastructure config, no application code.

---

### Task 1: Git init & root scaffolding files

**Files:**
- Create: `C:\Dev\ResolveGrid\.gitignore`
- Create: `C:\Dev\ResolveGrid\.env.example`
- Create: `C:\Dev\ResolveGrid\README.md`
- Create: `C:\Dev\ResolveGrid\CLAUDE.md`

- [ ] **Step 1: Initialize the git repository**

Run: `cd "C:\Dev\ResolveGrid" && git init`
Expected: `Initialized empty Git repository in C:/Dev/ResolveGrid/.git/`

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.uv-cache/

# Node
node_modules/
.next/
out/
.turbo/

# Env
.env
.env.local

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Logs
*.log
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
# --- Postgres (infra/docker-compose.yml) ---
POSTGRES_USER=resolvegrid
POSTGRES_PASSWORD=resolvegrid_dev
POSTGRES_DB=resolvegrid
DATABASE_URL=postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid

# --- Redis ---
REDIS_URL=redis://localhost:6380/0

# --- Ollama ---
OLLAMA_BASE_URL=http://localhost:11434

# --- OpenTelemetry ---
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# --- Cloud LLM providers (LiteLLM proxy config, wired in a later phase) ---
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

- [ ] **Step 4: Create `README.md`**

```markdown
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
uv sync --all-packages
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
```

- [ ] **Step 5: Create `CLAUDE.md`**

```markdown
# CLAUDE.md

Lean index only — durable instructions, invariants, and links. Do not duplicate architecture here; see `docs/`.

## Invariants
- No time estimates in any plan or phase document.
- All employee/ticket/knowledge data is synthetic — never real PII.
- RBAC/authorization is enforced in `packages/authz` (added Phase 2+), never only in the UI.
- `docs/PROGRESS.md` is the single status ledger — do not duplicate state elsewhere.
- Every phase's exit criteria must be objectively verified before marking it Verified in `docs/PROGRESS.md`.

## Critical commands
- `docker compose -f infra/docker-compose.yml up -d` — start local infra
- `uv sync` — install/sync Python workspace
- `pnpm install` — install Node workspace (run `corepack enable` once first)
- `make help` — list all cross-cutting commands

## Canonical docs (read before touching the matching area)
- `docs/ARCHITECTURE.md` — system architecture and ADR index
- `docs/DATA_MODEL.md` — schema shapes
- `docs/WORKFLOWS_TOOLS.md` — agent workflow stages, tool catalog
- `docs/RAG_INGESTION.md` — retrieval/ingestion design
- `docs/EVALUATIONS.md` — golden dataset & grading methodology
- `docs/TELEMETRY_COST.md` — telemetry/cost schema
- `docs/SECURITY.md` — threat model
- `docs/PROGRESS.md` — the single status ledger

## Full specification
`C:\Dev\ResolveGrid\plan.md` — original master spec. `C:\Users\shres\.claude\plans\follow-the-instructions-in-snazzy-sutton.md` — approved architecture/phase plan.
```

- [ ] **Step 6: Verify and make the first commit**

Run: `cd "C:\Dev\ResolveGrid" && git add .gitignore .env.example README.md CLAUDE.md && git status`
Expected: four files staged, nothing else (confirms `.gitignore` isn't needed yet since nothing ignorable exists).

Run: `git commit -m "chore: initialize repo with root docs and env template"`
Expected: commit succeeds on the initial branch.

---

### Task 2: Python workspace root + `apps/api` FastAPI health endpoint (TDD)

**Files:**
- Create: `C:\Dev\ResolveGrid\pyproject.toml`
- Create: `C:\Dev\ResolveGrid\apps\api\pyproject.toml`
- Create: `C:\Dev\ResolveGrid\apps\api\src\resolvegrid_api\__init__.py`
- Create: `C:\Dev\ResolveGrid\apps\api\src\resolvegrid_api\main.py`
- Create: `C:\Dev\ResolveGrid\apps\api\tests\__init__.py`
- Test: `C:\Dev\ResolveGrid\apps\api\tests\test_health.py`

- [ ] **Step 1: Create the uv workspace root `pyproject.toml`**

```toml
[project]
name = "resolvegrid-workspace"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = []

[tool.uv.workspace]
members = ["apps/api"]

[tool.uv]
package = false
```

- [ ] **Step 2: Create `apps/api/pyproject.toml`**

```toml
[project]
name = "resolvegrid-api"
version = "0.1.0"
description = "ResolveGrid FastAPI backend"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "httpx>=0.27",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/resolvegrid_api"]
```

- [ ] **Step 3: Create empty package init and test package init**

`apps/api/src/resolvegrid_api/__init__.py`:
```python
```

`apps/api/tests/__init__.py`:
```python
```

- [ ] **Step 4: Sync the workspace**

Run: `cd "C:\Dev\ResolveGrid" && uv sync`
Expected: creates `.venv/` and `uv.lock`, installs fastapi/uvicorn/pytest/httpx, no errors.

- [ ] **Step 5: Write the failing test**

`apps/api/tests/test_health.py`:
```python
from fastapi.testclient import TestClient

from resolvegrid_api.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run --package resolvegrid-api pytest apps/api/tests/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'resolvegrid_api.main'`

- [ ] **Step 7: Write minimal implementation**

`apps/api/src/resolvegrid_api/main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="ResolveGrid API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run --package resolvegrid-api pytest apps/api/tests/test_health.py -v`
Expected: `1 passed`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock apps/api/pyproject.toml apps/api/src apps/api/tests
git commit -m "feat(api): add FastAPI health endpoint with uv workspace"
```

---

### Task 3: `packages/telemetry` OTel tracing helper (TDD) + wire into `apps/api` startup span

**Files:**
- Create: `C:\Dev\ResolveGrid\packages\telemetry\pyproject.toml`
- Create: `C:\Dev\ResolveGrid\packages\telemetry\src\resolvegrid_telemetry\__init__.py`
- Create: `C:\Dev\ResolveGrid\packages\telemetry\src\resolvegrid_telemetry\tracing.py`
- Test: `C:\Dev\ResolveGrid\packages\telemetry\tests\test_tracing.py`
- Modify: `C:\Dev\ResolveGrid\pyproject.toml` (add workspace member)
- Modify: `C:\Dev\ResolveGrid\apps\api\pyproject.toml` (add dependency)
- Modify: `C:\Dev\ResolveGrid\apps\api\src\resolvegrid_api\main.py` (emit startup span)

- [ ] **Step 1: Create `packages/telemetry/pyproject.toml`**

```toml
[project]
name = "resolvegrid-telemetry"
version = "0.1.0"
description = "Shared OpenTelemetry instrumentation helpers for ResolveGrid"
requires-python = ">=3.13"
dependencies = [
    "opentelemetry-api>=1.27",
    "opentelemetry-sdk>=1.27",
    "opentelemetry-exporter-otlp-proto-grpc>=1.27",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/resolvegrid_telemetry"]
```

- [ ] **Step 2: Register the new workspace member**

Edit `pyproject.toml` (root), change:
```toml
[tool.uv.workspace]
members = ["apps/api"]
```
to:
```toml
[tool.uv.workspace]
members = ["apps/api", "packages/telemetry"]
```

Also add this section anywhere in the root `pyproject.toml` (found during Task 3 code review: both `apps/api/tests/` and `packages/telemetry/tests/` ship a `tests/__init__.py`, which makes both import as the same top-level `tests` module under pytest's default import mode — collection fails if both are ever passed to one `pytest` invocation, e.g. a bare `pytest` run from repo root with no path args):
```toml
[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
```

- [ ] **Step 3: Create test package init and write the failing test**

`packages/telemetry/tests/__init__.py`:
```python
```

`packages/telemetry/tests/test_tracing.py`:
```python
from opentelemetry import trace as otel_trace

from resolvegrid_telemetry import init_tracing


def test_init_tracing_sets_service_name_resource():
    tracer = init_tracing("test-service")
    assert tracer is not None
    provider = otel_trace.get_tracer_provider()
    assert provider.resource.attributes["service.name"] == "test-service"


def test_init_tracing_returns_working_tracer():
    tracer = init_tracing("test-service-2")
    with tracer.start_as_current_span("unit-test-span") as span:
        assert span.is_recording()
```

- [ ] **Step 4: Sync workspace to register the new member**

Run: `cd "C:\Dev\ResolveGrid" && uv sync --all-packages`
Expected: no errors (package has no source yet, so this may fail on missing `src/resolvegrid_telemetry/__init__.py` — if it does, create an empty `__init__.py` first, then re-run).

Run: `mkdir -p packages/telemetry/src/resolvegrid_telemetry && touch packages/telemetry/src/resolvegrid_telemetry/__init__.py`
Run: `uv sync --all-packages`
Expected: succeeds, installs `opentelemetry-*` packages.

Note: `uv sync` alone (no `--all-packages`) will NOT install workspace-member dependencies since the workspace root has `package = false` and empty `dependencies` — always use `--all-packages` (confirmed during Task 2 execution).

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v`
Expected: FAIL with `ImportError: cannot import name 'init_tracing'`

- [ ] **Step 6: Write minimal implementation**

`packages/telemetry/src/resolvegrid_telemetry/tracing.py` (revised after Task 3 code review found the original double-`init_tracing()` call silently discards the second provider — OTel's API only allows the global provider to be set once per process; this version is idempotent):
```python
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(service_name: str) -> trace.Tracer:
    """Configure the global OTel tracer provider for this process and return a Tracer.

    Idempotent: if a real TracerProvider is already set (e.g. an earlier call
    from another module in the same process), returns a Tracer from the
    existing provider instead of silently failing to override it — OTel's API
    only allows the global provider to be set once per process.
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)
```

`packages/telemetry/src/resolvegrid_telemetry/__init__.py`:
```python
from resolvegrid_telemetry.tracing import init_tracing

__all__ = ["init_tracing"]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v`
Expected: `2 passed`

- [ ] **Step 8: Wire `init_tracing` into `apps/api` startup**

Edit `apps/api/pyproject.toml`, change the `dependencies` list to:
```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "resolvegrid-telemetry",
]
```
and add below `[project]`:
```toml
[tool.uv.sources]
resolvegrid-telemetry = { workspace = true }
```

Replace `apps/api/src/resolvegrid_api/main.py` with (revised after code review: `init_tracing()` moved from module import time into the `lifespan` startup phase — calling it at import time meant merely importing this module, e.g. from `test_health.py`, opened a gRPC channel and started a background export thread for a test with nothing to do with tracing; a shutdown call was also added to the teardown phase since `lifespan`'s post-`yield` point is a natural place to flush pending spans):
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from opentelemetry import trace
from resolvegrid_telemetry import init_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    tracer = init_tracing("resolvegrid-api")
    with tracer.start_as_current_span("api.startup"):
        # Placeholder span proving the OTel pipe works end-to-end; real
        # startup instrumentation lands as later phases add real startup work.
        pass
    yield
    trace.get_tracer_provider().shutdown()


app = FastAPI(title="ResolveGrid API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 9: Re-sync and re-run the full Python test suite**

Run: `cd "C:\Dev\ResolveGrid" && uv sync --all-packages && uv run --package resolvegrid-api pytest apps/api/tests -v && uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v`
Expected: all tests still pass (the health test doesn't require a live OTel Collector — `BatchSpanProcessor` buffers/export happens on a background thread and does not block app startup or the test client).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml packages/telemetry apps/api
git commit -m "feat(telemetry): add OTel tracing helper, emit API startup span"
```

---

### Task 4: Docker Compose infra (Postgres+pgvector, Redis, GPU Ollama, OTel Collector)

**Files:**
- Create: `C:\Dev\ResolveGrid\infra\docker-compose.yml`
- Create: `C:\Dev\ResolveGrid\infra\otel-collector\config.yaml`

- [ ] **Step 1: Create the OTel Collector config**

`infra/otel-collector/config.yaml`:
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

(`debug` is a placeholder exporter for this phase only — it logs received spans to the collector's stdout so Task 9's smoke test can confirm delivery. A real backend, Langfuse, is added in a later phase per the approved architecture plan §9; this is not a code placeholder, it's the documented Phase-1-scoped exporter.)

- [ ] **Step 2: Create the Compose file**

`infra/docker-compose.yml` (includes a top-level `name:` key, added after Task 4 code review found Compose's default project-name-from-directory behavior causes volume-name collisions with any other local project whose compose file also lives in a folder named `infra` — this is exactly what caused the `infra_postgres_data` incident):
```yaml
name: resolvegrid

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: resolvegrid-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-resolvegrid}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-resolvegrid_dev}
      POSTGRES_DB: ${POSTGRES_DB:-resolvegrid}
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-resolvegrid} -d ${POSTGRES_DB:-resolvegrid}"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: resolvegrid-redis
    restart: unless-stopped
    ports:
      - "6380:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  ollama:
    image: ollama/ollama:0.32.15
    container_name: resolvegrid-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "ollama list || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.159.0
    container_name: resolvegrid-otel-collector
    restart: unless-stopped
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector/config.yaml:/etc/otel-collector-config.yaml:ro
    ports:
      - "4317:4317"
      - "4318:4318"

volumes:
  postgres_data:
  ollama_data:
```

- [ ] **Step 3: Bring the stack up and verify health**

Run: `cd "C:\Dev\ResolveGrid" && docker compose -f infra/docker-compose.yml up -d`
Expected: all four containers created and started.

Run: `sleep 15 && docker compose -f infra/docker-compose.yml ps`
Expected: `postgres`, `redis`, `ollama` show `healthy`; `otel-collector` shows `running` (no healthcheck defined for it, that's fine — it has no built-in CLI healthcheck command in the image).

- [ ] **Step 4: Verify GPU is actually reserved for the Ollama container**

Run: `docker exec resolvegrid-ollama nvidia-smi --query-gpu=name,memory.total --format=csv`
Expected: prints the RTX 4080's name and ~12282 MiB total memory — confirms GPU passthrough into the container, not just host-level access.

- [ ] **Step 5: Commit**

```bash
git add infra
git commit -m "feat(infra): add docker-compose stack (postgres+pgvector, redis, gpu ollama, otel-collector)"
```

---

### Task 5: Alembic setup + baseline migration applied against dockerized Postgres

**Files:**
- Create: `C:\Dev\ResolveGrid\apps\api\alembic.ini`
- Create: `C:\Dev\ResolveGrid\apps\api\migrations\env.py`
- Create: `C:\Dev\ResolveGrid\apps\api\migrations\script.py.mako`
- Create: `C:\Dev\ResolveGrid\apps\api\migrations\versions\0001_baseline_health_check.py`
- Modify: `C:\Dev\ResolveGrid\apps\api\pyproject.toml` (add alembic/psycopg deps)

- [ ] **Step 1: Add Alembic and psycopg to the API package**

Edit `apps/api/pyproject.toml`, change `dependencies` to:
```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "resolvegrid-telemetry",
    "alembic>=1.13",
    "psycopg[binary]>=3.2",
    "sqlalchemy>=2.0",
]
```

Run: `cd "C:\Dev\ResolveGrid" && uv sync --all-packages`
Expected: installs alembic/psycopg/sqlalchemy, no errors.

- [ ] **Step 2: Create `alembic.ini`**

`apps/api/alembic.ini`:
```ini
[alembic]
script_location = %(here)s/migrations
prepend_sys_path = .
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 3: Create `migrations/env.py`**

```python
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

database_url = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create `migrations/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 5: Create the baseline migration**

`apps/api/migrations/versions/0001_baseline_health_check.py`:
```python
"""baseline health_check table

Revision ID: 0001
Revises:
Create Date: 2026-08-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "health_check",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("health_check")
```

- [ ] **Step 6: Apply the migration against the dockerized Postgres from Task 4**

Run: `cd "C:\Dev\ResolveGrid" && DATABASE_URL="postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid" uv run --package resolvegrid-api alembic -c apps/api/alembic.ini upgrade head`
Expected: `Running upgrade  -> 0001, baseline health_check table`

- [ ] **Step 7: Verify the table exists**

Run: `docker exec resolvegrid-postgres psql -U resolvegrid -d resolvegrid -c "\dt"`
Expected: lists `health_check` and `alembic_version`.

- [ ] **Step 8: Commit**

```bash
git add apps/api/alembic.ini apps/api/migrations apps/api/pyproject.toml uv.lock
git commit -m "feat(api): add Alembic baseline migration (health_check table)"
```

---

### Task 6: Node workspace root + `apps/web` Next.js placeholder page

**Files:**
- Create: `C:\Dev\ResolveGrid\pnpm-workspace.yaml`
- Create: `C:\Dev\ResolveGrid\package.json`
- Create: `C:\Dev\ResolveGrid\apps\web\package.json`
- Create: `C:\Dev\ResolveGrid\apps\web\tsconfig.json`
- Create: `C:\Dev\ResolveGrid\apps\web\next.config.ts`
- Create: `C:\Dev\ResolveGrid\apps\web\next-env.d.ts`
- Create: `C:\Dev\ResolveGrid\apps\web\.eslintrc.json`
- Create: `C:\Dev\ResolveGrid\apps\web\app\layout.tsx`
- Create: `C:\Dev\ResolveGrid\apps\web\app\page.tsx`

- [ ] **Step 1: Enable pnpm via corepack**

Run: `corepack enable && corepack prepare pnpm@latest --activate`
Expected: pnpm becomes available; `pnpm --version` prints a version.

- [ ] **Step 2: Create `pnpm-workspace.yaml`**

```yaml
packages:
  - "apps/*"
  - "packages/*"
```

- [ ] **Step 3: Create the root `package.json`** (a `packageManager` field pinning the exact pnpm version was added after Task 6 code review found no version alignment between local dev and CI, since `pnpm/action-setup@v4` reads this field automatically when its own `version:` input is omitted)

```json
{
  "name": "resolvegrid",
  "private": true,
  "version": "0.1.0",
  "engines": {
    "node": ">=22"
  },
  "packageManager": "pnpm@11.22.0"
}
```

- [ ] **Step 4: Create `apps/web/package.json`**

```json
{
  "name": "@resolvegrid/web",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint .",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "next": "^16.3.1",
    "react": "^19.2.8",
    "react-dom": "^19.2.8"
  },
  "devDependencies": {
    "@types/node": "^22.10.2",
    "@types/react": "^19.0.2",
    "@types/react-dom": "^19.0.2",
    "eslint": "^9.17.0",
    "eslint-config-next": "^16.3.1",
    "typescript": "^5.7.2"
  }
}
```

- [ ] **Step 5: Create `apps/web/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 6: Create `apps/web/next.config.ts`**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {};

export default nextConfig;
```

- [ ] **Step 7: Create `apps/web/next-env.d.ts`**

```typescript
/// <reference types="next" />
/// <reference types="next/image-types/global" />
```

- [ ] **Step 8: Create `apps/web/eslint.config.mjs`** (revised after Task 6 code review found `.eslintrc.json` non-functional with ESLint 9 + `eslint-config-next@16.3.1` — ESLint 9 defaults to flat config, and this version of `eslint-config-next` already ships a native flat-config array, not a legacy shareable config. An initial attempt to bridge it via `FlatCompat` (`@eslint/eslintrc`) still crashed with the same circular-structure error, because `FlatCompat.extends()` bridges legacy configs *to* flat format — feeding it an already-flat config with real plugin objects made its legacy validator choke trying to serialize them. The fix verified working is importing the flat config array directly, no bridge needed):

`apps/web/eslint.config.mjs`:
```javascript
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = [...nextCoreWebVitals];

export default eslintConfig;
```

Do not create `apps/web/.eslintrc.json` — this flat config file replaces it entirely. No new dependency is needed (`eslint-config-next` is already a devDependency); do not add `@eslint/eslintrc`.

- [ ] **Step 9: Create the placeholder App Router pages**

`apps/web/app/layout.tsx`:
```tsx
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ResolveGrid",
  description: "Kestrel Softworks internal IT service-management + AI-ops platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

`apps/web/app/page.tsx`:
```tsx
export default function HomePage() {
  return (
    <main>
      <h1>ResolveGrid</h1>
      <p>Phase 1 scaffold — role-aware surfaces land in a later phase.</p>
    </main>
  );
}
```

- [ ] **Step 10: Install and build to verify**

Run: `cd "C:\Dev\ResolveGrid" && pnpm install`
Expected: installs all workspace deps, no errors.

Run: `pnpm --filter @resolvegrid/web typecheck`
Expected: exits 0, no type errors.

Run: `pnpm --filter @resolvegrid/web build`
Expected: `✓ Compiled successfully`, static page generated for `/`.

- [ ] **Step 11: Commit**

```bash
git add pnpm-workspace.yaml package.json pnpm-lock.yaml apps/web
git commit -m "feat(web): add Next.js placeholder app with pnpm workspace"
```

---

### Task 7: CI skeleton (GitHub Actions — lint + typecheck + test for API and web)

**Files:**
- Create: `C:\Dev\ResolveGrid\.github\workflows\ci.yml`

- [ ] **Step 1: Create the workflow**

`.github/workflows/ci.yml` (revised after Task 7 code review found `branches: [main]` would never fire — this repo's only branch is `master`, created by `git init`'s default in Task 1, and no remote/rename to `main` has happened; also added `timeout-minutes` and an explicit `permissions` block as cheap, standard hardening for a from-scratch workflow):
```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

permissions:
  contents: read

jobs:
  api:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          version: "0.8.13"
      - run: uv sync --all-packages
      - run: uv run --package resolvegrid-api pytest apps/api/tests -v
      - run: uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v

  web:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
      - run: pnpm install --frozen-lockfile
      - run: pnpm --filter @resolvegrid/web lint
      - run: pnpm --filter @resolvegrid/web typecheck
      - run: pnpm --filter @resolvegrid/web build
```

- [ ] **Step 2: Verify locally that every command the workflow runs actually succeeds**

Run: `cd "C:\Dev\ResolveGrid" && uv run --package resolvegrid-api pytest apps/api/tests -v && uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v`
Expected: all pass (already verified in Tasks 2–3, re-confirming after Task 5's dependency additions).

Run: `pnpm --filter @resolvegrid/web lint && pnpm --filter @resolvegrid/web typecheck && pnpm --filter @resolvegrid/web build`
Expected: lint passes with no errors, typecheck exits 0, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for api and web lint/typecheck/test"
```

---

### Task 8: `docs/` skeleton + `PROGRESS.md` ledger

**Files:**
- Create: `C:\Dev\ResolveGrid\docs\ARCHITECTURE.md`
- Create: `C:\Dev\ResolveGrid\docs\DATA_MODEL.md`
- Create: `C:\Dev\ResolveGrid\docs\WORKFLOWS_TOOLS.md`
- Create: `C:\Dev\ResolveGrid\docs\RAG_INGESTION.md`
- Create: `C:\Dev\ResolveGrid\docs\EVALUATIONS.md`
- Create: `C:\Dev\ResolveGrid\docs\TELEMETRY_COST.md`
- Create: `C:\Dev\ResolveGrid\docs\SECURITY.md`
- Create: `C:\Dev\ResolveGrid\docs\API_CONTRACTS.md`
- Create: `C:\Dev\ResolveGrid\docs\DESIGN_SYSTEM.md`
- Create: `C:\Dev\ResolveGrid\docs\EXPERIMENT_REGISTRY.md`
- Create: `C:\Dev\ResolveGrid\docs\RUNBOOKS.md`
- Create: `C:\Dev\ResolveGrid\docs\DECISION_LOG.md`
- Create: `C:\Dev\ResolveGrid\docs\PROGRESS.md`
- Create: `C:\Dev\ResolveGrid\docs\adr\0001-modular-monolith-topology.md`

- [ ] **Step 1: Create each canonical doc stub**

Each file below is a genuine (short) starting document, not a "TBD" placeholder — each states its real purpose so a future session knows what belongs there.

`docs/ARCHITECTURE.md`:
```markdown
# Architecture

System topology: modular monolith (`apps/api` FastAPI + `apps/web` Next.js) with LangGraph as the sole top-level agent orchestrator, one bounded in-process Google ADK specialist, no A2A in v1. Full rationale: `docs/adr/0001-modular-monolith-topology.md` and the approved architecture plan referenced in `CLAUDE.md`.

## Current topology (Phase 1)
- `apps/api` — FastAPI, `/health` only so far.
- `apps/web` — Next.js placeholder.
- `packages/telemetry` — OTel tracing helper, wired into API startup.
- `infra/` — Postgres+pgvector, Redis, GPU Ollama, OTel Collector (debug exporter only, no backend yet).

See `docs/PROGRESS.md` for what's actually built vs. planned.
```

`docs/DATA_MODEL.md`:
```markdown
# Data Model

Schema shapes are defined in the approved architecture plan (§2). As of Phase 1, only `health_check` exists (Alembic baseline migration, `apps/api/migrations/versions/0001_baseline_health_check.py`) — a placeholder-free smoke-test table, not a domain entity. Org/ticketing/knowledge/approval/telemetry tables land in Phases 2, 3, 7, 9.
```

`docs/WORKFLOWS_TOOLS.md`:
```markdown
# Agent Workflows & Tools

The 13-stage LangGraph workflow mapping is defined in the approved architecture plan (§6). No agent graph exists yet as of Phase 1 — this document gets its first real content in Phase 6.
```

`docs/RAG_INGESTION.md`:
```markdown
# RAG & Ingestion

Baseline RAG design (structure-aware chunking, `nomic-embed-text` embeddings, pgvector HNSW + Postgres FTS hybrid retrieval, bge-reranker-v2-m3) is defined in the approved architecture plan (§7). No ingestion pipeline exists yet as of Phase 1 — first real content lands in Phase 7.
```

`docs/EVALUATIONS.md`:
```markdown
# Evaluations

Golden-dataset and grading methodology (deterministic graders first, calibrated model judges second) is defined in the approved architecture plan (§8). No golden dataset exists yet as of Phase 1 — first real content lands in Phase 6 (tiny scaffold set) and Phase 10 (full harness).
```

`docs/TELEMETRY_COST.md`:
```markdown
# Telemetry & Cost

ResolveGrid-owned telemetry/cost schema (`AgentRun`/`Span`/`ModelCall`/`PricingVersion`) is defined in the approved architecture plan (§9). As of Phase 1, only a bare OTel Collector with a `debug` exporter exists (`infra/otel-collector/config.yaml`) — it logs received spans to stdout, nothing is persisted yet. `ModelCall`/`PricingVersion` land in Phase 4; Langfuse/Prometheus/Grafana land alongside later phases per §9.
```

`docs/SECURITY.md`:
```markdown
# Security & Threat Model

Full threat model (cache-as-security-boundary, prompt-injection boundaries, tool allowlists, approval binding) is defined in the approved architecture plan (§10). As of Phase 1 there is no authz module, no tools, and no user-facing surface with real data — this document gets its first real content in Phase 2 (`packages/authz` skeleton).
```

`docs/API_CONTRACTS.md`:
```markdown
# API Contracts

`packages/contracts` (shared typed tool/event/eval-case schemas) does not exist yet as of Phase 1. The only live endpoint is `GET /health` on `apps/api`, returning `{"status": "ok"}`. First real contract work lands in Phase 3 (ticket schemas).
```

`docs/DESIGN_SYSTEM.md`:
```markdown
# Design System

No design system exists yet as of Phase 1 — `apps/web` is an unstyled placeholder. Visual direction (three-pane workspace, waterfall trace view, purpose-built dashboards, anti-patterns to avoid) is defined in the approved architecture plan (§11) and gets executed in Phase 14 using the `frontend-design`, `impeccable`, `design-taste-frontend`, and `emil-design-eng` skills.
```

`docs/EXPERIMENT_REGISTRY.md`:
```markdown
# Experiment Registry

No experiments have been run yet as of Phase 1. Before/after experiment protocol (fixed dataset + seed, paired comparison, raw results under `eval/results/<experiment-id>/`) is defined in the approved architecture plan (§12). First entries land in Phase 7 (retrieval baseline) onward.
```

`docs/RUNBOOKS.md`:
```markdown
# Runbooks

No operational runbooks exist yet as of Phase 1 (no mutating tools, no incidents possible). First runbook (tool compensation/rollback guidance) lands in Phase 9 alongside the first mutating tool.
```

`docs/DECISION_LOG.md`:
```markdown
# Decision Log

| Date | Decision | Rationale | Ref |
|---|---|---|---|
| 2026-08-20 | Modular monolith over microservices; pnpm+uv workspaces over Nx/Turborepo; centralized `packages/authz` over per-service RLS or an external policy engine | See ADR | `docs/adr/0001-modular-monolith-topology.md` |
| 2026-08-20 | ADK scoped to exactly one bounded in-process specialist agent, no A2A in v1 | ADK's own docs show it wants to own orchestration itself (conflicts with LangGraph); A2A is not a sub-agent/tool-call protocol and this platform has no cross-trust-boundary service yet | Approved architecture plan §4 |
| 2026-08-20 | Anthropic primary / OpenAI fallback behind LiteLLM | User has API access/budget for both | Approved architecture plan §4 |
```

`docs/PROGRESS.md`:
```markdown
# Progress Ledger

The single authoritative status ledger. States: Proposed / Approved / In Progress / Blocked / Implemented / Verified / Superseded. No status is duplicated anywhere else — other docs link here.

| Unit ID | Description | State | Entry Dependencies | Exit-Criteria Evidence | Last Updated |
|---|---|---|---|---|---|
| Phase 1 | Repo scaffold & local infra skeleton | In Progress | none | see `docs/superpowers/plans/2026-08-20-phase1-repo-scaffold.md` Task 9 smoke test output | 2026-08-20 |
```

`docs/adr/0001-modular-monolith-topology.md`:
```markdown
# ADR 0001: Modular Monolith Topology

## Status
Accepted

## Context
Plan.md requires "clear bounded modules over premature microservices" and to "add queues/events only where async execution, retries, or isolation require them." Three topology options were weighed: full microservices per bounded module, a single synchronous process, and a modular monolith with targeted async workers.

## Decision
One FastAPI deployable importing `agent-orchestration`, `retrieval`, `authz`, `telemetry`, and `operational-adapters` as Python libraries, plus Redis/Arq workers for the four concrete async needs identified: ingestion, batch evaluation, load-replay, notification fan-out. Durability for HITL approvals comes from LangGraph's `AsyncPostgresSaver` checkpointing, not a queue.

## Consequences
Simpler deployment and debugging at current scale. Revisit if sustained CPU/memory contention appears between API request-handling and agent-graph execution, or agent workers need to scale independently of the HTTP tier.
```

- [ ] **Step 2: Commit**

```bash
git add docs
git commit -m "docs: add canonical documentation skeleton and progress ledger"
```

---

### Task 9: Infra smoke test script + full Phase 1 exit-criteria verification

**Files:**
- Create: `C:\Dev\ResolveGrid\scripts\smoke_test.sh`
- Create: `C:\Dev\ResolveGrid\Makefile`
- Modify: `C:\Dev\ResolveGrid\docs\PROGRESS.md` (mark Phase 1 Verified)

- [ ] **Step 1: Create the smoke test script**

`scripts/smoke_test.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0/6: dependencies synced =="
uv sync --all-packages
pnpm install

echo "== 1/6: infra containers healthy =="
docker compose -f infra/docker-compose.yml up -d
# otel-collector's image is fully distroless (no shell, no wget/curl/nc) so it has no
# Docker healthcheck by design (confirmed during Task 4) — it always reports blank health,
# which this grep already tolerates since blank never matches "unhealthy|starting". Its
# actual readiness is confirmed indirectly by step 4/6 below (the span-delivery log check).
for i in $(seq 1 20); do
  statuses=$(docker compose -f infra/docker-compose.yml ps --format '{{.Name}}: {{.Health}}')
  echo "$statuses"
  if ! echo "$statuses" | grep -qi "unhealthy\|starting"; then
    break
  fi
  sleep 3
done

echo "== 2/6: alembic migration applied =="
export DATABASE_URL="postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid"
uv run --package resolvegrid-api alembic -c apps/api/alembic.ini upgrade head
docker exec resolvegrid-postgres psql -U resolvegrid -d resolvegrid -c "\dt" | grep -q health_check

echo "== 3/6: FastAPI /health =="
uv run --package resolvegrid-api uvicorn resolvegrid_api.main:app --app-dir apps/api/src --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
sleep 3
curl -sf http://localhost:8000/health | grep -q '"status":"ok"'

echo "== 4/6: OTel collector received the startup span =="
# The "api.startup" span is emitted once during FastAPI's lifespan startup and
# queued in a BatchSpanProcessor, which flushes it on its own periodic timer
# (default ~5s) -- the server does NOT need to be killed/shut down first for
# the span to be exported. Poll for it here, with the server still running,
# rather than killing the process up front:
#
# Direct repro showed that killing the process immediately after the health
# check (before that periodic flush timer fires) permanently loses the span.
# The intuitive fix -- "kill it so the lifespan shutdown handler force-flushes
# the span via trace.get_tracer_provider().shutdown()" -- does not work on
# this platform: on Windows/Git-Bash, `uv run` spawns uvicorn as a native
# Windows child process outside bash's own job-control tree, so a plain
# `kill "$API_PID"` (an MSYS-internal pid) silently no-ops on the real
# process, and the only thing that actually reaches it -- `taskkill //F` --
# is a forceful TerminateProcess call (confirmed by direct repro: a
# non-forceful `taskkill` on this process errors out with "This process can
# only be terminated forcefully"). A forceful kill bypasses the ASGI lifespan
# shutdown hook entirely, so it destroys the queued span instead of flushing
# it if it fires before the periodic timer does.
#
# Separately, the OTLP gRPC exporter's *first* export against a freshly
# (re)started collector container can take much longer than a couple seconds
# to actually land -- confirmed by direct repro on a genuine cold start (fresh
# containers, no prior connection): the span took over a minute to appear.
# So poll (not a fixed sleep-then-grep) for up to ~60s with the server left
# running throughout, and only tear the server down afterward, once its span
# has had a real chance to be exported either way.
span_found=0
for i in $(seq 1 30); do
  if docker compose -f infra/docker-compose.yml logs otel-collector 2>&1 | grep -q "api.startup"; then
    span_found=1
    break
  fi
  sleep 2
done

# Tear down the server now that the span check is done (or has timed out).
# See the note above: on Windows/Git-Bash $API_PID doesn't reliably identify
# the real native process, so resolve the actual PID bound to port 8000 via
# `netstat` and force-terminate that one with `taskkill` when available. On
# real POSIX systems `taskkill` doesn't exist, `command -v` skips that
# branch, and the plain `kill "$API_PID"` below is sufficient since $! there
# IS the real pid.
REAL_PID=$(netstat -ano 2>/dev/null | grep ':8000 ' | grep -i LISTENING | awk '{print $NF}' | head -n1)
if [ -n "${REAL_PID:-}" ] && command -v taskkill >/dev/null 2>&1; then
  taskkill //F //PID "$REAL_PID" >/dev/null 2>&1 || true
fi
kill "$API_PID" 2>/dev/null || true

if [ "$span_found" -ne 1 ]; then
  echo "ERROR: api.startup span never appeared in otel-collector logs after ~60s" >&2
  exit 1
fi

echo "== 5/6: Next.js build =="
pnpm --filter @resolvegrid/web build

echo "ALL SMOKE CHECKS PASSED"
```

- [ ] **Step 2: Create the `Makefile`**

```makefile
.PHONY: help infra-up infra-down api-dev web-dev sync install smoke test

help:
	@echo "make infra-up     - start Docker Compose infra"
	@echo "make infra-down   - stop Docker Compose infra"
	@echo "make sync         - uv sync (Python workspace)"
	@echo "make install      - pnpm install (Node workspace)"
	@echo "make api-dev      - run FastAPI dev server"
	@echo "make web-dev      - run Next.js dev server"
	@echo "make test         - run all Python tests"
	@echo "make smoke        - run scripts/smoke_test.sh"

infra-up:
	docker compose -f infra/docker-compose.yml up -d

infra-down:
	docker compose -f infra/docker-compose.yml down

sync:
	uv sync --all-packages

install:
	pnpm install

api-dev:
	uv run --package resolvegrid-api uvicorn resolvegrid_api.main:app --reload --app-dir apps/api/src

web-dev:
	pnpm --filter @resolvegrid/web dev

test:
	uv run --package resolvegrid-api pytest apps/api/tests -v
	uv run --package resolvegrid-telemetry pytest packages/telemetry/tests -v

smoke:
	bash scripts/smoke_test.sh
```

- [ ] **Step 3: Run the smoke test**

Run: `cd "C:\Dev\ResolveGrid" && chmod +x scripts/smoke_test.sh && bash scripts/smoke_test.sh`
Expected: `ALL SMOKE CHECKS PASSED` printed at the end, no step exits non-zero.

If step 4 (OTel span check) fails because `debug` exporter logs weren't flushed by the time of the grep, add a `sleep 5` before the log check and re-run — `BatchSpanProcessor` exports on an interval, not instantly.

- [ ] **Step 4: Update `docs/PROGRESS.md` to mark Phase 1 Verified**

Edit `docs/PROGRESS.md`, change the Phase 1 row's State from `In Progress` to `Verified` and update Exit-Criteria Evidence to: `scripts/smoke_test.sh — ALL SMOKE CHECKS PASSED, run 2026-08-20`.

- [ ] **Step 5: Final commit**

```bash
git add scripts/smoke_test.sh Makefile docs/PROGRESS.md
git commit -m "feat: add infra smoke test and Makefile; mark Phase 1 verified"
```

---

## Self-Review

**Spec coverage:** Every Phase 1 deliverable from the approved architecture plan (§14 Phase 1) is covered — monorepo skeleton (Task 1, 2, 6), Docker Compose infra (Task 4), Alembic baseline migration (Task 5), CI skeleton (Task 7), docs skeleton + PROGRESS.md (Task 8), health endpoints + OTel startup span (Task 2, 3), smoke test proving all exit criteria (Task 9). No plan.md requirement for Phase 1 specifically is left uncovered.

**Placeholder scan:** No "TBD"/"implement later"/"add appropriate error handling" strings appear in any step. The one exporter marked "placeholder" (OTel `debug` exporter) is explicitly scoped as the real Phase-1 choice with a stated reason and a named future phase that replaces it — not an unfinished step.

**Type consistency:** `init_tracing(service_name: str) -> trace.Tracer` (Task 3) is called identically in `apps/api/main.py` (Task 3 Step 8) and both test files. `resolvegrid-telemetry` package name matches across `pyproject.toml` workspace members, `tool.uv.sources`, and `uv run --package` invocations throughout. `resolvegrid-api` package name matches across all `uv run --package` calls in Tasks 2, 3, 5, 7, 9 and the Makefile.
