#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0/6: dependencies synced =="
uv sync --all-packages
pnpm install

echo "== 1/6: infra containers healthy =="
# --env-file is required, not optional, on this Git-Bash/Windows setup: Compose
# does not reliably auto-discover .env from this shell's cwd (confirmed during
# Phase 5) -- without it, unset-with-no-inline-default vars like
# ANTHROPIC_API_KEY/OPENAI_API_KEY silently resolve to "" instead of erroring,
# breaking cloud-provider routing with no visible failure anywhere in this
# script. Also strip any pre-existing shell-level ANTHROPIC_API_KEY/
# OPENAI_API_KEY (Compose prioritizes real shell env over --env-file, so a
# stale exported value would silently shadow the real one in .env).
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY docker compose -f infra/docker-compose.yml --env-file .env up -d
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
docker exec resolvegrid-postgres psql -U resolvegrid -d resolvegrid -c "\dt" | grep -q employee

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
