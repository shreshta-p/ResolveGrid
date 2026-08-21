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
sleep 3
curl -sf http://localhost:8000/health | grep -q '"status":"ok"'
kill $API_PID

echo "== 4/6: OTel collector received the startup span =="
docker compose -f infra/docker-compose.yml logs otel-collector | grep -q "api.startup"

echo "== 5/6: Next.js build =="
pnpm --filter @resolvegrid/web build

echo "ALL SMOKE CHECKS PASSED"
