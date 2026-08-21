.PHONY: help infra-up infra-down api-dev web-dev sync install smoke test

help:
	@echo "make infra-up     - start Docker Compose infra"
	@echo "make infra-down   - stop Docker Compose infra"
	@echo "make sync         - uv sync --all-packages (Python workspace)"
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
