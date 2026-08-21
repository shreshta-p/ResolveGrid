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
- `uv sync --all-packages` — install/sync Python workspace
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
- `docs/API_CONTRACTS.md` — shared typed tool/event/eval-case schemas
- `docs/DESIGN_SYSTEM.md` — visual direction, design tokens
- `docs/EXPERIMENT_REGISTRY.md` — before/after experiment protocol and results
- `docs/RUNBOOKS.md` — operational runbooks (tool compensation/rollback, etc.)
- `docs/DECISION_LOG.md` — dated architecture/scope decisions with rationale
- `docs/PROGRESS.md` — the single status ledger

## Full specification
`plan.md` — original master spec. `docs/PLAN_APPROVED.md` — approved architecture/phase plan.
