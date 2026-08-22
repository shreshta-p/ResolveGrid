# Decision Log

| Date | Decision | Rationale | Ref |
|---|---|---|---|
| 2026-08-20 | Modular monolith over microservices; pnpm+uv workspaces over Nx/Turborepo; centralized `packages/authz` over per-service RLS or an external policy engine | See ADR | `docs/adr/0001-modular-monolith-topology.md` |
| 2026-08-20 | ADK scoped to exactly one bounded in-process specialist agent, no A2A in v1 | ADK's own docs show it wants to own orchestration itself (conflicts with LangGraph); A2A is not a sub-agent/tool-call protocol and this platform has no cross-trust-boundary service yet | Approved architecture plan §4 |
| 2026-08-20 | Anthropic primary / OpenAI fallback behind LiteLLM | User has API access/budget for both | Approved architecture plan §4 |
| 2026-08-21 | Deferred adding a naming_convention to Base.metadata | Retrofitting one after migrations already exist risks autogenerate proposing to rename already-applied constraints; needs its own dedicated task, not a quick fix | apps/api/src/resolvegrid_api/models/base.py |
| 2026-08-21 | Directory API resolves the requesting principal via a temporary `X-Debug-Employee-Id` header, not real authentication | No auth system exists yet (JWT/session auth is a later phase per the agent-workflow's stage 1); Phase 2's exit criteria require role-restricted endpoints to exist and be tested now. The header is explicitly documented as insecure scaffolding in `apps/api/src/resolvegrid_api/deps.py`'s docstring and must be replaced before any non-local deployment. | `apps/api/src/resolvegrid_api/deps.py` |
