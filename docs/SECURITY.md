# Security & Threat Model

Full threat model (cache-as-security-boundary, prompt-injection boundaries, tool allowlists, approval binding) is defined in the approved architecture plan (`docs/PLAN_APPROVED.md` §10) and gets built out progressively as each concern becomes real.

## Phase 2: authorization is real, authentication is not (yet)

- **Authorization is centralized and enforced**: `packages/authz`'s `authorize()` is the single policy entry point; the directory API (`apps/api/src/resolvegrid_api/routers/directory.py`) delegates every access decision to it and independently verifies the specific resource returned against that decision (`_in_scope`) rather than trusting a query filter alone.
- **Authentication does NOT exist yet.** The directory API resolves "who is asking" via a temporary `X-Debug-Employee-Id` HTTP header (`apps/api/src/resolvegrid_api/deps.py`) — anyone can claim to be any employee id. This is explicitly documented scaffolding, not a security boundary, and MUST be replaced with real authentication (JWT/session, per the agent-workflow's stage 1) before any non-local deployment. See `docs/DECISION_LOG.md` for the dated decision record.
- **Manager-hierarchy integrity**: `resolvegrid_api.org.cycle_guard.would_create_cycle()` exists to prevent reporting-line cycles, though no mutation endpoint calls it yet in Phase 2 (only the seed generator writes `manager_id`, via direct acyclic-by-construction tree building).

Cache-authorization-as-security-boundary, prompt-injection boundaries, and tool allowlists remain not-yet-applicable (no cache, no LLM tools exist yet) — tracked for the phases that introduce them.
