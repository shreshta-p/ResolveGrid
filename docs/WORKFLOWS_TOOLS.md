# Agent Workflows & Tools

The 13-stage LangGraph workflow mapping is defined in the approved architecture plan (§6).

## Agent graph (Phase 6)

Stages **1, 2, 3, 11, 12, 13** are implemented; stages 4-10 (retrieval, tool selection/execution, approvals) are not — there is no knowledge base or tool-calling yet, per the approved plan's own explicit Phase 6 scope. The graph itself — node logic, prompts, the `AgentState` shape — is defined in [`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`](../services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py), the single source of truth; not duplicated by hand here (same anti-drift reasoning as the ticket state machine above).

| Stage | Node | Notes |
|---|---|---|
| 1. Authentication and requesting principal | `apps/api/src/resolvegrid_api/deps.py`'s `get_principal` | Resolved once, before the graph runs — same temporary debug-header shim used everywhere else in this codebase. |
| 2. Intent, risk, and data-scope classification | `classify_intent` | Structured JSON classification into a fixed `intent`/`risk_level` set. No `data_scope_tags` yet (that's a retrieval-era field — Phase 7+). No real enforcement is tied to `risk_level` yet; it exists to establish the field/pattern early. |
| 3. Retrieval/tool decision | *(trivial — no branching)* | Every request goes straight to `compose_response`; there is nothing to route to yet (no retrieval, no tools). |
| 11. Structured response and citation verification | *(no-op)* | Folded into `finalize` — there are no citations to verify without a knowledge base. |
| 12. Safe answer, abstention, or escalation | `finalize` | Passes through `compose_response`'s answer, or substitutes a fixed fallback message if an earlier node recorded an error. |
| 13. Telemetry, evaluation hooks, feedback | `AgentRun`/`Span` (`apps/api/src/resolvegrid_api/models/agent.py`) + OTel | One `AgentRun` row per `/chat` call, one `Span` row per conceptual stage. Latency is only measured for the whole graph invocation today, split evenly across stages — not real per-node timing (documented in `chat.py`'s docstring). |

**Durability**: the graph is checkpointed to real Postgres via `AsyncPostgresSaver` (not the in-memory `MemorySaver`) — see `docs/adr/0002-postgres-checkpointer.md`. Proven via `apps/api/tests/test_checkpoint_restore.py`: state written by one graph/checkpointer instance is correctly read back by a completely separate, independently-constructed instance, the same guarantee a real process restart depends on.

**LLM calls**: graph nodes never import `resolvegrid_api.llm_gateway` directly — `services/agent-orchestration` is a workspace library `apps/api` depends on, so importing the other way would be backwards. Instead, every node is built by a factory function taking an injected `complete_fn: Callable[[str], str]`; `apps/api`'s lifespan wires in `llm_gateway.complete(prompt).text`. See `graph.py`'s module docstring for the full reasoning.

**User-facing surface**: `POST /chat` (`apps/api/src/resolvegrid_api/routers/chat.py`) and `apps/web/app/chat/page.tsx` — a single-turn "ask a question, get a general-knowledge answer" feature, explicitly labeled in the UI as having no ticket/company-specific knowledge base yet.

## Ticket lifecycle (Phase 3)

A ticket has one of 5 statuses: `open`, `in_progress`, `resolved`, `closed`, `reopened`. The legal transitions between them are defined in exactly one place — `ALLOWED_TRANSITIONS` in [`packages/contracts/src/resolvegrid_contracts/tickets.py`](../packages/contracts/src/resolvegrid_contracts/tickets.py) — and are not duplicated by hand here, since a hand-copied table drifts (this already happened once: the `apps/web` transition-button UI briefly diverged from this table before being caught in review and fixed to read from it as the source of truth). Both `apps/api`'s enforcement (`apps/api/src/resolvegrid_api/routers/tickets.py`) and the frontend's transition-button affordance (`apps/web/app/tickets/[id]/page.tsx`) derive from this same table.

### Who can do what

- **Employee**: create tickets; view only their own tickets (`ticket.list`/`ticket.view` fall back to self-scope when no department grant exists).
- **Analyst / Approver**: view and transition tickets within their department scope (via a `RoleAssignment` row with `scope="department"`). `ticket.transition` is a staff-only action — an employee with no department-scoped grant is denied outright, never falls back to self-scope.
- **Admin**: unrestricted view/transition across all departments.

All of the above is enforced by `packages/authz`'s single `authorize(principal, action)` entry point — never re-implemented per route, never enforced only in the UI (the frontend renders transition buttons regardless of role; the backend's 403 is the actual boundary, verified in Task 6's manual walkthrough).

### Audit trail

Every mutation is audited via `apps/api/src/resolvegrid_api/audit.py`'s hash-chained `AuditLog`:
- **Create**: one `AuditLog` row (`action="ticket.create"`). No `TicketStateTransition` row is written for creation — that table records only status *changes*, and a newly created ticket has no prior status to transition from.
- **Transition**: one `TicketStateTransition` row (`from_status`/`to_status`/actor) *and* one `AuditLog` row (`action="ticket.transition"`, `before`/`after` capturing the status change) are written together, atomically, in the same request.

The audit log is append-only and tamper-evident (`record_hash`/`previous_record_hash` chain covering the full payload including `metadata_json`); `verify_chain_integrity()` detects any row tampered with after the fact.
