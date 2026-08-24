# Agent Workflows & Tools

The 13-stage LangGraph workflow mapping is defined in the approved architecture plan (§6). No agent graph exists yet — this document gets its full content in Phase 6.

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
