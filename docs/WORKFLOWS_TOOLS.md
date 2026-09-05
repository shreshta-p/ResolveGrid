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

## Tools and approvals (Phase 9)

Stages 8-9-10 of the 13-stage workflow (tool selection/execution, approval
staging, mutation execution) are now real: typed tool contracts, allowlist
filtering applied before the model ever sees a tool definition, a durable
`interrupt()`/`Command(resume=...)` approval boundary, and one real mutating
tool (`grant_vpn_access`) wired end-to-end through the actual running app.

### Tool catalog

`packages/contracts/src/resolvegrid_contracts/tools.py`'s `TOOL_REGISTRY`
(a frozen `MappingProxyType`, entries themselves frozen pydantic models) is
the single source of truth for every tool this app knows about — not
duplicated by hand here, same anti-drift reasoning as the ticket state
machine and the agent graph above:

| Tool | `required_role` | `mutating` | `requires_approval` | Params |
|---|---|---|---|---|
| `lookup_employee_entitlements` | `analyst` | No | No | `{employee_id: int}` |
| `grant_vpn_access` | `analyst` | Yes | Yes | `{employee_id: int, justification: str}` |

`grant_vpn_access` was chosen (Task 2) because it maps directly onto the
`Entitlement`/`AccessGroup`/`EmployeeEntitlement` schema Task 1 added and is
a plausible, boring, realistic IT action — not a theatrical demo. Neither
tool sets `required_entitlement` yet; that field exists on `ToolContract`
for forward compatibility with a future tool that needs it, and
`available_tools_for_principal` already supports filtering on it.

### Allowlist-before-the-model design

`apps/api/src/resolvegrid_api/tool_execution.py` enforces a strict order,
and this order is the actual security property, not an implementation
detail: `available_tools_for_principal(principal)` filters `TOOL_REGISTRY`
down to what this specific principal may even be offered — via
`packages/authz`'s new `principal_has_role()` helper (shared, via
`_is_global_admin`, with `authorize()`'s admin-dominates precedent) — and
**only that filtered result may ever be formatted into a prompt or shown in
a UI**. `select_tool(tool_name, available)` then resolves a requested tool
name against that already-filtered list, raising `ToolNotAllowedError` for
BOTH "no such tool" and "exists but not permitted" identically (a principal
must never be able to distinguish the two by probing). `validate_tool_schema`
validates params against `ToolContract.params_schema` (real JSON Schema, via
the `jsonschema` library — a new `apps/api` dependency this task added)
before anything executes. A principal missing the required role never
receives `grant_vpn_access`'s definition at all — proven by
`apps/api/tests/test_tool_execution.py`'s
`test_principal_with_no_role_grants_sees_no_tools`/
`test_select_tool_raises_when_filtered_out_of_available`.

### Two-graph architecture

`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`
now compiles **two separate graphs**, sharing one real `AsyncPostgresSaver`
checkpointer instance (wired in `apps/api/main.py`'s lifespan):

- **`build_graph`** — the existing chat pipeline (`classify_intent -> retrieve
  -> compose_response -> verify_citations -> finalize`), untouched by this
  phase.
- **`build_tool_invocation_graph`** (Task 7a) — a new, separate two-node
  graph, `START -> request_approval -> execute_mutation -> END`, for explicit
  tool invocation.

Why a second graph rather than teaching the chat graph to infer tool intent
from free-form text: that would be real LLM-driven tool selection (a much
larger, genuinely different undertaking no task in this phase specifies) and
would risk regressing the already-well-tested chat flow. A real IT-analyst
UI doesn't work by typing free text and hoping an LLM infers "grant this
person VPN access" either — it works by the analyst deliberately choosing
that action from a form (`apps/web/app/tools/page.tsx`), which is exactly
the explicit `proposed_tool_name`/`proposed_tool_params` this graph expects
already sitting in initial state. Sharing one checkpointer across both
compiled graphs is safe because `AsyncPostgresSaver` checkpoint rows are
keyed only by `(thread_id, checkpoint_ns)` from the caller's own config,
never by which compiled graph wrote them (confirmed against the installed
`langgraph==1.2.11` source) — the only real requirement is that a given
`thread_id` is never invoked against both graphs, which `routers/tools.py`'s
fresh-`uuid4()`-per-invocation convention already guarantees.

`request_approval` (Task 5, reused verbatim, not reimplemented) is this
codebase's first real use of LangGraph's `interrupt()`/`Command(resume=...)`
human-in-the-loop primitives — Phase 6 only ever wired the checkpointer, it
never actually paused a run. LangGraph re-runs an interrupted node's ENTIRE
body from the top on every resume (verified against the installed
`langgraph==1.2.11` source, since the docs site's HIL pages 404/redirect as
of this writing) — the snapshot-hash/idempotent-upsert design below exists
specifically because of this.

### Full request lifecycle

1. **`POST /tools/{tool_name}/invoke`** (`apps/api/src/resolvegrid_api/routers/tools.py`,
   Task 7a) — allowlist/validate (above), then:
   - Non-mutating tool: calls `execute_readonly_tool` (Task 6) directly
     against the request's own session — no graph, no interrupt.
   - Mutating, approval-gated tool: mints a fresh `thread_id`, `ainvoke()`s
     `build_tool_invocation_graph`'s compiled graph. This call is *expected*
     to hit `request_approval`'s `interrupt()` and return with an
     `"__interrupt__"` key rather than a completed state (the interrupted
     node's own `return` never executes) — the real `approval_request_id`
     is read off `result["__interrupt__"][0].value["approval_request_id"]`,
     which `request_approval` populates from `request_approval_fn`'s return
     value *before* calling `interrupt()`. Response:
     `{"status": "pending_approval", "approval_request_id": <int>,
     "thread_id": <str>}`.
2. **`request_approval_for_agent`** (`apps/api/src/resolvegrid_api/approval_service.py`,
   Task 5) computes the snapshot hash and does an identity-keyed idempotent
   upsert against `ApprovalRequest` (`status="pending"`, `expires_at` = now
   + 24h). Concurrent calls with an identical identity are serialized via a
   Postgres `pg_advisory_xact_lock`, not a DB uniqueness constraint (several
   identity columns are nullable, and Postgres treats `NULL <> NULL` in a
   composite unique index — see `docs/DECISION_LOG.md`).
3. **`GET /approvals`** (`apps/api/src/resolvegrid_api/routers/approvals.py`,
   Task 7b) — an approver lists pending requests (action type, params, risk
   context, evidence refs, requester, expiry) via `apps/web/app/approvals/page.tsx`.
4. **`POST /approvals/{id}/decide`** (Task 7b) — writes `ApprovalDecision` +
   flips `ApprovalRequest.status`, **commits that decision first**, then only
   afterward resumes the paused graph run via
   `tool_invocation_graph.ainvoke(Command(resume=payload.decision),
   config={"configurable": {"thread_id": approval_request.agent_run_id}})`.
   This ordering means a human decision is durably recorded even if the
   resume call itself fails.
5. **Resume drives `execute_mutation`** (Task 6/7a) to completion: re-fetches
   the `ApprovalRequest`, re-verifies the snapshot hash, checks
   `status=="approved"` and not expired, checks the duplicate-replay guard,
   then dispatches to the real adapter (`grant_vpn_access`, Task 3) and
   writes both a `ToolCall` row and a hash-chained `AuditLog` row
   (`action="tool.grant_vpn_access"`, `actor_type="agent"`).

See `docs/SECURITY.md`'s "Approval binding (Phase 9)" section for the full
defense-in-depth detail on steps 2 and 5.

### Who can do what

- **`analyst`** (or admin) — required to even see `grant_vpn_access`/
  `lookup_employee_entitlements` in `available_tools_for_principal`'s output,
  via `principal_has_role`.
- **`approval.list`/`approval.decide`** (`packages/authz/policy.py`) — new
  staff-only actions (joining `ticket.transition`'s precedent): a principal
  with no admin/department-scoped analyst-or-approver grant is denied
  outright, never downgraded to a self-scoped view. **Deliberate scope
  limit**: `GET /approvals` applies no department filtering — any principal
  who passes the `approval.list` check sees every pending request
  company-wide, since `ApprovalRequest` has no department column and this
  phase's only `action_type` has no department-routing semantics anywhere
  yet (`ApprovalPolicy.stages_json` exists but nothing reads it). See
  `docs/DECISION_LOG.md` for the full reasoning.

### Durability guarantee

`apps/api/tests/test_restart_mid_approval.py`'s
`test_restart_mid_approval_cross_instance_resume_executes_mutation_exactly_once`
is this phase's exit-criteria proof, mirroring
`test_checkpoint_restore.py`'s established cross-instance methodology (see
`docs/adr/0002-postgres-checkpointer.md`): a `grant_vpn_access` run is paused
at a real `interrupt()` by one fully-independent `AsyncPostgresSaver` +
compiled-graph instance, that instance's connection is then completely
closed, and a second, separately-constructed instance resumes it to
completion — proven, via direct DB row-count assertions (never re-checking
through either graph instance), to leave exactly one `ApprovalRequest` row,
exactly one `EmployeeEntitlement` grant, and exactly one `ToolCall`
`status="success"` row behind, despite `request_approval` genuinely
re-executing its entire body on the resume.
