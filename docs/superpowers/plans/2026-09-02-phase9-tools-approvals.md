# Phase 9 — Tools, Approvals, and Mutation Execution (stages 8-9-10)

Approved master plan reference: `docs/PROGRESS.md`, the plan.md master spec, and the plan-mode
architecture document (§6 stage table, §14 Phase 9 entry). This document is the as-executed task
breakdown for that phase, authored the same way Phase 7/Phase 8's plan docs were.

## Objective

Typed tool contracts, the first bounded operational adapters (one read-only lookup, one mutating
action), tool allowlist filtering applied *before* the model sees tool definitions, the
`ApprovalRequest`/`ApprovalDecision`/`ApprovalPolicy` schema, LangGraph `interrupt()`/
`Command(resume=...)` wiring for durable human-in-the-loop approval, and a staged approval policy.

## Entry dependencies

Phase 8 (complete, commit `42e91e1`) and Phase 3 (ticketing core — `Ticket`, `AuditLog`,
state-machine module, both present).

## Exit criteria

- A restart-mid-approval test proves **zero duplicate mutation side effects**: kill the process
  after `interrupt()` fires (before resume), restart, resume via `Command(resume=...)`, confirm the
  mutating tool executed exactly once.
- Approval-tamper, approval-expiry, and duplicate-replay adversarial tests all pass.
- Tool-allowlist test passes: a principal without the required role/entitlement never receives the
  tool's definition in the first place (filtered before the model call, not rejected after).
- One real mutating tool works end-to-end through the real UI: proposal → durable approval request
  → approver decision → executed mutation → audit trail, with `docker compose down -v` fresh-state
  verification.

## Task breakdown

### Task 1 — Approval, tool-call, and entitlement schema

Add to `apps/api/src/resolvegrid_api/models/`:

- New file `approvals.py`:
  - `ApprovalRequest`: `id`, `ticket_id` (nullable FK to `ticket.id`), `agent_run_id` (nullable —
    no `AgentRun` table exists yet in this codebase; store as a plain string/UUID column, not a FK,
    with a comment noting Phase 10's `EvalRun`/telemetry work is expected to formalize this), `action_type`
    (str), `action_params_json` (str), `bound_evidence_refs_json` (str, nullable), `risk_context`
    (str, nullable), `status` (str, default `"pending"` — values: pending/approved/rejected/expired),
    `snapshot_hash` (str, unique — sha256 hex digest, see Task 5 for exact composition), `requested_by_id`
    (nullable FK employee), `expires_at` (datetime), `created_at`, `updated_at`.
  - `ApprovalDecision`: `id`, `approval_request_id` (FK), `approver_id` (FK employee), `decision`
    (str — approved/rejected), `comment` (str, nullable), `decision_evidence_snapshot_json` (str,
    nullable), `decided_at` (server_default now).
  - `ApprovalPolicy`: `id`, `action_type` (str, unique), `stages_json` (str — ordered list of
    `{role, scope}` stage requirements, e.g. peer-review then higher-authority), `description`
    (str, nullable).
- New file `tools.py`:
  - `ToolCall`: `id`, `agent_run_id` (string, same rationale as above), `tool_name`, `tool_version`,
    `input_params_json`, `output_json` (nullable), `status` (str — success/error/timeout/dry_run),
    `error_taxonomy_code` (str, nullable), `idempotency_key` (str, nullable, indexed),
    `approval_request_id` (nullable FK), `created_at`.
- Append to `org.py`: `AccessGroup` (`id`, `name` unique, `description` nullable),
  `Entitlement` (`id`, `access_group_id` FK, `name`, `description` nullable),
  `EmployeeEntitlement` (`id`, `employee_id` FK, `entitlement_id` FK, `granted_at` server_default
  now, `revoked_at` nullable, `source_ticket_id` nullable FK to `ticket.id`).

Write the Alembic migration (`0012_...`). Add indexes on `ApprovalRequest.status`,
`ApprovalRequest.snapshot_hash` (already unique so implicitly indexed), `ToolCall.idempotency_key`,
`EmployeeEntitlement.employee_id`.

No business logic in this task — schema + migration + a smoke test that every table round-trips a
row. Follow this codebase's existing model conventions exactly (see `ticketing.py`/`knowledge.py`
for the established style: `Mapped[...]`, `mapped_column`, `Index` in `__table_args__`, no
`metadata` attribute name collisions).

**Context for the implementer:** read `apps/api/src/resolvegrid_api/models/ticketing.py` and
`apps/api/src/resolvegrid_api/models/knowledge.py` first for the established model style. Read the
latest migration under `apps/api/alembic/versions/` (should be `0011_...`) to see the current head
revision to chain from.

### Task 2 — Typed tool contracts + two concrete tool definitions

In `packages/contracts/src/resolvegrid_contracts/`, add `tools.py`:

- `ToolContract` (pydantic `BaseModel` or dataclass, match whatever `tickets.py` in this same
  package already uses — read it first): `name`, `version`, `description`, `params_schema` (JSON
  Schema dict), `required_role` (str, e.g. `"analyst"`), `required_entitlement` (str, nullable),
  `mutating` (bool), `requires_approval` (bool).
- A small static registry, `TOOL_REGISTRY: dict[str, ToolContract]`, with exactly two entries for
  this phase:
  1. `lookup_employee_entitlements` — read-only, `mutating=False`, `requires_approval=False`,
     params: `{employee_id: int}`.
  2. `grant_vpn_access` — mutating, `requires_approval=True`, params:
     `{employee_id: int, justification: str}`. Chosen because it maps directly onto the seeded
     `Entitlement`/`AccessGroup` model from Task 1 and is a plausible, boring, realistic IT action
     (not a toy example) — matches plan.md's "no theatrical demo actions" instruction.

**Context for the implementer:** read `packages/contracts/src/resolvegrid_contracts/tickets.py` for
the existing contract style/pattern in this package before adding `tools.py`.

### Task 3 — entitlement lookup + VPN grant adapters

**Amended from the original plan-mode sketch (documented here, not silently changed):** the
plan-mode architecture doc's §5 layout names `services/operational-adapters/` as a top-level uv
workspace package. Building it that way here would require it to import `apps/api`'s
`Employee`/`EmployeeEntitlement`/`Entitlement`/`AccessGroup` ORM models directly (Task 1 put them in
`apps/api/src/resolvegrid_api/models/org.py`, and there is no DB-free way to query/insert them).
That is exactly the "app importing a library that imports back into the app" anti-pattern
`services/agent-orchestration/graph.py`'s module docstring documents and rejects, and it is the same
tension `apps/api/src/resolvegrid_api/knowledge_store.py`'s docstring already resolved once for
Phase 7: `services/retrieval` stays a pure, DB-free library; the SQLAlchemy-touching glue that turns
its output into real rows lives in `apps/api` instead, which `apps/api` already owns. Task 3 follows
that same established, twice-precedented resolution rather than the original directory sketch:
implement this as a **new module inside `apps/api`**, not a new workspace package. Add this as a new
Phase 9 Task 3 decision-log entry in Task 9's documentation pass.

New file `apps/api/src/resolvegrid_api/operational_adapters/entitlements.py` (new subpackage —
add an `__init__.py`):

- `lookup_employee_entitlements(session, employee_id: int) -> list[EntitlementSummary]` — read-only
  query against `EmployeeEntitlement`/`Entitlement`/`AccessGroup`, returns active (non-revoked)
  entitlements. `EntitlementSummary` is a small local dataclass/pydantic model (not the ORM row
  itself) — mirrors `retrieval.py`'s established pattern of returning plain result shapes rather
  than leaking ORM instances past the query boundary.
- `grant_vpn_access(session, employee_id: int, justification: str) -> EmployeeEntitlement` — inserts
  an `EmployeeEntitlement` row for the well-known `"VPN Access"` entitlement (create it via a small
  idempotent `ensure_vpn_entitlement_seeded(session)` helper, or seed it in `seed_corpus.py`/a new
  seed step — implementer's judgment, document the choice). Must be **idempotent**: granting VPN
  access to an employee who already has an active (non-revoked) grant is a no-op that returns the
  existing row, not a duplicate insert — this idempotency is what Task 9's restart-mid-approval test
  depends on being real, not assumed.

This module takes a SQLAlchemy `Session` as a plain parameter, exactly like
`apps/api/src/resolvegrid_api/knowledge_store.py` — no new workspace package, no new
`pyproject.toml`, no new CI job. `packages/contracts`'s `tools.py` (Task 2) already lives in a
package `apps/api` imports for the *contract shape*; the *adapter implementation* stays inside
`apps/api` itself, matching every other DB-touching module in this codebase (`knowledge_store.py`,
`ingestion.py`, `retrieval.py`).

**Context for the implementer:** read `apps/api/src/resolvegrid_api/knowledge_store.py` in full for
the established "plain function takes a Session, returns plain result shapes" pattern and its
documented dependency-direction rationale before writing this module — match that reasoning and
style exactly, don't re-derive it from scratch.

### Task 4 — Tool allowlist filtering + select/validate nodes

New module `apps/api/src/resolvegrid_api/tool_execution.py`:

- `available_tools_for_principal(principal: Principal) -> list[ToolContract]` — filters
  `TOOL_REGISTRY` down to tools whose `required_role`/`required_entitlement` the principal actually
  holds, using `packages/authz`'s `authorize()` (add a new `_SELF_SCOPED_ACTIONS` or dedicated check
  — implementer's judgment on the cleanest fit against the existing `policy.py` shape; do not
  bypass `authorize()` with ad hoc role-string comparisons). **This filtering must happen before any
  tool list is exposed to the model** — i.e. this function is called first, and only its result is
  ever formatted into a prompt or passed to `select_tool`.
- `select_tool(tool_name: str, available: list[ToolContract]) -> ToolContract` — raises a typed
  `ToolNotAllowedError` if `tool_name` isn't in `available` (covers both "doesn't exist" and "not
  permitted" identically — never leak which case it was, per plan.md's safe-error-envelope
  instruction).
- `validate_tool_schema(tool: ToolContract, params: dict) -> dict` — validates `params` against
  `tool.params_schema` (use `jsonschema` if not already a dependency, or a minimal hand-rolled
  required-keys+type check if adding a new dependency isn't warranted — implementer's judgment,
  document the choice), raises a typed `ToolValidationError` on mismatch.

This task does NOT wire these into the LangGraph graph yet — that's Task 5/6. This task is the
allowlist/validation logic plus unit tests proving: (a) a principal missing the required role never
sees the mutating tool in `available_tools_for_principal`'s output, (b) `select_tool` on a filtered
tool name raises rather than falling through, (c) schema validation rejects wrong types/missing
required params.

**Context for the implementer:** read `packages/authz/src/resolvegrid_authz/policy.py` in full
(already shown above in this plan's research — reproduce it for the implementer verbatim if
dispatching without file access assumptions) and `apps/api/src/resolvegrid_api/retrieval_authz.py`
for how a prior task translated an `authorize()` `Decision` into a concrete filter.

### Task 5 — `request_approval` node: durable `interrupt()`, snapshot hash, idempotent upsert

In `services/agent-orchestration`, add:

- `AgentState` fields (in `state.py`, following the exact documentation-density precedent of every
  existing field — each new field gets a comment explaining what populates it and what consumes
  it): `proposed_tool_name: str | None`, `proposed_tool_params: dict | None`,
  `approval_request_id: int | None`, `approval_decision: str | None` (approved/rejected/None while
  pending).
- `request_approval` node (`graph.py`): when the classified/selected tool `requires_approval`, calls
  an injected `RequestApprovalFn` (same DI pattern as `CompleteFn`/`RetrieveFn` — this package must
  not import `apps/api`'s DB code directly, per the module docstring's established dependency-
  direction rule) that:
  1. Computes `snapshot_hash = sha256(json.dumps({action_type, params (sorted keys), actor:
     principal_employee_id, evidence_refs, risk_context, expires_at}, sort_keys=True))`. Document
     the exact field set in a docstring — this hash is re-verified byte-for-byte at execution time
     (Task 6), so its composition must be exact and stable.
  2. **Idempotent upsert**: if an `ApprovalRequest` with this exact `snapshot_hash` already exists
     (e.g. this node re-executes because LangGraph resumed from an earlier checkpoint after a
     restart), return the existing row's id rather than inserting a second one. This is the
     concrete mechanism that makes the node safe to re-execute — required reading:
     https://langchain-ai.github.io/langgraph's documented re-execution caveat around `interrupt()`
     nodes (this codebase already handled an analogous caveat in Phase 6's checkpoint-restore test
     — read `services/agent-orchestration/tests/` for that established test pattern before writing
     Task 9's restart test).
  3. Calls LangGraph's `interrupt()` with a payload describing the pending decision, then on
     resume reads `Command(resume=...)`'s value to populate `approval_decision`.
- `RequestApprovalFn` type + real implementation in `apps/api` (new file
  `apps/api/src/resolvegrid_api/approval_service.py`): builds the snapshot hash, does the upsert
  against `ApprovalRequest` via SQLAlchemy, sets `expires_at` (pick a concrete default, e.g. 24h,
  document it and where it'd move to `ApprovalPolicy` config in a later phase if needed).

**Context for the implementer:** read `services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`
and `state.py` in full first (both reproduced above in this plan) — match their exact documentation
density and the `CompleteFn`/`RetrieveFn` dependency-injection pattern precisely; do not deviate
from the established style. Read LangGraph's `interrupt()`/`Command(resume=...)` API docs
(WebFetch `https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/` if needed) before
implementing — this is the first task in this codebase to use `interrupt()` for real (Phase 6 only
wired the checkpointer, it never actually called `interrupt()`).

### Task 6 — `execute_mutation` post-interrupt + `execute_readonly_tool` direct path

- `execute_readonly_tool` node: for non-mutating tools, calls the tool directly (no approval), logs
  a `ToolCall` row (`status="success"`/`"error"`), returns its result into state.
- `execute_mutation` node: placed **strictly after** the `request_approval`/resume boundary.
  1. Re-fetches the `ApprovalRequest` by `approval_request_id`.
  2. **Re-verifies the snapshot hash** by recomputing it from the request's current stored fields
     and comparing — if it doesn't match (shouldn't be reachable given Task 5's design, but this is
     the actual tamper defense plan.md requires, not a formality), raise `ApprovalTamperError` and
     write a `ToolCall` row with `status="error"`, `error_taxonomy_code="ApprovalTamperError"`.
  3. Checks `status == "approved"` (not pending/rejected) and `expires_at > now()` — else raise
     `ApprovalExpiredError` or a rejection-appropriate response, again logged.
  4. **Duplicate-replay guard**: uses `ToolCall.idempotency_key = f"approval:{approval_request_id}"`
     — if a `ToolCall` with that key already has `status="success"`, return its recorded
     `output_json` instead of re-executing the mutating adapter call. This is what makes Task 9's
     restart-mid-approval test provable, not assumed.
  5. Only after all of the above: calls the real mutating adapter (`grant_vpn_access` from Task 3),
     records the `ToolCall` row, writes an `AuditLog` row (reuse the existing hash-chained
     `AuditLog` model from Phase 3 — `before_json`/`after_json` around the entitlement grant).

**Context for the implementer:** read Task 5's output before starting (this task cannot proceed
without `request_approval`'s exact `ApprovalRequest`/snapshot-hash shape). Read
`apps/api/src/resolvegrid_api/audit.py` for the established `AuditLog` hash-chain write pattern
before adding a new write site.

### Task 7 — Tool-invocation graph wiring, approver API, and minimal approver UI

**Amended from the original plan-mode sketch (a real gap found while executing, documented here,
not silently papered over):** Tasks 4, 5, and 6 each correctly, deliberately declined to wire
`request_approval`/`execute_mutation` into `build_graph`'s real conditional routing — each task's
own docstring says so explicitly, and each was right to defer it, since inventing that routing logic
wasn't in any of their stated scopes. But this means that, as of Task 6's commit, there is still NO
real, running LangGraph graph anywhere in this app that ever actually calls `interrupt()` for real —
`request_approval`/`execute_mutation` exist only as directly-callable, directly-testable functions.
Task 7's original text ("resumes the paused graph run via `Command(resume=...)`") assumed a paused
run already existed to resume, which isn't true yet. Task 8's restart-mid-approval test (the phase's
core exit criterion) needs a REAL paused `interrupt()` run in the actual app to restart against, not
just a test harness's ad hoc graph (which is what Task 5's own unit tests already used, correctly,
for Task 5's narrower scope). So Task 7 now explicitly includes closing this gap, split into two
parts:

#### Task 7a — the tool-invocation graph + invoke endpoint

Rather than teaching the existing `/chat` graph (`classify_intent -> retrieve -> compose_response ->
verify_citations -> finalize`) to infer tool intent from free-form chat text — a much bigger,
genuinely separate undertaking (real LLM-driven tool selection, stage 8's actual scope) that no task
in this phase specifies and that would risk regressing the well-tested existing chat flow — build a
small, SEPARATE, dedicated graph purely for explicit tool invocation, matching how a real IT-analyst
UI actually works (an analyst deliberately chooses "grant this employee VPN access," they don't type
free text and hope an LLM infers it).

In `services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`, add
`build_tool_invocation_graph(checkpointer, request_approval_fn: RequestApprovalFn,
execute_mutation_fn: ExecuteMutationFn)` — a real, separate compiled graph with exactly two nodes:
`request_approval` (Task 5's existing node builder — reuse it, don't reimplement) and a new
`execute_mutation` node wired after it. Add `ExecuteMutationFn = Callable[[dict], dict]` (mirroring
`RequestApprovalFn`'s DI shape) — its real implementation lives in `apps/api` (new function in
`apps/api/src/resolvegrid_api/approval_service.py` or a new small module, your judgment) that opens
its own session (mirroring `agent_retrieval.py`/`request_approval_for_agent`'s "closure built once,
opens its own session per call" pattern) and calls Task 6's real `execute_mutation`/
`execute_readonly_tool`. The `execute_mutation` graph node reads `state["approval_request_id"]`,
`state["approval_decision"]`, `state["proposed_tool_name"]`, `state["proposed_tool_params"]` — if
`approval_decision == "approved"`, calls `execute_mutation_fn`; if `"rejected"`, records that
outcome without calling it; either way sets an `output_text`/result field on state. This graph is
compiled with the SAME real `AsyncPostgresSaver` checkpointer as the main chat graph (both graphs can
share one checkpointer instance — confirm this is fine by checking how `AsyncPostgresSaver` scopes
its tables; if graphs need distinguishing, LangGraph threads are already scoped by `thread_id`, which
this task controls).

New endpoint in a new file, `apps/api/src/resolvegrid_api/routers/tools.py`:
`POST /tools/{tool_name}/invoke` (body: `{params: dict}`) —
1. `available_tools_for_principal` (Task 4) to get the principal's allowed tools, `select_tool` to
   resolve `tool_name` against that list (403/404-equivalent via `ToolNotAllowedError` if not
   allowed), `validate_tool_schema` to validate `params`.
2. If the resolved tool is NOT `requires_approval`: call `execute_readonly_tool` (Task 6) directly,
   return the result — no graph, no interrupt, matches Task 6's own "read-only tools skip the
   approval machinery" design.
3. If it IS `requires_approval`: generate a fresh `thread_id` (mirror `chat.py`'s
   `uuid4().hex` pattern), build the initial state for `build_tool_invocation_graph`'s compiled
   graph (`proposed_tool_name`, `proposed_tool_params`, `principal_employee_id`, plus whatever
   `request_approval`'s payload needs — `action_type`, evidence refs, risk context), `ainvoke()` it.
   This call will hit `interrupt()` and return with `"__interrupt__"` in the result rather than a
   final state — confirm this from the real LangGraph behavior Task 5 already verified against the
   installed version (`langgraph==1.2.11`) and handle it correctly (don't treat the interrupted
   return shape as if it were a completed run). Return the created `approval_request_id` and a
   `"pending_approval"` status to the caller, along with `thread_id` (needed by Task 7b's resume
   call — store it somewhere `GET`/`POST /approvals/...` can find it again: the natural place is a
   new nullable column or reuse of `ApprovalRequest.agent_run_id`, since that's already the plain
   string thread/run identifier `ApprovalRequest`'s own docstring documents — confirm
   `request_approval_for_agent`'s payload already threads `agent_run_id` through to the row, and use
   THIS graph's `thread_id` as that value; if it doesn't yet, that's a legitimate small addition here).

Tests: an integration test that calls the invoke endpoint for `grant_vpn_access`, confirms a real
`ApprovalRequest` row exists with `status="pending"`, and confirms the underlying graph is genuinely
paused at `interrupt()` (LangGraph exposes a way to check a thread's pending interrupts via the
checkpointer/graph's `get_state()` — use the real API, verified against the installed version, not
assumed). A read-only-tool invoke test (`lookup_employee_entitlements`) proving it returns
immediately with no `ApprovalRequest` created.

#### Task 7b — Approver API + minimal approver UI surface (depends on 7a)

- `apps/api/src/resolvegrid_api/routers/approvals.py`: `GET /approvals` (list pending, filtered by
  the approver's department/role via `authorize()` — new `"approval.list"`/`"approval.decide"`
  actions in `packages/authz/policy.py`, following the exact `_SELF_SCOPED_ACTIONS`/
  `_STAFF_ONLY_ACTIONS` pattern already established — `approval.decide` should be staff-only, never
  self-scoped, matching `ticket.transition`'s precedent), `POST /approvals/{id}/decide` (body:
  `{decision: "approved"|"rejected", comment}` — writes `ApprovalDecision`, updates
  `ApprovalRequest.status`, then resumes the paused graph run for real: fetch the row's
  `agent_run_id` (== `thread_id`), call the SAME compiled `build_tool_invocation_graph` graph's
  `ainvoke(Command(resume=payload.decision), config={"configurable": {"thread_id": agent_run_id}})` —
  this is what actually drives execution past `interrupt()` into the `execute_mutation` node for
  real, closing the loop Task 7a opened).
- `apps/web/app/approvals/page.tsx`: minimal approver surface — list of pending approval requests
  with action type, params, risk context, evidence refs, requester, expiry countdown; approve/reject
  buttons with a required comment field on reject. Follow the existing `apps/web/app/chat/page.tsx`
  for this codebase's established minimal-UI style/conventions (no design-system polish expected
  yet — that's Phase 14). Since there's no tool-invocation UI yet (Task 7a only added an API), also
  add a minimal `apps/web/app/tools/page.tsx` (or fold into the approvals page) letting an analyst
  fill in `employee_id`/`justification` and POST to `/tools/grant_vpn_access/invoke` — otherwise the
  approver UI has no real way to generate a pending approval to demonstrate end-to-end. Keep this
  genuinely minimal (a form, not a tool browser) — the two-tool registry doesn't warrant more.

**Context for the implementer (both parts):** read `apps/api/src/resolvegrid_api/routers/chat.py` in
full (graph invocation pattern, `thread_id` generation, error handling shape) and
`apps/web/app/chat/page.tsx` (frontend conventions) before starting. Read Task 5's
`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py` and Task 6's
`apps/api/src/resolvegrid_api/mutation_execution.py` in their CURRENT committed state (don't rely on
older reproductions elsewhere in this doc) before writing any wiring code.

### Task 8 — Adversarial tests: restart-mid-approval, tamper, expiry, duplicate-replay, allowlist

This is the phase's primary proof artifact, mirroring Phase 7/8's adversarial-test discipline.

- **Restart-mid-approval test** (the exit-criteria test): start a graph run that reaches
  `request_approval` and calls real `interrupt()`; simulate a process restart by tearing down and
  rebuilding the graph/checkpointer against the same Postgres-backed `AsyncPostgresSaver` thread
  (do not just re-call the same Python object — that wouldn't prove anything about durability);
  resume via `Command(resume=...)`; assert exactly one `ToolCall` row with `status="success"` exists
  for that approval, and exactly one `EmployeeEntitlement` grant exists (not two).
- **Tamper test**: mutate an `ApprovalRequest`'s stored params after it was created but before
  decision; confirm `execute_mutation` detects the hash mismatch and refuses.
- **Expiry test**: set `expires_at` in the past; confirm rejection with the correct error taxonomy
  code, no mutation executed.
- **Duplicate-replay test**: call `execute_mutation` twice for the same already-approved,
  already-executed request; confirm the second call returns the recorded result without a second
  adapter call/side effect (assert the adapter mock/spy was called exactly once, or assert the DB
  row count directly against a real adapter call — prefer the real DB-row-count assertion per this
  codebase's established real-verification discipline over mocking).
- **Tool-allowlist test**: a principal without `grant_vpn_access`'s required role/entitlement never
  receives it from `available_tools_for_principal`; attempting `select_tool("grant_vpn_access", ...)`
  against their filtered list raises.

Run the FULL existing suite before this task adds any new seed/fixture data (same "empty-corpus"
discipline documented in `docs/DECISION_LOG.md`'s 2026-08-28 entry — confirm this task's new tests
don't reintroduce that class of interference; if they need seed data, seed it inside the test's own
transaction/fixture, not via the shared `seed_corpus`/ingestion path).

**Context for the implementer:** read `docs/DECISION_LOG.md`'s entry about the empty-corpus test
constraint before writing fixtures. Read
`services/agent-orchestration/tests/test_injected_document_adversarial.py` for this codebase's
established adversarial-test documentation style (docstring records the real finding/fix history) —
match it if any of these tests surface a real bug (expected, given precedent).

### Task 9 — Documentation + fresh-state verification

- `docs/WORKFLOWS_TOOLS.md`: tool catalog (both `ToolContract` entries), approval-policy staging
  explanation.
- `docs/SECURITY.md`: new "Approval binding" section — snapshot hash composition, tamper/expiry/
  replay defenses, cross-reference to Task 8's adversarial tests.
- `docs/RUNBOOKS.md`: new file (first use of this doc per §13 of the architecture plan) — a
  compensation runbook for "VPN access was granted in error" (how to revoke, what audit trail to
  check).
- `docs/DECISION_LOG.md`: entries for (a) the snapshot-hash field-set choice, (b) idempotent-upsert-
  by-hash as the re-execution defense (vs. alternatives considered), (c) duplicate-replay-via-
  idempotency-key as the mutation-side defense, (d) any real bug found during Task 8.
- `docs/PROGRESS.md`: Phase 9 row(s), state `Verified`, exit-criteria evidence links (test names/
  commit SHAs).
- Fresh-state verification: full `docker compose down -v`, rebuild, migrate, run the full suite
  (respecting the empty-corpus-first ordering), then a real Playwright browser walkthrough of the
  approver UI end-to-end (propose → approve → verify entitlement granted) before any seeding that
  would break the untested-last-step rule.
- Push, watch CI green (6/6 or however many jobs now exist — confirm `services/operational-adapters`
  needs its own CI job added to `.github/workflows/ci.yml`, mirroring `retrieval`'s job).

**Context for the implementer:** read `docs/DECISION_LOG.md` and `docs/PROGRESS.md`'s existing
Phase 8 rows for the established format before adding Phase 9's.

## Notes for whoever executes this plan

- This phase is the first to touch real state-mutating side effects end-to-end through the agent
  graph — treat Task 6/8 with the same security rigor Phase 8 applied to prompt injection and the
  distractor regression: real, non-mocked verification, not trust in a subagent's self-report.
- Tasks 1-4 have no ordering dependency on each other beyond "schema before adapters before
  contracts-consumers" — 1 → 3, 2 is independent of 1/3, 4 depends on 2. Tasks 5-9 are strictly
  sequential.
- Follow this codebase's extremely high documentation-density convention in every new module
  (visible throughout `graph.py`/`state.py` above) — this is a deliberate, established project
  style, not incidental verbosity.
