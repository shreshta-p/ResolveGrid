# Runbooks

First real runbook, added in Phase 9 alongside the first mutating tool
(`grant_vpn_access`). No revoke/rollback endpoint was built this phase —
this is honestly flagged below as tracked future work, not silently
omitted.

## VPN access granted in error

**Scenario**: `grant_vpn_access` executed successfully (an approver approved
a request they shouldn't have, or the justification turns out to be
invalid/fraudulent after the fact) and an employee now holds a real,
active `EmployeeEntitlement` row for the well-known `"VPN Access"`
entitlement that needs to be revoked.

### What does NOT exist yet

There is no self-service revoke endpoint or UI action anywhere in this
codebase as of Phase 9. `apps/api/src/resolvegrid_api/operational_adapters/entitlements.py`'s
`grant_vpn_access`/`ensure_vpn_entitlement_seeded` only ever grant; nothing
calls `EmployeeEntitlement.revoked_at` anywhere in the app. This is a real,
documented gap — see "Tracked future work" below — not an oversight this
runbook is pretending doesn't exist.

### Remediation: direct database revoke

Until a real revoke endpoint exists, an admin/approver with direct database
access performs the revoke manually:

1. **Identify the grant.** Find the `EmployeeEntitlement` row for the
   affected employee and the `"VPN Access"` entitlement (joined through
   `Entitlement`/`AccessGroup` — see `entitlements.py`'s
   `VPN_ENTITLEMENT_NAME`/`VPN_ACCESS_GROUP_NAME` constants for the exact
   seeded names, `"VPN Access"` / `"Network Access"`):

   ```sql
   SELECT ee.id, ee.employee_id, ee.entitlement_id, ee.granted_at, ee.revoked_at
   FROM employee_entitlement ee
   JOIN entitlement e ON e.id = ee.entitlement_id
   WHERE e.name = 'VPN Access'
     AND ee.employee_id = :employee_id
     AND ee.revoked_at IS NULL;
   ```

2. **Revoke it.** Set `revoked_at` to the current time — do NOT delete the
   row. `lookup_employee_entitlements` (the read-only tool) and
   `grant_vpn_access`'s own idempotency check both filter on
   `revoked_at IS NULL`, so a revoked row is automatically excluded from
   "active entitlements" everywhere in the app without needing a schema
   change or a code deploy:

   ```sql
   UPDATE employee_entitlement
   SET revoked_at = now()
   WHERE id = :employee_entitlement_id;
   ```

   After this, `lookup_employee_entitlements(session, employee_id)` will no
   longer return this grant, and a future `grant_vpn_access` call for the
   same employee will correctly treat them as not currently holding it
   (i.e. it will grant a *new* row rather than silently no-op against the
   revoked one — `grant_vpn_access`'s idempotency check is scoped to
   `revoked_at IS NULL`).

3. **Confirm the revoke took effect** by re-running the SELECT in step 1 —
   `revoked_at` should now be non-null. Optionally also confirm via the
   application layer: an analyst calling `lookup_employee_entitlements` for
   this employee (via `POST /tools/lookup_employee_entitlements/invoke`)
   should no longer see the "VPN Access" entry.

### What to check in the audit trail

Every piece of the chain that led to the grant is real and queryable —
this is exactly what Phase 3's hash-chained `AuditLog` and Phase 9's
`ApprovalRequest`/`ApprovalDecision`/`ToolCall` schema exist to make
possible:

1. **`AuditLog`** — the actual mutation record. `mutation_execution.execute_mutation`
   writes this via `record_audit_event(actor_type="agent", action=f"tool.{tool_name}",
   ...)`, so the exact action string to search for is `action = 'tool.grant_vpn_access'`,
   `entity_type = 'employee_entitlement'`, `entity_id = <employee_entitlement_id>`.
   `before_json`/`after_json` show the before/after state
   (`{"employee_id": ..., "had_vpn_access": false}` ->
   `{"employee_id": ..., "had_vpn_access": true, "employee_entitlement_id": ...,
   "justification": "..."}` — the justification text the requester supplied
   is captured here, since `EmployeeEntitlement` itself has no column for
   it). `metadata_json` carries `approval_request_id`/`tool_call_id`, the
   join keys back to the rest of the chain below. Run
   `resolvegrid_api.audit.verify_chain_integrity()` first if tampering is
   suspected — it will detect any row altered after the fact.
2. **`ToolCall`** — find the row via `metadata_json.tool_call_id` from the
   `AuditLog` row above, or directly via
   `idempotency_key = 'approval:<approval_request_id>'` and
   `status = 'success'`. Confirms exactly what params were actually
   dispatched (`input_params_json`) and what the adapter returned
   (`output_json`).
3. **`ApprovalRequest`** / **`ApprovalDecision`** — the human-in-the-loop
   trail: who requested it (`ApprovalRequest.requested_by_id`), the exact
   bound params/risk context/expiry (`action_params_json`/`risk_context`/
   `expires_at`), and who approved it and when
   (`ApprovalDecision.approver_id`/`decided_at`/`comment`). Together with
   the `AuditLog`/`ToolCall` rows above, this fully reconstructs "who asked
   for what, who approved it, and what actually happened" for any granted
   entitlement — the complete picture needed to determine whether a given
   grant really was an approver error (and thus this runbook's remediation
   applies) versus, say, a since-changed business need (a different,
   non-security remediation).

### Tracked future work

A real `POST /entitlements/{id}/revoke` (or similar) endpoint — mirroring
`grant_vpn_access`'s own approval-gated pattern (revoking access is itself
arguably a mutating, auditable action, plausibly approval-gated the same
way granting it is) — is genuine, undone work, not a silent gap. It was
out of scope for Phase 9, whose exit criteria were about proving the
grant path end-to-end, not building the inverse operation. Revisit trigger:
before any non-local/shared deployment where manual SQL access to
production is not an acceptable normal operating procedure.

## Background job workers (Arq) — none run as a persistent container

**This codebase has no persistent Arq-worker container.** `infra/docker
-compose.yml` runs Postgres/Redis/Ollama/LiteLLM/the observability stack,
but every real Arq `WorkerSettings` in `apps/api` —
`resolvegrid_api.ingestion_worker` (seed-corpus ingestion),
`resolvegrid_api.eval_worker` (eval-suite batch runs), and
`resolvegrid_api.cache_maintenance_worker` (Phase 12 Task 6: hourly
semantic-response-cache purge) — must be started manually, as its own
long-running process, for its jobs to ever actually execute:

```bash
uv run --package resolvegrid-api arq resolvegrid_api.ingestion_worker.WorkerSettings
uv run --package resolvegrid-api arq resolvegrid_api.eval_worker.WorkerSettings
uv run --package resolvegrid-api arq resolvegrid_api.cache_maintenance_worker.WorkerSettings
```

**Operational consequence an on-call engineer must know**: if nobody is
running the third command above, `semantic_response_cache` (Postgres) grows
unbounded forever — `lookup_semantic_cache`'s own 15-minute TTL filter stops
a stale row from being *matched*, but only the `cache_maintenance_worker`
cron job (fires hourly, plus once at worker startup) actually *deletes*
expired rows. A slow-growing `semantic_response_cache` table with no
Arq-worker process visible in `ps`/your process manager is the real symptom
to check for — see `docs/DECISION_LOG.md`'s 2026-10-20 entry and
`cache_maintenance_worker.py`'s own module docstring for the full reasoning
(why Arq cron, why no persistent container was stood up for it this phase).
Each of the three workers now also owns its own real Arq queue name
(`QUEUE_NAME` in each module — `resolvegrid:ingestion` /
`resolvegrid:eval` / `resolvegrid:cache-maintenance`) rather than sharing
Arq's single library-default queue, so starting one does not risk it
dequeuing and failing a job meant for another.

Before any non-local/shared deployment: stand up each of these three as a
real persistent process (a dedicated container/systemd unit/supervisor
entry per worker, or one process with all three `WorkerSettings` merged —
not decided yet, tracked future work) — manual-start-per-dev-session is not
an acceptable production operating model, the same revisit trigger the VPN
-grant runbook above already names for its own manual-SQL remediation step.
