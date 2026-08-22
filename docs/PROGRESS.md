# Progress Ledger

The single authoritative status ledger. States: Proposed / Approved / In Progress / Blocked / Implemented / Verified / Superseded. No status is duplicated anywhere else — other docs link here.

| Unit ID | Description | State | Entry Dependencies | Exit-Criteria Evidence | Last Updated |
|---|---|---|---|---|---|
| Phase 1 | Repo scaffold & local infra skeleton | Verified | none | `scripts/smoke_test.sh` — ALL SMOKE CHECKS PASSED on a genuine cold start (`docker compose down -v`) and a subsequent warm run, re-verified 2026-08-21 after fixing an OTel-span-flush regression (a forceful process kill was destroying the span before its periodic batch-export could flush it) and a Windows PID-resolution bug | 2026-08-21 |
| Phase 2 | Identity, org data model & seeded directory | Verified | Phase 1 | `apps/api/tests` 19 passed, `packages/telemetry/tests` 2 passed, `packages/authz/tests` 9 passed (30 total); `pnpm --filter @resolvegrid/web lint`/`typecheck`/`build` all clean, Next.js production build succeeded; `scripts/seed_employees.sh --seed 42 --employees 75` seeded 75 employees against the live Postgres container; manual curl spot-check against a running `uvicorn` instance confirmed role-based restriction with real seeded data — admin (employee 1175) received all 75 employees from `GET /directory/employees`, a no-grant employee (id 1193) received exactly 1 record (themselves), and a request with no `X-Debug-Employee-Id` header was rejected with 401 | 2026-08-22 |
