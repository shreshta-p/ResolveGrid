# Progress Ledger

The single authoritative status ledger. States: Proposed / Approved / In Progress / Blocked / Implemented / Verified / Superseded. No status is duplicated anywhere else — other docs link here.

| Unit ID | Description | State | Entry Dependencies | Exit-Criteria Evidence | Last Updated |
|---|---|---|---|---|---|
| Phase 1 | Repo scaffold & local infra skeleton | Verified | none | `scripts/smoke_test.sh` — ALL SMOKE CHECKS PASSED on a genuine cold start (`docker compose down -v`) and a subsequent warm run, re-verified 2026-08-21 after fixing an OTel-span-flush regression (a forceful process kill was destroying the span before its periodic batch-export could flush it) and a Windows PID-resolution bug | 2026-08-21 |
