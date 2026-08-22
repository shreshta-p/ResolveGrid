# Data Model

Full schema shapes are defined in the approved architecture plan (`docs/PLAN_APPROVED.md` §2).

## Org / identity (Phase 2)

- `Location` — id, name (unique), region, timezone.
- `Department` — id, name (unique), parent_department_id (self-ref, nullable), cost_center (nullable).
- `Team` — id, name, department_id (FK), lead_employee_id (FK to employee, nullable).
- `Employee` — id, display_name, email (unique), title, employment_status, hire_date, termination_date (nullable), timezone, location_id/department_id/team_id (FKs, nullable), manager_id (self-ref FK, nullable, indexed). Manager reassignment must go through `resolvegrid_api.org.cycle_guard.would_create_cycle()` before being applied — no code path is expected to set `manager_id` directly without that check once mutation endpoints exist (none do yet in Phase 2; only the seed generator writes `manager_id`, and it constructs a shallow, guaranteed-acyclic tree by direct construction rather than needing the guard).
- `RoleAssignment` — id, employee_id (FK, indexed), role (plain string: "admin" | "analyst" | "approver" | employee has none), scope ("global" | "department" | "team"), scope_id (nullable), granted_by_id (FK, nullable), granted_at, expires_at (nullable).

Seeded via `resolvegrid_api.seed.generate_org()` (`scripts/seed_employees.sh`) — deterministic per seed, idempotent (wipes and rebuilds these 5 tables). Phase 1's `health_check` smoke-test table was retired in migration `0002` now that real schema exists.

Note: `Employee.team_id` and `Team.lead_employee_id` are mutually circular FKs (an Employee belongs to a Team, a Team can have an Employee as its lead). `Employee.team_id`'s FK is declared with `use_alter=True` in the model to prevent this from breaking Alembic autogenerate; any code that deletes/wipes both tables must null one side of the reference first (see `resolvegrid_api.seed.generate_org`'s wipe logic for the reference implementation).

Ticketing/knowledge/approval/telemetry tables land in Phases 3, 4, 7, 9 — see the approved architecture plan §2 for their full shapes.
