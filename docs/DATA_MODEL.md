# Data Model

Schema shapes are defined in the approved architecture plan (§2). As of Phase 1, only `health_check` exists (Alembic baseline migration, `apps/api/migrations/versions/0001_baseline_health_check.py`) — a placeholder-free smoke-test table, not a domain entity. Org/ticketing/knowledge/approval/telemetry tables land in Phases 2, 3, 4, 7, 9.
