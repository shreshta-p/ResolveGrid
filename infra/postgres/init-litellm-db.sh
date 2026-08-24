#!/usr/bin/env bash
# Runs automatically on a genuinely fresh postgres_data volume (the base
# postgres image executes everything in /docker-entrypoint-initdb.d/ once,
# only when the data directory is empty -- it will NOT re-run against an
# existing volume, so this has no effect on an already-initialized database).
#
# LiteLLM must NOT share the resolvegrid database with apps/api: its Prisma
# startup baselines against whatever database it's pointed at, and when it
# finds an existing schema with no _prisma_migrations table (exactly what
# apps/api's Alembic-managed database looks like), it generates and executes
# a diff that includes DROP TABLE for every table it doesn't recognize --
# this actually happened during Phase 4 Task 2's development and destroyed
# every apps/api table before it was caught and the data was regenerated
# from seed.py. A dedicated database is the fix, not a workaround.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE litellm OWNER $POSTGRES_USER;
EOSQL
