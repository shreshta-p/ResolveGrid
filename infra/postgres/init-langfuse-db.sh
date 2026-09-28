#!/usr/bin/env bash
# Runs automatically on a genuinely fresh postgres_data volume (the base
# postgres image executes everything in /docker-entrypoint-initdb.d/ once,
# only when the data directory is empty -- it will NOT re-run against an
# existing volume, so this has no effect on an already-initialized database;
# see init-litellm-db.sh's own docstring, run alongside this one).
#
# Langfuse (Phase 11 Task 5) must NOT share the resolvegrid database with
# apps/api either, for the exact same reason init-litellm-db.sh's
# CREATE DATABASE litellm exists: Langfuse's web/worker services run their
# own Prisma migrations against whatever database LANGFUSE_DATABASE_URL
# points to, and Prisma's migrate-deploy baselines against the target
# database's existing schema -- pointing it at "resolvegrid" (Alembic-owned,
# no _prisma_migrations table) risks the same class of destructive
# unrecognized-table diff already proven to happen once with LiteLLM (see
# init-litellm-db.sh). A dedicated, empty "langfuse" database sidesteps this
# entirely: Prisma only ever sees its own tables there.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE langfuse OWNER $POSTGRES_USER;
EOSQL
