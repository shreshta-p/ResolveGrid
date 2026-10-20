"""Arq-backed cache-maintenance worker (Phase 12 Task 6, `docs/superpowers/
plans/2026-09-29-phase12-caching.md`'s Task 6 section: "Schedule the
semantic-cache purge function for real").

`resolvegrid_api.semantic_cache.purge_expired_semantic_cache_entries` has
existed and been fully tested since Task 5 (`apps/api/tests/
test_semantic_cache.py`'s `test_purge_deletes_only_rows_older_than_the_ttl_
cutoff` and its siblings), but was never actually CALLED from anywhere real
-- that function's own docstring explicitly names this as the gap this
module closes: "Not wired to run automatically anywhere yet -- this is the
real, callable mechanism Task 6 will schedule."

Why Arq `cron`, not a new scheduler technology: `apps/api/src/resolvegrid_
api/ingestion_worker.py` already established this codebase's one and only
real async-job mechanism (Arq, backed by the same shared `resolvegrid-redis`
container every cache module in this phase already uses) -- confirmed, by
reading `arq.cron`/`arq.worker.Worker` source directly (version 0.28.0, this
repo's pinned `uv.lock` version) before writing anything here, that Arq
genuinely supports real periodic (not just on-demand-enqueued) jobs via
`arq.cron.cron(...)` + `WorkerSettings.cron_jobs` -- a worker process's own
heartbeat loop (`Worker.heart_beat` -> `run_cron`) automatically enqueues a
registered `CronJob` whenever its schedule next matches, with no separate
enqueue call from application code required. Both existing Arq entry points
in this codebase (`ingestion_worker.WorkerSettings`, `eval_worker.
WorkerSettings`) are strictly on-demand today (`functions=[...]` only, no
`cron_jobs=`) -- this module is the first real user of Arq's cron mechanism
here, not a reach for a second scheduler technology this task's own
instructions warn against (no Celery beat / APScheduler introduced).

A DEDICATED new `WorkerSettings`/module, not added to `ingestion_worker.
WorkerSettings` or `eval_worker.WorkerSettings`: this is a periodic
*maintenance* job with no topical relationship to either existing worker's
own batch concern (seed-corpus ingestion; eval-suite execution) -- folding
it into either would blur exactly the one-module-per-concern convention
this codebase's cache modules themselves already established (see `retrieval_
cache.py`'s own module docstring for why that convention exists). A real
operational cost of this choice, stated honestly: a human must separately
start this worker process (`uv run --package resolvegrid-api arq
resolvegrid_api.cache_maintenance_worker.WorkerSettings`) for the cron
schedule below to ever actually fire continuously -- but that is true of
EVERY Arq worker in this codebase as of Phase 12 Task 6 (`infra/docker
-compose.yml` runs no persistent Arq-worker container for ingestion or eval
either; both are started manually per their own module docstrings), so this
is not a new operational gap this task introduces, only the first time
anything in this codebase needs to be left running continuously rather than
invoked once per batch. Recorded in `docs/DECISION_LOG.md`.

Schedule: `minute=0` (`hour`/`day`/`weekday`/`month` all left `None`, i.e.
"match any") -- fires once every real hour, on the hour, per `arq.cron.cron`'s
matching semantics (a field left `None` matches unconditionally; only `minute`
is constrained). Chosen deliberately relative to `semantic_cache.
SEMANTIC_CACHE_TTL_SECONDS` (15 minutes): a row becomes unmatchable by
`lookup_semantic_cache` after 15 minutes regardless of when this job next
runs (that staleness bound is already enforced by the lookup's own
`created_at` filter, independent of purging), so this job's only real job is
bounding the table's live row COUNT, not correctness -- a 1-hour cadence
keeps at most ~4 TTL-windows' worth of expired-but-undeleted rows
accumulated between passes (a small, bounded multiple of this cache's actual
write rate), while not running a DELETE against a `created_at`-indexed table
more often than real operational value justifies. `run_at_startup=True`
additionally runs one real purge pass the moment this worker process starts
(useful operationally -- a freshly started worker doesn't wait up to an hour
for its first pass -- and is also what makes this mechanism provable in a
single real burst-mode test cycle; see `test_cache_maintenance_worker.py`).
"""

import os

from arq.connections import RedisSettings
from arq.cron import cron
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.semantic_cache import purge_expired_semantic_cache_entries

# Same env var name/default every other cache/Arq module in this codebase
# uses (`cache.py`'s `REDIS_URL`, `ingestion_worker.py`'s `REDIS_URL`) --
# re-declared here rather than imported, matching `ingestion_worker.py`'s own
# documented precedent of not cross-importing this constant between
# independent worker modules.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# Code-review follow-up, applied here from the start (and retrofitted onto
# `ingestion_worker.py`/`eval_worker.py` in the same commit): every
# `WorkerSettings` in this codebase previously had no explicit `queue_name`,
# so all shared Arq's library-default queue on the same Redis instance -- a
# real risk (not theoretical: a worker that dequeues a job for a function
# name outside its own registry fails it immediately, confirmed by reading
# `arq.worker.Worker.run_job`'s real `KeyError` -> `job_failed(...)` path)
# that a stray job from one worker's queue gets picked up and failed by a
# DIFFERENT worker's burst-mode test run. Each worker module now owns its
# own queue name; see `ingestion_worker.QUEUE_NAME`'s comment for the full
# writeup.
QUEUE_NAME = "resolvegrid:cache-maintenance"


async def purge_expired_semantic_cache_task(ctx: dict) -> dict:
    """The real Arq job function -- registered in both `WorkerSettings.
    functions` (so it's directly enqueue-able on demand, like any other real
    Arq job) and `WorkerSettings.cron_jobs` (so it ALSO fires automatically
    on the real hourly schedule above). Opens its own DB session (Arq jobs
    run in a separate worker process with no access to FastAPI's
    request-scoped `get_db`), mirroring `ingestion_worker.ingest_seed_corpus_
    task`'s identical shape exactly.

    Returns the real number of rows deleted this pass (`rows_deleted`), so a
    real worker-process log line / a real burst-mode test (see
    `test_cache_maintenance_worker.py`) can confirm real purge activity
    happened, not a silent no-op.
    """
    engine = create_engine(DATABASE_URL)
    try:
        with Session(engine) as session:
            rows_deleted = purge_expired_semantic_cache_entries(session)
            return {"rows_deleted": rows_deleted}
    finally:
        engine.dispose()


class WorkerSettings:
    """Arq worker process entry point:
    `uv run --package resolvegrid-api arq
    resolvegrid_api.cache_maintenance_worker.WorkerSettings`
    starts a real worker process that purges expired semantic-cache rows
    once at startup and once every real hour thereafter, consuming/enqueuing
    jobs via the same shared `resolvegrid-redis` container every other
    real job/cache mechanism in this codebase already uses.
    """

    functions = [purge_expired_semantic_cache_task]
    cron_jobs = [
        cron(
            purge_expired_semantic_cache_task,
            minute=0,
            run_at_startup=True,
        )
    ]
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
