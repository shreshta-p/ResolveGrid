"""Tests for `resolvegrid_api.cache_maintenance_worker` (Phase 12 Task 6:
real scheduling of `semantic_cache.purge_expired_semantic_cache_entries`,
which existed and was tested as of Task 5 but was never actually called from
anywhere real -- its scheduling was explicitly deferred to this task, per
that function's own docstring).

Mirrors `test_ingestion_worker.py`'s
`test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis` almost
exactly (read that test first if this one is unclear) -- same real-Redis
-queue, real-burst-mode-`arq.worker.Worker`, same Windows `SIGUSR1` shim --
proving `purge_expired_semantic_cache_task` is wired as a genuine Arq job
AND that `WorkerSettings.cron_jobs` genuinely fires it without an explicit
`enqueue_job` call, via `run_at_startup=True` (burst mode runs exactly one
real heartbeat/cron-check cycle, which a `run_at_startup` cron job's
`next_run` always matches -- see `arq.worker.Worker.run_cron`): this is the
one behavioral difference from the ingestion-worker precedent worth calling
out, since this task is specifically about the *scheduling* mechanism, not
just "is this function callable from an Arq job" (Task 5 already proved the
function itself works; this only needs to prove the real automatic trigger).
"""

import asyncio
import signal

from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import run_worker
from sqlalchemy import text

# Same Windows-only shim as test_ingestion_worker.py -- see that module's
# comment for the full explanation (Worker.close() unconditionally
# references signal.SIGUSR1 even with handle_signals=False, which doesn't
# exist in Windows' stdlib signal module).
if not hasattr(signal, "SIGUSR1"):
    signal.SIGUSR1 = signal.SIGTERM

from resolvegrid_api.cache_maintenance_worker import REDIS_URL, WorkerSettings
from resolvegrid_api.models.knowledge import SemanticResponseCache
from resolvegrid_api.semantic_cache import SEMANTIC_CACHE_TTL_SECONDS

_TEST_SCOPE_PREFIX = "test-cache-maintenance-worker:"


def _clear_test_rows(session) -> None:
    from sqlalchemy import delete

    session.execute(delete(SemanticResponseCache).where(SemanticResponseCache.scope_sig.like(f"{_TEST_SCOPE_PREFIX}%")))
    session.commit()


def test_arq_cron_job_purges_a_real_expired_semantic_cache_row_via_real_redis(raw_db_session):
    _clear_test_rows(raw_db_session)

    expired_scope = f"{_TEST_SCOPE_PREFIX}expired"
    fresh_scope = f"{_TEST_SCOPE_PREFIX}fresh"

    expired_entry = SemanticResponseCache(
        scope_sig=expired_scope,
        kb_version=1,
        query_embedding=[0.01] * 768,
        answer="should be purged",
        citations_json="[]",
        caption="c",
        intent="general_question",
        risk_level="low",
    )
    fresh_entry = SemanticResponseCache(
        scope_sig=fresh_scope,
        kb_version=1,
        query_embedding=[0.02] * 768,
        answer="should survive",
        citations_json="[]",
        caption="c",
        intent="general_question",
        risk_level="low",
    )
    raw_db_session.add_all([expired_entry, fresh_entry])
    raw_db_session.commit()
    expired_id, fresh_id = expired_entry.id, fresh_entry.id

    # Backdate only the expired row's created_at past the real TTL cutoff,
    # same real-SQL technique test_semantic_cache.py's own purge test uses.
    raw_db_session.execute(
        text(
            "UPDATE semantic_response_cache SET created_at = now() - make_interval(secs => :past) "
            "WHERE id = :id"
        ),
        {"past": SEMANTIC_CACHE_TTL_SECONDS + 60, "id": expired_id},
    )
    raw_db_session.commit()

    try:
        async def _noop_enqueue_check() -> None:
            # No explicit enqueue_job call here, deliberately -- unlike
            # test_ingestion_worker.py's on-demand job, this test is
            # specifically proving the CRON mechanism fires the job on its
            # own via run_at_startup=True, not that the function is
            # separately triggerable (that's just arq's standard registered
            # -function behavior, already proven by the ingestion-worker
            # precedent). Still open/close a real pool so a connectivity
            # failure here surfaces clearly rather than masked by the
            # worker's own connection below.
            pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
            await pool.aclose()

        asyncio.run(_noop_enqueue_check())

        worker = run_worker(WorkerSettings, burst=True, handle_signals=False)

        assert worker.jobs_complete >= 1, "the run_at_startup cron job must have fired in this one burst cycle"
        assert worker.jobs_failed == 0

        remaining_ids = set(
            raw_db_session.scalars(
                text("SELECT id FROM semantic_response_cache WHERE id IN (:a, :b)").bindparams(
                    a=expired_id, b=fresh_id
                )
            ).all()
        )
        assert expired_id not in remaining_ids, "the real cron-triggered purge must have deleted the expired row"
        assert fresh_id in remaining_ids, "the fresh row must survive the same real purge pass"
    finally:
        _clear_test_rows(raw_db_session)
