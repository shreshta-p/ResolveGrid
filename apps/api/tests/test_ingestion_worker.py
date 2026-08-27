"""Tests for `resolvegrid_api.ingestion_worker` (Phase 7 Task 6).

`test_run_seed_corpus_ingestion_creates_completed_ingestion_run_with_correct_counts`
runs the full seed corpus through the real pipeline (real chunker, real
Ollama embedder, real DB writes) and checks the resulting `IngestionRun`
row's stats against an independently-computed chunk count -- not just
pass/fail.

`test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis` proves
`ingest_seed_corpus_task` is wired as a genuine Arq job, not just a
same-process function: it enqueues a real job onto the live
`resolvegrid-redis` queue (via `arq.create_pool`) and runs a real
`arq.worker.Worker` in burst mode to execute it, then checks the
`IngestionRun` row the job itself committed. `handle_signals=False` is
required on Windows -- `Worker.run()` otherwise calls
`loop.add_signal_handler`, which raises `NotImplementedError` on
win32's default event loop.
"""

import asyncio
import signal

from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import run_worker
from sqlalchemy import func, select

# arq.worker.Worker.close() unconditionally references signal.SIGUSR1, even
# when handle_signals=False (see site-packages/arq/worker.py's `close()`) --
# it's used only to log which "signal" triggered shutdown (handle_sig()
# just logs and cancels pending tasks; it never actually registers an OS
# signal handler in this path). `signal.SIGUSR1` doesn't exist in Windows'
# stdlib `signal` module (it's POSIX-only), so `run_worker(...,
# handle_signals=False)` raises AttributeError during its own cleanup on a
# Windows dev machine, even though the job itself already completed
# successfully by that point. This shim supplies a harmless placeholder
# value for that logging call, scoped to this test process. The real
# deployment target for this worker is a Linux container (this repo's
# compose/production pattern), where SIGUSR1 exists natively and this
# workaround is a no-op.
if not hasattr(signal, "SIGUSR1"):
    signal.SIGUSR1 = signal.SIGTERM

from resolvegrid_api.ingestion_worker import (
    CHUNKING_VERSION,
    EMBEDDING_VERSION,
    PARSER_VERSION,
    REDIS_URL,
    WorkerSettings,
    run_seed_corpus_ingestion,
)
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, IngestionRun
from resolvegrid_api.seed_corpus import load_seed_corpus


def test_run_seed_corpus_ingestion_creates_completed_ingestion_run_with_correct_counts(db_session):
    corpus = load_seed_corpus()

    run = run_seed_corpus_ingestion(db_session)
    db_session.flush()

    assert run.id is not None
    assert run.status == "completed"
    assert run.error_message is None
    assert run.completed_at is not None
    assert run.parser_version == PARSER_VERSION
    assert run.chunking_version == CHUNKING_VERSION
    assert run.embedding_version == EMBEDDING_VERSION
    assert run.documents_processed == len(corpus)

    # Independently computed expected chunk count: sum of actual Chunk
    # rows for every corpus document's title, via a fresh query -- not
    # trusting the same code path that produced run.chunks_created.
    titles = [doc.title for doc in corpus]
    expected_chunks = db_session.scalar(
        select(func.count())
        .select_from(Chunk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.title.in_(titles))
    )

    assert run.chunks_created > 0
    assert run.chunks_created == expected_chunks

    persisted_run = db_session.get(IngestionRun, run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"
    assert persisted_run.documents_processed == len(corpus)
    assert persisted_run.chunks_created == expected_chunks

    # Every corpus document actually landed in the DB under its expected
    # source_type/tags -- spot-check the stale/superseded pair, since
    # that's the one relationship worth confirming end to end here.
    v1 = db_session.scalars(
        select(Document).where(Document.title == "Kestrel VPN Access Policy (v1, deprecated)")
    ).first()
    v2 = db_session.scalars(select(Document).where(Document.title == "Kestrel VPN Access Policy (v2)")).first()
    assert v1 is not None and v1.status == "superseded"
    assert v2 is not None and v2.status == "active"
    assert v2.supersedes_document_id == v1.id


def test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis(raw_db_session):
    """`ingest_seed_corpus_task` opens and commits its own DB session (see
    ingestion_worker.py's module docstring) -- independent of this test's
    `raw_db_session`, so its writes are real commits, not something a
    transactional fixture can roll back. To keep the shared dev DB from
    accumulating permanent seed-corpus rows every time this test runs
    (which would make Task 4/5's fixed-content search-ranking tests in
    test_retrieval.py increasingly fragile -- their queries and result
    -set assertions assume no unrelated matching content exists), this
    test deletes exactly the rows it caused to exist afterwards: the
    corpus Documents (by title) and everything that cascades from them,
    plus the IngestionRun row(s) created during this test.
    """
    from sqlalchemy import delete

    from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding

    async def _enqueue() -> None:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            job = await pool.enqueue_job("ingest_seed_corpus_task")
            assert job is not None, "enqueue_job returned None -- job may already be queued/deduped"
        finally:
            await pool.aclose()

    before_max_id = raw_db_session.scalar(select(func.max(IngestionRun.id))) or 0

    asyncio.run(_enqueue())

    worker = run_worker(WorkerSettings, burst=True, handle_signals=False)

    try:
        assert worker.jobs_complete >= 1
        assert worker.jobs_failed == 0

        latest_run = raw_db_session.scalars(
            select(IngestionRun).where(IngestionRun.id > before_max_id).order_by(IngestionRun.id.desc())
        ).first()

        assert latest_run is not None
        assert latest_run.status == "completed"
        assert latest_run.documents_processed == len(load_seed_corpus())
        assert latest_run.chunks_created > 0
    finally:
        titles = [doc.title for doc in load_seed_corpus()]
        document_ids = raw_db_session.scalars(select(Document.id).where(Document.title.in_(titles))).all()
        version_ids = raw_db_session.scalars(
            select(DocumentVersion.id).where(DocumentVersion.document_id.in_(document_ids))
        ).all()
        chunk_ids = raw_db_session.scalars(
            select(Chunk.id).where(Chunk.document_version_id.in_(version_ids))
        ).all()

        raw_db_session.execute(delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids)))
        raw_db_session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))
        raw_db_session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_(version_ids)))
        raw_db_session.execute(delete(Document).where(Document.id.in_(document_ids)))
        raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id > before_max_id))
        raw_db_session.commit()
