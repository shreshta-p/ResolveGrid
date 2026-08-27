"""Arq-backed ingestion worker (Phase 7 Task 6) -- the first real async job
in this repo. Redis was already running via `infra/docker-compose.yml`'s
`resolvegrid-redis` service (host port 6380, per that compose file), but
nothing in this repo consumed it before this task; confirmed `arq` (and
any equivalent) was not already a dependency anywhere in the uv workspace
before adding it to `apps/api/pyproject.toml`.

Two entry points into the same pipeline logic, so the `IngestionRun`
bookkeeping (status transitions, per-run stats, error handling) is
implemented in exactly one place:

1. `run_seed_corpus_ingestion(session)` -- a plain, synchronous, directly
   -callable function. This is what `apps/api/tests/test_ingestion_worker.py`
   calls to verify the pipeline end-to-end without needing a live
   Arq-worker/Redis round trip, and what this module's `main()`
   (`uv run --package resolvegrid-api python -m
   resolvegrid_api.ingestion_worker`) calls for a real, manually
   -triggerable run against the live DB.
2. `ingest_seed_corpus_task(ctx)` -- the real Arq job function (the async
   calling convention Arq requires: registered in `WorkerSettings.functions`
   below). A real worker process is started with
   `uv run --package resolvegrid-api arq resolvegrid_api.ingestion_worker.WorkerSettings`,
   which connects to Redis via `REDIS_URL` and waits for enqueued jobs.
   Arq jobs run in a separate worker process with no access to FastAPI's
   request-scoped `get_db` dependency, so this function opens its own
   SQLAlchemy session against `DATABASE_URL` (same pattern as `seed.py`'s
   `main()`). The actual blocking DB/HTTP work
   (`run_seed_corpus_ingestion`, which calls the real chunker, the real
   embedder over HTTP to Ollama, and synchronous SQLAlchemy) is offloaded
   via `asyncio.to_thread` so it doesn't block Arq's event loop from
   servicing other concurrent jobs, even though this phase only ever
   enqueues the one seed-corpus job.

`apps/api/tests/test_ingestion_worker.py`'s
`test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis` proves
`ingest_seed_corpus_task` is wired as a genuine wire-format Arq job, not
just a same-process function call: it enqueues a real job onto the live
`resolvegrid-redis` queue via `arq.create_pool`, then runs a real
`arq.worker.Worker` in burst mode (`run_worker(..., burst=True)`, which
processes every queued job then exits rather than polling forever) to
execute it.
"""

import asyncio
import os
from datetime import datetime, timezone

from arq.connections import RedisSettings
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.ingestion import ingest_document
from resolvegrid_api.models.knowledge import Chunk, IngestionRun
from resolvegrid_api.seed_corpus import load_seed_corpus
from resolvegrid_retrieval.embedder import DEFAULT_EMBEDDING_MODEL

# Redis connection for Arq, following DATABASE_URL's env-override
# convention (apps/api/src/resolvegrid_api/db.py) -- defaults to this
# repo's docker-compose host port mapping for `resolvegrid-redis`
# (infra/docker-compose.yml: "6380:6379").
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# Pinned versions for this phase's ingestion runs. Bumping any of these
# is a deliberate re-ingestion-triggering event (a new parser_version or
# chunking_version changes chunk_markdown's output; a new embedding_model
# or embedding_version changes embed_texts's output) -- kept as named
# constants here, not scattered string literals, so a version bump is a
# one-line change.
PARSER_VERSION = "markdown-v1"
CHUNKING_VERSION = "heading-aware-v1"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL  # "nomic-embed-text"
EMBEDDING_VERSION = "v1"


def _chunk_count(session: Session, document_version_id: int) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_version_id == document_version_id)
        )
        or 0
    )


def run_seed_corpus_ingestion(session: Session) -> IngestionRun:
    """Ingest every `resolvegrid_api.seed_corpus.SEED_CORPUS` document
    through `resolvegrid_api.ingestion.ingest_document`, recording one
    `IngestionRun` row for the whole batch (started `status="running"`,
    finished `status="completed"` with `documents_processed`/
    `chunks_created` populated, or `status="error"` with
    `error_message` set if any document fails partway through -- in the
    error case this function re-raises after recording the row, matching
    `IngestionRun.status`'s "running" | "completed" | "error" contract).

    `chunks_created` counts the chunks attached to each processed
    document's *returned* `DocumentVersion` (via a fresh `Chunk` count
    query per document), whether that version was newly created this run
    or reused via `ingest_document`'s idempotent-no-op path -- i.e. it is
    "chunks associated with this run's documents," not strictly "chunks
    newly inserted this run." Documented here since `IngestionRun`'s own
    docstring only says "aggregate counts" without pinning down the
    idempotent-rerun case.

    Does not commit -- callers control the transaction boundary, matching
    `ingest_document`'s own convention.
    """
    run = IngestionRun(
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_version=EMBEDDING_VERSION,
        status="running",
    )
    session.add(run)
    session.flush()

    # SeedDocument.supersedes_title names another entry's title;
    # resolving it to a real Document.id requires that document to
    # already be ingested (see seed_corpus.py's SeedDocument docstring),
    # so this tracks each processed title's resulting Document.id as we
    # go, and SEED_CORPUS is ordered so every supersedes_title reference
    # appears earlier in the list than the document that uses it.
    title_to_document_id: dict[str, int] = {}
    documents_processed = 0
    chunks_created = 0

    try:
        for seed_doc in load_seed_corpus():
            supersedes_document_id = (
                title_to_document_id[seed_doc.supersedes_title]
                if seed_doc.supersedes_title
                else None
            )
            version = ingest_document(
                session,
                title=seed_doc.title,
                source_type=seed_doc.source_type,
                raw_markdown=seed_doc.raw_markdown,
                access_scope_tags=seed_doc.access_scope_tags,
                url=seed_doc.url,
                publisher=seed_doc.publisher,
                retrieved_at=seed_doc.retrieved_at,
                effective_date=seed_doc.effective_date,
                license=seed_doc.license,
                status=seed_doc.status,
                supersedes_document_id=supersedes_document_id,
                parser_version=PARSER_VERSION,
                chunking_version=CHUNKING_VERSION,
                embedding_model=EMBEDDING_MODEL,
                embedding_version=EMBEDDING_VERSION,
            )
            title_to_document_id[seed_doc.title] = version.document_id
            documents_processed += 1
            chunks_created += _chunk_count(session, version.id)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
        # failure partway through the batch must still record the run as
        # errored (with the partial counts so far) before propagating,
        # rather than leaving the row stuck at status="running" forever.
        run.status = "error"
        run.error_message = str(exc)
        run.documents_processed = documents_processed
        run.chunks_created = chunks_created
        run.completed_at = datetime.now(timezone.utc)
        session.flush()
        raise

    run.status = "completed"
    run.documents_processed = documents_processed
    run.chunks_created = chunks_created
    run.completed_at = datetime.now(timezone.utc)
    session.flush()
    return run


async def ingest_seed_corpus_task(ctx: dict) -> dict:
    """The real Arq job function -- registered in `WorkerSettings.functions`
    below. Opens its own DB session (see module docstring) and delegates
    to `run_seed_corpus_ingestion` so the actual ingestion/bookkeeping
    logic exists in exactly one place.
    """
    engine = create_engine(DATABASE_URL)
    try:
        with Session(engine) as session:
            run = await asyncio.to_thread(run_seed_corpus_ingestion, session)
            session.commit()
            return {
                "ingestion_run_id": run.id,
                "status": run.status,
                "documents_processed": run.documents_processed,
                "chunks_created": run.chunks_created,
            }
    finally:
        engine.dispose()


class WorkerSettings:
    """Arq worker process entry point:
    `uv run --package resolvegrid-api arq resolvegrid_api.ingestion_worker.WorkerSettings`
    starts a real worker process consuming jobs from `resolvegrid-redis`.
    """

    functions = [ingest_seed_corpus_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)


def main() -> None:
    """Direct, non-Arq entry point for manually verifying the pipeline
    against the live DB: `uv run --package resolvegrid-api python -m
    resolvegrid_api.ingestion_worker`.
    """
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        run = run_seed_corpus_ingestion(session)
        session.commit()
        print(
            f"IngestionRun {run.id}: status={run.status} "
            f"documents_processed={run.documents_processed} "
            f"chunks_created={run.chunks_created}"
        )


if __name__ == "__main__":
    main()
