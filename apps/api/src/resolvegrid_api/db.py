import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://resolvegrid:resolvegrid_dev@localhost:5433/resolvegrid",
)

_engine = create_engine(DATABASE_URL)


def get_db() -> Iterator[Session]:
    with Session(_engine) as session:
        yield session


def session_factory() -> Session:
    """A fresh `Session` bound to this module's shared engine, for callers
    that need their own session lifecycle instead of the request-scoped
    generator `get_db()` provides above.

    Used by `resolvegrid_api.agent_retrieval`'s `retrieve_fn` closure
    (Phase 7 Task 7): that closure is built once at app startup, outside
    FastAPI's per-request `Depends(get_db)` scope, so it cannot reuse a
    request's session -- but unlike `ingestion_worker.py`'s Arq jobs
    (which create/dispose a brand new `Engine` per call, since ingestion
    is rare), `/chat` is a hot path, so this reuses the shared, already
    -pooled `_engine` rather than paying connection-pool setup cost on
    every request.
    """
    return Session(_engine)
