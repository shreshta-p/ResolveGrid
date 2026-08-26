import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry import trace
from resolvegrid_agent_orchestration import build_graph
from resolvegrid_telemetry import init_tracing

from resolvegrid_api import llm_gateway
from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.routers import chat, directory, tickets

# `langgraph-checkpoint-postgres`'s AsyncPostgresSaver uses psycopg's async
# driver, which raises psycopg.InterfaceError unconditionally under Windows'
# default ProactorEventLoop ("Psycopg cannot use the 'ProactorEventLoop' to
# run in async mode") -- empirically verified while wiring the checkpointer
# into this lifespan. Must switch to the Selector event loop policy before
# any event loop is created, so this runs at import time.
#
# CAVEAT (documented, not silently papered over): this fixes `TestClient`
# (starlette/anyio's asyncio backend respects the current global policy via
# `asyncio.run()`/`Runner(loop_factory=None)`), which is what this task's own
# verification (the pytest regression suite) actually depends on. It does
# NOT fix a real `uvicorn` dev-server process on Windows started without
# `--loop none` (e.g. scripts/smoke_test.sh's and the Makefile's current
# `uvicorn resolvegrid_api.main:app --reload ...` invocation): uvicorn's own
# loop factory (`uvicorn.loops.asyncio.asyncio_loop_factory`) hard-codes
# `asyncio.ProactorEventLoop` on win32 and passes it explicitly to
# `asyncio.run(..., loop_factory=...)`, which bypasses the global policy set
# here entirely. In production this doesn't matter (the app runs in a Linux
# container, where this whole Proactor/Selector split doesn't exist), but a
# local Windows dev-server run needs `--loop none` added to pick up this
# policy. Out of scope to fix here since that's scripts/Makefile, not
# apps/api source -- flagged for a follow-up, not silently left unstated.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# AsyncPostgresSaver.from_conn_string() hands its connection string straight
# to psycopg's raw AsyncConnection.connect() (libpq), which does not
# understand SQLAlchemy's "+psycopg" dialect+driver suffix -- empirically
# verified: passing DATABASE_URL as-is raises
# `psycopg.ProgrammingError: missing "=" after "postgresql+psycopg://..."`.
# Strip the SQLAlchemy dialect suffix to get the plain libpq-style URI the
# checkpointer actually needs; both point at the same database.
_CHECKPOINTER_DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)

# TEMPORARY: dev-only CORS allowlist so the apps/web Next.js dev server (which
# picks whichever of localhost:3000/3001 is free) can call this API from the
# browser during local development. No CORS middleware existed before this;
# without it every browser-based request is blocked by preflight, regardless
# of dev-server port. Must be replaced with a real, environment-driven origin
# allowlist before any non-local deployment (see debug-principal.ts for the
# matching temporary-auth caveat).
_DEV_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    tracer = init_tracing("resolvegrid-api")
    with tracer.start_as_current_span("api.startup"):
        # Placeholder span proving the OTel pipe works end-to-end; real
        # startup instrumentation lands as later phases add real startup work.
        pass

    # AsyncPostgresSaver.from_conn_string() is itself an async context
    # manager (it owns the underlying AsyncConnection's lifetime) -- entered
    # here, same scope as init_tracing()/its shutdown below, so the
    # checkpointer's connection lives for the whole app lifetime and is
    # cleanly closed on shutdown, symmetric with tracer provider shutdown.
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINTER_DATABASE_URL) as checkpointer:
        # setup() creates the checkpointer's own tables if missing and is
        # documented + empirically confirmed idempotent (safe to call again
        # on an already-set-up database) -- required since TestClient(app)
        # re-runs this lifespan once per test file that imports `app` at
        # module level.
        await checkpointer.setup()
        complete_fn = lambda prompt: llm_gateway.complete(prompt).text  # noqa: E731
        app.state.agent_graph = build_graph(checkpointer, complete_fn)
        yield
    trace.get_tracer_provider().shutdown()


app = FastAPI(title="ResolveGrid API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(directory.router)
app.include_router(tickets.router)
app.include_router(chat.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
