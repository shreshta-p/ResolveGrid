from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from resolvegrid_telemetry import init_tracing

from resolvegrid_api.routers import directory, tickets

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
