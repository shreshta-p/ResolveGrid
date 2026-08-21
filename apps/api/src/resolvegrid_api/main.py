from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from opentelemetry import trace
from resolvegrid_telemetry import init_tracing


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
