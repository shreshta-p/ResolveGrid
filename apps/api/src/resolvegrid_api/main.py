from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from resolvegrid_telemetry import init_tracing

tracer = init_tracing("resolvegrid-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with tracer.start_as_current_span("api.startup"):
        pass
    yield


app = FastAPI(title="ResolveGrid API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
