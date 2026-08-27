"""Local-dev-only launcher for `resolvegrid_api.main:app` on Windows.

Why this exists, not a plain `uvicorn resolvegrid_api.main:app ...` invocation:
`AsyncPostgresSaver` (wired into the app's lifespan since Phase 6) uses
psycopg's async driver, which raises `psycopg.InterfaceError` under Windows'
default `ProactorEventLoop`. `apps/api/src/resolvegrid_api/main.py` already
sets `asyncio.WindowsSelectorEventLoopPolicy()` at import time to fix this --
but that fix only helps callers who let the *policy* decide the loop class
(e.g. `TestClient`, which is why the pytest suite works fine). A real
`uvicorn` process does not: `uvicorn.loops.asyncio.asyncio_loop_factory()`
(and `uvicorn.loops.auto.auto_loop_factory()`, `--loop auto`'s default path
on Windows since `uvloop` doesn't support Windows at all) both hard-code
`return asyncio.ProactorEventLoop` on `sys.platform == "win32"` and hand that
class directly to `asyncio.run(loop_factory=...)`, bypassing any
process-wide policy entirely -- confirmed by reading uvicorn's own source
(`.venv/Lib/site-packages/uvicorn/loops/{asyncio,auto}.py`). There is no
`--loop` CLI value that avoids this; it isn't a flag problem.

This script monkey-patches those two factory functions, on Windows only,
before uvicorn's own startup machinery runs, so it returns a
`SelectorEventLoop` instead -- confirmed working empirically (a real
`AsyncPostgresSaver`-backed lifespan starts and serves `/health` correctly
under this launcher). Irrelevant in a real Linux container deployment,
where none of this Proactor/Selector split exists and the real
`uvicorn resolvegrid_api.main:app` entrypoint is used directly, unmodified.

Usage (equivalent to the plain uvicorn command it replaces):
    uv run --package resolvegrid-api python scripts/run_api_dev.py [--reload] [--port PORT]
"""

import argparse
import asyncio
import sys

if sys.platform == "win32":
    import uvicorn.loops.asyncio as _uv_asyncio_loops
    import uvicorn.loops.auto as _uv_auto_loops

    def _selector_loop_factory(use_subprocess: bool = False):
        return asyncio.SelectorEventLoop

    _uv_asyncio_loops.asyncio_loop_factory = _selector_loop_factory
    _uv_auto_loops.auto_loop_factory = _selector_loop_factory

import uvicorn  # noqa: E402 -- must import after the patch above is applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "resolvegrid_api.main:app",
        app_dir="apps/api/src",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
