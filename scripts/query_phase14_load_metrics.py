"""Phase 14 Task 2: real server-side metrics for a load-test run, queried
directly from Postgres `AgentRun`/`Span`/`ModelCall` rows -- per the phase
plan's explicit requirement that latency/cost/throughput numbers be
"computed from real AgentRun/Span/ModelCall rows, not the harness's own
client-side timers alone (both should be captured and cross-checked)."

Real schema limitation, found while writing this (reported here rather
than silently worked around): `ModelCall` has no foreign key to
`AgentRun` (confirmed via `\\d model_call`/`\\d agent_run` against the
real live DB) -- it only carries `trace_id` (an OTel trace id, unique
per incoming HTTP request) and `created_at`, and `Span` has no
`trace_id` column either, so there is no SQL join path from a
`ModelCall` row back to the specific `AgentRun` row it served. This
script therefore correlates `ModelCall` to a run by `created_at` TIME
WINDOW (the run's own `[started_at, completed_at]`, read from the raw
results JSON this run wrote) rather than a precise per-request FK join
-- accurate in aggregate for a run executed in isolation (no other
traffic hitting this same API process during the window), but NOT a
precise per-request cost attribution. Reported explicitly as a
limitation in the final comparison, not hidden.

Usage:
    uv run --package resolvegrid-api python scripts/query_phase14_load_metrics.py --label full_system
    uv run --package resolvegrid-api python scripts/query_phase14_load_metrics.py --label baseline

Phase 14 Task 3 note: the actual aggregation SQL this script runs now lives
in `resolvegrid_api.performance_metrics.query_metrics` -- extracted
verbatim (same queries, same correlation-by-time-window limitation
documented above) so the new admin performance dashboard endpoint
(`routers/admin_performance.py`) reuses this exact, already-correct logic
rather than re-deriving it. This script is unchanged in behavior; it now
just imports the shared function and still owns its own buffering decision
(`_WINDOW_BUFFER` below) before calling it, same as before.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.performance_metrics import query_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "eval" / "results" / "phase14_load_v1"

# Small buffer on both ends: a request that arrives right at the
# schedule's last arrival time can still be mid-flight (classify_intent
# -> retrieve -> compose_response -> finalize, or a multi-turn diagnose
# loop) when the client-side harness returns, so its AgentRun/Span/
# ModelCall rows can commit a few seconds AFTER the harness's own
# `completed_at`. Widening the window costs nothing here since each run
# is executed in isolation against an otherwise-idle API process.
_WINDOW_BUFFER = timedelta(seconds=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["full_system", "baseline"])
    args = parser.parse_args()

    raw_path = _RESULTS_DIR / f"{args.label}_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(raw["started_at"])
    completed_at = datetime.fromisoformat(raw["completed_at"])

    window_start = started_at.replace(tzinfo=None) - _WINDOW_BUFFER
    window_end = completed_at.replace(tzinfo=None) + _WINDOW_BUFFER

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        metrics = query_metrics(session, window_start, window_end)
    engine.dispose()

    client_summary = raw["client_summary"]
    total_requests = client_summary["total_requests"]
    successful_agent_runs = metrics["agent_run"]["status_counts"].get("completed", 0)

    cost_per_request = (
        metrics["model_call_totals"]["total_cost_usd"] / total_requests if total_requests else None
    )
    cost_per_successful_task = (
        metrics["model_call_totals"]["total_cost_usd"] / successful_agent_runs if successful_agent_runs else None
    )

    out = {
        "label": args.label,
        "run_id": raw["run_id"],
        "git_commit": raw["git_commit"],
        "client_summary": client_summary,
        "server_metrics": metrics,
        "cost_per_request_usd": cost_per_request,
        "cost_per_successful_task_usd": cost_per_successful_task,
        "redis_keyspace_hit_rate_delta": (
            client_summary["redis_keyspace_hits_delta"]
            / (client_summary["redis_keyspace_hits_delta"] + client_summary["redis_keyspace_misses_delta"])
            if (client_summary["redis_keyspace_hits_delta"] + client_summary["redis_keyspace_misses_delta"]) > 0
            else None
        ),
    }

    out_path = _RESULTS_DIR / f"{args.label}_server_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
