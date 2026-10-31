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
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text

from resolvegrid_api.db import DATABASE_URL

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


def _percentile_sql(column: str, pct: float) -> str:
    return f"percentile_cont({pct}) within group (order by {column})"


def query_metrics(engine, started_at: datetime, completed_at: datetime) -> dict:
    window_start = started_at - _WINDOW_BUFFER
    window_end = completed_at + _WINDOW_BUFFER

    with engine.connect() as conn:
        agent_run_rows = conn.execute(
            text(
                """
                select status, cache_hit, cache_hit_kind,
                       extract(epoch from (completed_at - created_at)) * 1000 as latency_ms
                from agent_run
                where created_at between :start and :end
                """
            ),
            {"start": window_start, "end": window_end},
        ).mappings().all()

        latencies = [r["latency_ms"] for r in agent_run_rows if r["latency_ms"] is not None]
        latency_percentiles = {}
        if latencies:
            result = conn.execute(
                text(
                    f"""
                    select
                        {_percentile_sql('latency_ms', 0.50)} as p50,
                        {_percentile_sql('latency_ms', 0.95)} as p95,
                        {_percentile_sql('latency_ms', 0.99)} as p99,
                        avg(latency_ms) as mean
                    from (
                        select extract(epoch from (completed_at - created_at)) * 1000 as latency_ms
                        from agent_run
                        where created_at between :start and :end and completed_at is not null
                    ) t
                    """
                ),
                {"start": window_start, "end": window_end},
            ).mappings().one()
            latency_percentiles = dict(result)

        status_counts: dict[str, int] = {}
        cache_hit_kind_counts: dict[str, int] = {}
        cache_hit_count = 0
        for r in agent_run_rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            if r["cache_hit"]:
                cache_hit_count += 1
                kind = r["cache_hit_kind"] or "unknown"
                cache_hit_kind_counts[kind] = cache_hit_kind_counts.get(kind, 0) + 1

        # Phase 14 Task 2 code-review follow-up: the comparison report's
        # chat-only vs. diagnose-only server-side latency split was
        # originally reconstructed ad hoc (by `input_text` pattern,
        # outside this script) -- not reproducible by re-running this
        # script alone. Folded in here using the one real, structural
        # signal that distinguishes the two without any string matching:
        # `/tickets/{id}/diagnose` (routers/tickets.py) writes exactly one
        # `Span(stage_name="diagnose_ticket")` row for every AgentRun it
        # creates (both the success path and all three error paths); the
        # `/chat` graph (routers/chat.py's `_STAGE_NAMES`) never writes
        # that stage name. An AgentRun is therefore classified "diagnose"
        # iff a `Span` row with that `stage_name` exists for it, "chat"
        # otherwise -- exact, not a heuristic, and correct even if a
        # future workload mix adds a third endpoint that writes AgentRun
        # rows under some other stage-name shape.
        classified_rows = conn.execute(
            text(
                """
                with runs as (
                    select a.id,
                           extract(epoch from (a.completed_at - a.created_at)) * 1000 as latency_ms,
                           exists (
                               select 1 from span s
                               where s.agent_run_id = a.id and s.stage_name = 'diagnose_ticket'
                           ) as is_diagnose
                    from agent_run a
                    where a.created_at between :start and :end and a.completed_at is not null
                )
                select
                    is_diagnose,
                    count(*) as n,
                    {p50} as p50_ms,
                    {p95} as p95_ms,
                    {p99} as p99_ms,
                    avg(latency_ms) as mean_ms
                from runs
                group by is_diagnose
                """.format(
                    p50=_percentile_sql("latency_ms", 0.50),
                    p95=_percentile_sql("latency_ms", 0.95),
                    p99=_percentile_sql("latency_ms", 0.99),
                )
            ),
            {"start": window_start, "end": window_end},
        ).mappings().all()
        agent_run_by_classification = {
            ("diagnose" if row["is_diagnose"] else "chat"): {
                "n": row["n"],
                "p50_ms": float(row["p50_ms"]) if row["p50_ms"] is not None else None,
                "p95_ms": float(row["p95_ms"]) if row["p95_ms"] is not None else None,
                "p99_ms": float(row["p99_ms"]) if row["p99_ms"] is not None else None,
                "mean_ms": float(row["mean_ms"]) if row["mean_ms"] is not None else None,
            }
            for row in classified_rows
        }

        span_rows = conn.execute(
            text(
                """
                select s.stage_name,
                       {p50} as p50_ms,
                       {p95} as p95_ms,
                       {p99} as p99_ms,
                       avg(s.latency_ms) as mean_ms,
                       count(*) as n
                from span s
                join agent_run a on a.id = s.agent_run_id
                where a.created_at between :start and :end
                group by s.stage_name
                """.format(
                    p50=_percentile_sql("s.latency_ms", 0.50),
                    p95=_percentile_sql("s.latency_ms", 0.95),
                    p99=_percentile_sql("s.latency_ms", 0.99),
                )
            ),
            {"start": window_start, "end": window_end},
        ).mappings().all()

        model_call_rows = conn.execute(
            text(
                """
                select purpose, provider, model,
                       count(*) as n,
                       sum(input_tokens) as input_tokens,
                       sum(output_tokens) as output_tokens,
                       sum(estimated_cost_usd) as cost_usd,
                       avg(latency_ms) as mean_latency_ms,
                       sum(case when status != 'success' then 1 else 0 end) as error_count
                from model_call
                where created_at between :start and :end
                group by purpose, provider, model
                order by n desc
                """
            ),
            {"start": window_start, "end": window_end},
        ).mappings().all()

        total_model_call_cost = conn.execute(
            text("select coalesce(sum(estimated_cost_usd), 0) from model_call where created_at between :start and :end"),
            {"start": window_start, "end": window_end},
        ).scalar_one()
        total_model_call_count = conn.execute(
            text("select count(*) from model_call where created_at between :start and :end"),
            {"start": window_start, "end": window_end},
        ).scalar_one()

    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "agent_run": {
            "total": len(agent_run_rows),
            "status_counts": status_counts,
            "cache_hit_count": cache_hit_count,
            "cache_hit_rate": cache_hit_count / len(agent_run_rows) if agent_run_rows else None,
            "cache_hit_kind_counts": cache_hit_kind_counts,
            "latency_ms_percentiles": {k: float(v) if v is not None else None for k, v in latency_percentiles.items()},
        },
        "agent_run_by_classification": agent_run_by_classification,
        "span_by_stage": {
            row["stage_name"]: {
                "p50_ms": float(row["p50_ms"]) if row["p50_ms"] is not None else None,
                "p95_ms": float(row["p95_ms"]) if row["p95_ms"] is not None else None,
                "p99_ms": float(row["p99_ms"]) if row["p99_ms"] is not None else None,
                "mean_ms": float(row["mean_ms"]) if row["mean_ms"] is not None else None,
                "n": row["n"],
            }
            for row in span_rows
        },
        "model_call_by_purpose": [dict(row) for row in model_call_rows],
        "model_call_totals": {
            "total_cost_usd": float(total_model_call_cost),
            "total_calls": total_model_call_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["full_system", "baseline"])
    args = parser.parse_args()

    raw_path = _RESULTS_DIR / f"{args.label}_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(raw["started_at"])
    completed_at = datetime.fromisoformat(raw["completed_at"])

    engine = create_engine(DATABASE_URL)
    metrics = query_metrics(engine, started_at.replace(tzinfo=None), completed_at.replace(tzinfo=None))
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
