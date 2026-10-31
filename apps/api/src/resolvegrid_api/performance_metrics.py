"""Shared real `AgentRun`/`Span`/`ModelCall` aggregation core.

Extracted from `scripts/query_phase14_load_metrics.py` (Phase 14 Task 2) so
Task 3's admin performance dashboard endpoint
(`resolvegrid_api.routers.admin_performance`) reuses the exact same,
already-correct, independently-verified SQL rather than re-deriving the
aggregation from scratch -- per this phase's explicit instruction. Task 2's
script now imports `query_metrics` from here instead of defining it inline;
its own module docstring (the `ModelCall`-has-no-`AgentRun`-FK limitation,
the correlate-by-time-window caveat) still applies unchanged to every call
of this function, from either caller.

`query_metrics` itself is a pure window aggregation: given a session and an
exact `[window_start, window_end]`, it queries real rows and returns
real numbers -- it adds no implicit buffering of its own. Any padding
around a reference window (e.g. Task 2's `_WINDOW_BUFFER` to tolerate a
request still in flight when the harness's own `completed_at` was recorded)
is each caller's own, explicit decision, made before calling this function
-- not hidden inside it, since a general admin-chosen `since`/`until`
window (Task 3's endpoint) must mean exactly what the admin typed, not get
silently widened the way a load-test run's own crash-tolerant window may
want to be.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session


def _percentile_sql(column: str, pct: float) -> str:
    return f"percentile_cont({pct}) within group (order by {column})"


def query_metrics(session: Session, window_start: datetime, window_end: datetime) -> dict:
    """Real aggregation of `AgentRun`/`Span`/`ModelCall` rows whose
    `created_at` falls in `[window_start, window_end]`. See module docstring
    for the no-implicit-buffering contract and `scripts/
    query_phase14_load_metrics.py`'s module docstring for the real
    `ModelCall`-to-`AgentRun` correlation limitation this inherits
    unchanged (time-window correlation, not a precise per-request FK join).
    """
    agent_run_rows = session.execute(
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
        result = session.execute(
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

    # An AgentRun is classified "diagnose" iff a real `Span(stage_name=
    # "diagnose_ticket")` row exists for it (written by every `/tickets/
    # {id}/diagnose` code path, success and all three error paths; the
    # `/chat` graph never writes that stage name) -- exact, not a string
    # heuristic. See `scripts/query_phase14_load_metrics.py`'s module
    # docstring (code-review follow-up) for the full reasoning.
    classified_rows = session.execute(
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

    span_rows = session.execute(
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

    model_call_rows = session.execute(
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

    total_model_call_cost = session.execute(
        text("select coalesce(sum(estimated_cost_usd), 0) from model_call where created_at between :start and :end"),
        {"start": window_start, "end": window_end},
    ).scalar_one()
    total_model_call_count = session.execute(
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
