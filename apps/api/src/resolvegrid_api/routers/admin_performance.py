"""Phase 14 Task 3: real admin performance dashboard API --
`GET /performance/metrics`.

Explicit exit-criterion this endpoint exists to satisfy (`PLAN_APPROVED.md`
line 280, restated in the phase plan doc's own Task 3 section): "Genuine
telemetry only -- no static mock charts." Every number this endpoint
returns comes from a real query against `AgentRun`/`Span`/`ModelCall` rows
owned by this schema, aggregated over a real, explicit time window chosen
by the caller -- never a fixed total row count (this shared dev Postgres
has `AgentRun`/`Span` rows pruned by other tests' fixtures over time, per
Task 2's independently-verified observation that ~20 chat `AgentRun` rows
disappeared between its report being generated and its re-verification
shortly after) and never a hardcoded/pre-baked response.

Reuses `resolvegrid_api.performance_metrics.query_metrics` -- the exact
same, already-correct SQL Task 2's `scripts/query_phase14_load_metrics.py`
report used and had independently verified -- rather than re-deriving the
aggregation. This module's own job is purely: (1) resolve the caller's
window-selection params into a concrete `[window_start, window_end]`, (2)
call that shared function, (3) layer on the few additional derived numbers
(throughput, error rate, cost/request, cost/successful-task) the plan doc's
Task 3 section asks for that aren't already part of that shared function's
return shape, and (4) report two REAL, HONEST GAPS explicitly rather than
faking them (see `_KNOWN_GAPS` below).

Window selection -- three mutually exclusive modes, so the admin can pick
Task 2's own load-test windows (LoadTestRun ids 16/17) as ONE way to select
a window, not the only way, per this task's explicit instruction:
  1. `load_test_run_id`: looks up a real `LoadTestRun` row's own
     `started_at`/`completed_at`, with the same `_LOAD_TEST_WINDOW_BUFFER`
     padding `scripts/query_phase14_load_metrics.py` already applies and
     documents (a request can still be mid-flight when the harness's own
     `completed_at` was recorded).
  2. `since`/`until`: an explicit real range, used exactly as given (no
     implicit padding -- an admin-chosen window means what it says).
  3. `window_minutes` (or neither given, defaulting to
     `_DEFAULT_WINDOW_MINUTES`): the trailing window ending now.
At most one of these three modes may be given per request; combining any
two (including `window_minutes` alongside `load_test_run_id` or
`since`/`until`) is a 400, since there would be no well-defined single
window to aggregate -- a caller-supplied `window_minutes` must never be
silently dropped just because another mode was also given.

Authz: `performance.view` (packages/authz/.../policy.py's
`_STAFF_ONLY_ACTIONS`) -- mirrors `routers/evals.py`'s `eval.view` pattern
exactly (same reasoning: this surfaces aggregate behavior across every
employee's activity, never a single employee's own resource, so a
principal with no admin/department-scoped analyst-or-approver grant is
denied outright, never downgraded to a self-scoped Decision).

`_KNOWN_GAPS` -- real, honestly-reported instrumentation gaps, per this
task's explicit instruction to report a genuinely-untracked metric as a
real gap rather than approximate or fake it:
  - TTFT (time-to-first-token): grep-verified (see this module's own
    development notes) that no code path in this repo records a
    first-token timestamp anywhere -- `/chat` and `/tickets/{id}/diagnose`
    both compose a full `output_text` before returning (no token-level
    streaming instrumentation exists today, confirmed by `main.py`/
    `routers/chat.py`/`routers/tickets.py` having no `stream`/`ttft`/
    `first_token` reference at all). Reported as `available: false` with
    an explicit note, never a fabricated number.
  - Cache hit rate "per tier": `AgentRun.cache_hit_kind` (models/agent.py)
    only ever takes the real values "exact" or "semantic" -- the retrieve
    node's own result cache (`resolvegrid_api.retrieval_cache`) never
    writes any cache-tier marker back to `Span.detail_json` or anywhere
    else queryable (grep-verified against `agent_retrieval.py`/
    `retrieval_cache.py`/`routers/chat.py`). So a genuine "retrieval" tier
    hit-rate number cannot be computed from owned schema today -- reported
    as `retrieval: null` with an explicit note alongside the real
    exact/semantic counts, not silently omitted or estimated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models.load_test import LoadTestRun
from resolvegrid_api.performance_metrics import query_metrics
from resolvegrid_authz import Principal, authorize

# No `/admin` path segment -- matching `routers/evals.py`'s precedent
# (prefix="/evals", tags=["evals"]) exactly: every staff-only router in
# this API enforces admin-ness purely through the authz layer
# (`authorize(principal, "...")` below), never through the URL path. This
# is the one router that previously deviated from that convention.
router = APIRouter(prefix="/performance", tags=["performance"])

# Trailing-window default when the caller gives no window-selection param at
# all -- a reasonable "what's happening right now" default, matching this
# task's instruction that the endpoint not assume a fixed total row count.
_DEFAULT_WINDOW_MINUTES = 60
_MAX_WINDOW_MINUTES = 7 * 24 * 60  # one week -- generous, still bounded

# Same padding `scripts/query_phase14_load_metrics.py` applies and
# documents around a load-test run's own started_at/completed_at: a
# request still mid-flight when the harness recorded its own completed_at
# can commit its AgentRun/Span/ModelCall rows a little later.
_LOAD_TEST_WINDOW_BUFFER = timedelta(seconds=60)

_TTFT_GAP_NOTE = (
    "Not tracked. No code path in this repo records a first-token "
    "timestamp -- /chat and /tickets/{id}/diagnose both compose a full "
    "response before returning (no token-level streaming instrumentation "
    "exists today). This is a real, honest gap, not an approximation."
)
_RETRIEVAL_CACHE_TIER_GAP_NOTE = (
    "Not tracked. AgentRun.cache_hit_kind only ever records 'exact' or "
    "'semantic' (the full-response caches) -- the retrieval-result cache "
    "(resolvegrid_api.retrieval_cache) never writes a cache-tier marker "
    "back to Span.detail_json or anywhere else queryable. A genuine "
    "retrieval-tier hit rate cannot be computed from owned schema today."
)
_COST_NOTE = (
    "This is a local-only dev environment. Most model calls route through "
    "LiteLLM -> Ollama (a real $0 PricingVersion row), so dollar cost is "
    "$0 for those -- not a placeholder. A real non-zero PricingVersion "
    "row also exists for the cloud-fallback path (routing.py routes "
    "risk_level=='high' calls there), but every attempt observed in this "
    "sandbox's load-test windows failed before completing (no outbound "
    "network here), so $0 cost in the windows below reflects THIS "
    "environment's real call outcomes, not an architectural guarantee "
    "that cloud calls are free. Token totals are the real signal; dollar "
    "figures are shown because the plan doc asks for them, not because "
    "they are reliably meaningful here."
)


def _resolve_window(
    load_test_run_id: int | None,
    since: datetime | None,
    until: datetime | None,
    window_minutes: int | None,
    session: Session,
) -> tuple[datetime, datetime, dict]:
    """Resolves the caller's window-selection params into a concrete,
    timezone-naive `[window_start, window_end]` plus a `source` dict
    describing how the window was chosen (surfaced in the response so the
    admin/UI can see exactly what was aggregated). Raises `HTTPException`
    on an invalid/ambiguous combination.

    At most one window-selection mode may be given (code-review follow-up:
    originally `window_minutes` combined with `load_test_run_id` or with
    `since`/`until` was silently ignored rather than rejected, unlike the
    `load_test_run_id`-vs-`since`/`until` combination, which WAS already
    rejected -- an inconsistency between two equally-ambiguous cases, fixed
    here so all three modes are checked the same way).
    """
    explicit_range_given = since is not None or until is not None
    modes_given = sum(
        (load_test_run_id is not None, explicit_range_given, window_minutes is not None)
    )
    if modes_given > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "give at most one window-selection mode: load_test_run_id, "
                "since/until, or window_minutes -- combining them leaves no "
                "well-defined single window to aggregate"
            ),
        )

    if load_test_run_id is not None:
        run = session.get(LoadTestRun, load_test_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="load test run not found")
        if run.completed_at is None:
            raise HTTPException(
                status_code=400,
                detail=f"load test run {load_test_run_id} has not completed yet (completed_at is null)",
            )
        window_start = run.started_at.replace(tzinfo=None) - _LOAD_TEST_WINDOW_BUFFER
        window_end = run.completed_at.replace(tzinfo=None) + _LOAD_TEST_WINDOW_BUFFER
        return (
            window_start,
            window_end,
            {
                "mode": "load_test_run",
                "load_test_run_id": load_test_run_id,
                "load_test_run_started_at": run.started_at.isoformat(),
                "load_test_run_completed_at": run.completed_at.isoformat(),
                "buffer_seconds_applied": _LOAD_TEST_WINDOW_BUFFER.total_seconds(),
            },
        )

    if explicit_range_given:
        if since is None or until is None:
            raise HTTPException(status_code=400, detail="give both since and until, or neither")
        if until <= since:
            raise HTTPException(status_code=400, detail="until must be after since")
        return (
            since.replace(tzinfo=None),
            until.replace(tzinfo=None),
            {"mode": "explicit_range", "buffer_seconds_applied": 0},
        )

    minutes = window_minutes if window_minutes is not None else _DEFAULT_WINDOW_MINUTES
    if not (0 < minutes <= _MAX_WINDOW_MINUTES):
        raise HTTPException(
            status_code=400,
            detail=f"window_minutes must be between 1 and {_MAX_WINDOW_MINUTES}",
        )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (
        now - timedelta(minutes=minutes),
        now,
        {
            "mode": "window_minutes_default" if window_minutes is None else "window_minutes",
            "window_minutes": minutes,
            "buffer_seconds_applied": 0,
        },
    )


@router.get("/metrics")
def get_performance_metrics(
    load_test_run_id: int | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    window_minutes: int | None = Query(default=None, gt=0, le=_MAX_WINDOW_MINUTES),
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """Real p50/p95/p99 latency (overall + chat-vs-diagnose split), cache
    hit rate, token/cost totals, throughput, and error rate -- aggregated
    from `AgentRun`/`Span`/`ModelCall` rows over a real, explicit window.
    See module docstring for window-selection modes and the two honestly-
    reported instrumentation gaps (TTFT, retrieval-tier cache hits).
    """
    decision = authorize(principal, "performance.view")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    window_start, window_end, window_source = _resolve_window(
        load_test_run_id, since, until, window_minutes, session
    )

    metrics = query_metrics(session, window_start, window_end)

    duration_seconds = (window_end - window_start).total_seconds()
    agent_run = metrics["agent_run"]
    total_agent_runs = agent_run["total"]
    completed_count = agent_run["status_counts"].get("completed", 0)
    error_count = agent_run["status_counts"].get("error", 0)

    total_cost_usd = metrics["model_call_totals"]["total_cost_usd"]
    total_input_tokens = sum(row["input_tokens"] or 0 for row in metrics["model_call_by_purpose"])
    total_output_tokens = sum(row["output_tokens"] or 0 for row in metrics["model_call_by_purpose"])

    return {
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "duration_seconds": duration_seconds,
            **window_source,
        },
        "throughput": {
            "completed_agent_runs_per_sec": (
                completed_count / duration_seconds if duration_seconds > 0 else None
            ),
            "completed_count": completed_count,
            "total_agent_run_count": total_agent_runs,
        },
        "latency_ms": {
            "overall": agent_run["latency_ms_percentiles"],
            "chat": metrics["agent_run_by_classification"].get("chat"),
            "diagnose": metrics["agent_run_by_classification"].get("diagnose"),
        },
        "error_rate": {
            # None (not 0.0) when the window has no AgentRun rows at all --
            # "no data" must never be reported as "zero errors."
            "agent_run_error_rate": (error_count / total_agent_runs) if total_agent_runs else None,
            "error_count": error_count,
            "total_count": total_agent_runs,
            "status_counts": agent_run["status_counts"],
        },
        "cache": {
            "hit_rate_overall": agent_run["cache_hit_rate"],
            "hit_count": agent_run["cache_hit_count"],
            "total_count": total_agent_runs,
            "hit_rate_by_tier": {
                "exact": agent_run["cache_hit_kind_counts"].get("exact"),
                "semantic": agent_run["cache_hit_kind_counts"].get("semantic"),
                "retrieval": None,
            },
            "known_gap_retrieval_tier": _RETRIEVAL_CACHE_TIER_GAP_NOTE,
        },
        "cost": {
            "total_model_call_cost_usd": total_cost_usd,
            # "request" here means "AgentRun" (the only real per-request
            # unit this endpoint's window can see from DB rows alone) --
            # Task 2's own script additionally had the harness's
            # client-side total_requests count available from its raw
            # results file, which this general admin-window endpoint does
            # not; stated explicitly rather than silently reusing the same
            # label for a different real denominator.
            "cost_per_request_usd": (
                total_cost_usd / total_agent_runs if total_agent_runs else None
            ),
            "cost_per_successful_task_usd": (
                total_cost_usd / completed_count if completed_count else None
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "note": _COST_NOTE,
        },
        "ttft": {
            "available": False,
            "note": _TTFT_GAP_NOTE,
        },
        "model_call_by_purpose": metrics["model_call_by_purpose"],
        "span_by_stage": metrics["span_by_stage"],
    }
