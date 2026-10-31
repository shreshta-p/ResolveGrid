"""Phase 14 Task 2: real load-test run against the live Docker Compose
stack, using Task 1's `resolvegrid_api.load_test` scaffold (NHPP arrival
sampler, real-seeded-data workload-mix generator, `BoundedConcurrency
LoadClient`) -- this script is the thing that actually points that
scaffold at a real running API process and fires real HTTP requests.

Lives in `scripts/`, not `resolvegrid_api/`, mirroring `run_api_dev.py`'s
precedent of a thin orchestration entry point outside the package that
still imports the package's real machinery -- this script's own job
(argparse, process-level run orchestration, writing result files under
`eval/results/`) is a different concern from `load_test.py`'s own
library code.

Compression factor -- stated and justified, per the phase plan's own
requirement (`docs/superpowers/plans/2026-10-29-phase14-performance-
load-testing.md`, Task 2):

`NHPPConfig.duration_seconds` and `Arrival.t` are both in the SAME
literal-seconds units `intensity()` uses to compute `hour_of_day = t /
3600 % 24` -- so sampling arrivals with `duration_seconds=86400` (one
real simulated day, so the configured `peak_hours`/day-shaped curve
actually completes a full cycle) produces a list of `Arrival`s whose
`.t` values span the full 24h. Firing THAT schedule's `.t` values
directly at `asyncio.sleep` (as `BoundedConcurrencyLoadClient.
run_schedule` does) would take 24 real wall-clock hours -- impractical.
`COMPRESSION_FACTOR` divides every sampled arrival's `.t` by itself
AFTER sampling (`compress_arrivals` below), mapping the 86400-second
simulated day onto an `86400 / COMPRESSION_FACTOR`-second real
wall-clock window while preserving the exact RELATIVE shape of the
arrival sequence (same relative burst timing, same peak/trough
proportions, same total arrival COUNT) -- exactly "simulating a full
day's arrival pattern compressed into a shorter window preserves
relative burst/arrival-rate shape while being practical to run," the
exact example the phase plan itself gives.

`COMPRESSION_FACTOR = 48` (24h -> 30 minutes nominal) was chosen after
empirically probing this real stack's actual per-request latency
(`curl` timing against the real running API, see the phase-14 Task 2
report): a real local Ollama-backed `qwen3:14b` chat/diagnose call
takes roughly 7-25 real seconds once the model is warm (and 45-65s on a
cold Ollama load after 5+ minutes idle -- Ollama's default
`keep_alive` unload). Given `NHPPConfig(base_rate_rps=0.005,
peak_rate_rps=0.03, ...)` below samples ~1,150 total arrivals per real
day (mean rate ~1,109 req/day, comfortably over the phase's "1,000+
requests/day" target), the REAL bottleneck on how fast this run
actually completes is backend LLM service capacity, not the nominal
30-minute schedule -- so the schedule's job is only to preserve a
realistic RELATIVE arrival shape (bursts, two business-hours peaks,
overnight trough) for the harness to replay against bounded
concurrency, not to dictate the run's real wall-clock duration. This
script measures and reports the REAL observed wall-clock duration
separately from the nominal 30-minute schedule -- expected (and
reported, not hidden) to run longer than 30 minutes because of real
request queueing under bounded concurrency, which is itself a genuine,
useful finding (see PLAN_APPROVED.md Section 12's queue-time metric).

Usage (run against a real `uvicorn` process already serving
`resolvegrid_api.main:app`, e.g. via `scripts/run_api_dev.py`):

    # Full-system run (real current defaults: caching + reranking on,
    # diagnose included in the mix):
    uv run --package resolvegrid-api python scripts/run_phase14_load_test.py \
        --label full_system --seed 42

    # Baseline run (against a SEPARATE uvicorn process started with
    # RESOLVEGRID_RESPONSE_CACHE_ENABLED=false RESOLVEGRID_SEMANTIC_CACHE_ENABLED=false
    # RESOLVEGRID_RETRIEVAL_CACHE_ENABLED=false RESOLVEGRID_RERANK_ENABLED=false --
    # see chat.py/agent_retrieval.py's matching toggles):
    uv run --package resolvegrid-api python scripts/run_phase14_load_test.py \
        --label baseline --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.load_test import (
    Arrival,
    BoundedConcurrencyLoadClient,
    NHPPConfig,
    build_run_config_json,
    build_workload_mix,
    complete_load_test_run,
    create_load_test_run,
    mean_intensity_rps,
    sample_nhpp_arrivals,
)
from resolvegrid_api.models import LoadTestRun

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "eval" / "results" / "phase14_load_v1"

FULL_DAY_SECONDS = 86400.0
COMPRESSION_FACTOR = 48.0  # 24h simulated day -> 30 real minutes nominal
NOMINAL_WINDOW_SECONDS = FULL_DAY_SECONDS / COMPRESSION_FACTOR

# Same NHPPConfig for BOTH runs (full_system and baseline) -- the paired-
# comparison variable under test is "which Phase 9-13 capabilities are
# enabled," never the arrival schedule itself. Calibrated (see this
# module's docstring) to ~1,150 total arrivals/day (~1,109 req/day mean
# rate), comfortably over the phase's "1,000+ requests/day" target.
BURST_SCHEDULE = (8 * 3600.0, 13 * 3600.0, 20 * 3600.0)  # 8am/1pm/8pm "incidents"


def build_nhpp_config() -> NHPPConfig:
    return NHPPConfig(
        duration_seconds=FULL_DAY_SECONDS,
        base_rate_rps=0.005,
        peak_rate_rps=0.03,
        burst_schedule=BURST_SCHEDULE,
        burst_mean_size=20.0,
        burst_jitter_seconds=5.0,
    )


# Real usage-ratio assumption, same as `load_test.DEFAULT_WORKLOAD_MIX`
# (and its own documented rationale) for the full-system run. The
# baseline run drops "diagnose" (weight 0 -- never selected; `rng.
# choices` renormalizes over the remaining non-zero weights
# automatically) per the phase plan's explicit instruction: "just don't
# include /tickets/{id}/diagnose requests in the baseline's workload
# mix" -- the ADK diagnostic specialist path is simply never exercised,
# not degraded/stubbed.
FULL_SYSTEM_MIX = {"chat": 0.50, "create_ticket": 0.35, "diagnose": 0.15}
BASELINE_MIX = {"chat": 0.50, "create_ticket": 0.35, "diagnose": 0.0}


def compress_arrivals(arrivals: list[Arrival], factor: float) -> list[Arrival]:
    """Divides every arrival's `.t` by `factor` -- see module docstring's
    "Compression factor" section. Preserves list order/length/is_burst
    flags exactly; only the time axis changes.
    """
    return [Arrival(t=a.t / factor, is_burst=a.is_burst) for a in arrivals]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _redis_keyspace_stats(redis_url: str) -> dict:
    """Real `redis-cli INFO stats` keyspace_hits/keyspace_misses, via a
    real redis connection -- used as a before/after delta to approximate
    aggregate cache activity across ALL Redis-backed caches this process
    uses (response/semantic/retrieval/embedding) during a run's wall-
    clock window. Documented limitation (see the Task 2 report): this is
    a single aggregate across every key in the instance, not broken out
    per cache tier -- `AgentRun.cache_hit`/`cache_hit_kind` (queried
    separately, see `query_server_metrics.py`) gives the precise
    per-tier breakdown for the response/semantic tiers; there is no
    equivalent DB-backed marker for the retrieval-cache tier, which is
    why this aggregate Redis-level proxy is reported alongside it rather
    than instead of it.
    """
    import redis

    client = redis.from_url(redis_url)
    info = client.info("stats")
    return {
        "keyspace_hits": info.get("keyspace_hits", 0),
        "keyspace_misses": info.get("keyspace_misses", 0),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize_client_results(results, actual_wall_clock_seconds: float) -> dict:
    by_kind: dict[str, list] = {}
    for r in results:
        by_kind.setdefault(r.item.kind, []).append(r)

    def _kind_summary(rs) -> dict:
        latencies = [r.latency_seconds for r in rs if r.error is None]
        statuses = [r.status_code for r in rs]
        errors = [r for r in rs if r.error is not None]
        non_2xx = [r for r in rs if r.status_code is not None and not (200 <= r.status_code < 300)]
        return {
            "count": len(rs),
            "p50_latency_s": _percentile(latencies, 0.50),
            "p95_latency_s": _percentile(latencies, 0.95),
            "p99_latency_s": _percentile(latencies, 0.99),
            "mean_latency_s": statistics.fmean(latencies) if latencies else None,
            "transport_error_count": len(errors),
            "non_2xx_count": len(non_2xx),
            "status_code_counts": {
                str(code): statuses.count(code) for code in sorted(set(statuses), key=lambda c: (c is None, c))
            },
        }

    overall_latencies = [r.latency_seconds for r in results if r.error is None]
    return {
        "total_requests": len(results),
        "actual_wall_clock_seconds": actual_wall_clock_seconds,
        "client_throughput_rps": len(results) / actual_wall_clock_seconds if actual_wall_clock_seconds > 0 else None,
        "overall": _kind_summary(results),
        "by_kind": {kind: _kind_summary(rs) for kind, rs in by_kind.items()},
        "429_count_create_ticket": sum(
            1 for r in by_kind.get("create_ticket", []) if r.status_code == 429
        ),
    }


async def _run(base_url: str, arrivals: list[Arrival], items, max_concurrency: int, timeout_seconds: float):
    client = BoundedConcurrencyLoadClient(base_url, max_concurrency=max_concurrency, timeout_seconds=timeout_seconds)
    return await client.run_schedule(arrivals, items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["full_system", "baseline"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--redis-url", default="redis://localhost:6380/0")
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    config = build_nhpp_config()
    full_day_arrivals = sample_nhpp_arrivals(config, seed=args.seed)
    compressed_arrivals = compress_arrivals(full_day_arrivals, COMPRESSION_FACTOR)
    mix = FULL_SYSTEM_MIX if args.label == "full_system" else BASELINE_MIX

    print(
        f"[{args.label}] sampled {len(full_day_arrivals)} arrivals over a simulated "
        f"{FULL_DAY_SECONDS:.0f}s day (mean rate {mean_intensity_rps(config):.5f} req/s "
        f"= {mean_intensity_rps(config) * FULL_DAY_SECONDS:.0f} req/day-equivalent), "
        f"compressed {COMPRESSION_FACTOR:.0f}x into a {NOMINAL_WINDOW_SECONDS:.0f}s "
        f"({NOMINAL_WINDOW_SECONDS / 60:.1f} min) nominal real-time window."
    )

    engine = create_engine(DATABASE_URL)
    rng = random.Random(args.seed)
    with Session(engine) as session:
        items = build_workload_mix(session, rng, len(compressed_arrivals), mix=mix)
        kind_counts: dict[str, int] = {}
        for it in items:
            kind_counts[it.kind] = kind_counts.get(it.kind, 0) + 1
        print(f"[{args.label}] workload mix ({len(items)} items): {kind_counts}")

        config_json = build_run_config_json(args.seed, config, mix=mix)
        run_row = create_load_test_run(session, config_json=config_json, git_commit=_git_commit())
        session.commit()
        run_id = run_row.id

    redis_stats_before = _redis_keyspace_stats(args.redis_url)
    run_started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    results = asyncio.run(_run(args.base_url, compressed_arrivals, items, args.max_concurrency, args.timeout_seconds))

    actual_wall_clock_seconds = time.perf_counter() - t0
    run_completed_at = datetime.now(timezone.utc)
    redis_stats_after = _redis_keyspace_stats(args.redis_url)

    client_summary = summarize_client_results(results, actual_wall_clock_seconds)
    client_summary["redis_keyspace_hits_delta"] = (
        redis_stats_after["keyspace_hits"] - redis_stats_before["keyspace_hits"]
    )
    client_summary["redis_keyspace_misses_delta"] = (
        redis_stats_after["keyspace_misses"] - redis_stats_before["keyspace_misses"]
    )

    print(f"[{args.label}] DONE in {actual_wall_clock_seconds:.1f}s real wall-clock " f"(nominal was {NOMINAL_WINDOW_SECONDS:.0f}s).")
    print(json.dumps(client_summary, indent=2, default=str))

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _RESULTS_DIR / f"{args.label}_raw.json"
    raw_payload = {
        "label": args.label,
        "run_id": run_id,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "base_url": args.base_url,
        "max_concurrency": args.max_concurrency,
        "timeout_seconds": args.timeout_seconds,
        "compression_factor": COMPRESSION_FACTOR,
        "nominal_window_seconds": NOMINAL_WINDOW_SECONDS,
        "started_at": run_started_at.isoformat(),
        "completed_at": run_completed_at.isoformat(),
        "mix": mix,
        "config_json": config_json,
        "client_summary": client_summary,
        "results": [
            {
                "kind": r.item.kind,
                "employee_id": r.item.employee_id,
                "path_params": r.item.path_params,
                "scheduled_at": r.scheduled_at,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "latency_seconds": r.latency_seconds,
                "status_code": r.status_code,
                "error": r.error,
            }
            for r in results
        ],
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2, default=str), encoding="utf-8")
    print(f"[{args.label}] raw results written to {raw_path}")

    with Session(engine) as session:
        run_row = session.get(LoadTestRun, run_id)
        complete_load_test_run(
            session,
            run_row,
            result_summary_json=json.dumps(client_summary, default=str),
            raw_results_path=str(raw_path.relative_to(_REPO_ROOT)),
        )
        session.commit()
    engine.dispose()

    print(f"[{args.label}] LoadTestRun id={run_id} completed_at recorded.")


if __name__ == "__main__":
    main()
