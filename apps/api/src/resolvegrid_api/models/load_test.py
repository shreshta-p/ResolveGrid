from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class LoadTestRun(Base):
    """One execution of the Phase 14 load-test harness (`resolvegrid_api.
    load_test`) -- a reproducibility manifest, not a metrics table.

    Deliberately NOT a metrics store: `AgentRun`/`Span`/`ModelCall`
    (Phase 6/9) already own every real per-request latency/token/cost
    number a load run touches (per this phase's plan doc: "computed from
    real AgentRun/Span/ModelCall rows, not the harness's own client-side
    timers alone"). Re-deriving p50/p95/p99 etc. into columns here would
    create a second, divorceable source of truth for numbers this project
    already owns -- the same anti-pattern `EvalCaseResult.details_json`
    avoids by staying a grading record, not a duplicate telemetry store.
    This table instead answers "what config produced this run, and where
    do I find its materialized output" -- exactly the reproducibility
    metadata `PLAN_APPROVED.md`'s schema note calls for (seed, duration,
    target rps, burst schedule, code version), so a later comparison can
    confirm two runs are actually comparable (same seed/duration/rps/
    burst-schedule/git_commit) before trusting a before/after delta.

    `config_json`/`result_summary_json` are plain `String` columns (not a
    native JSON/JSONB column type), matching `EvalCaseResult.details_json`
    (migration 0013) and `AuditLog.before_json`/`after_json`/
    `metadata_json` (migration 0004)'s established convention in this
    schema: every other free-form structured-data column in this project
    is a serialized JSON string, not a native JSON column -- consistent,
    not novel, and this table has no query needing JSON operators (it's
    read back whole by the harness/report code, never filtered on a key
    inside the blob).

    `completed_at IS NULL` means the run is still in progress (or crashed
    before finishing) -- mirrors `EvalRun.completed_at`'s existing nullable
    -until-done convention (migration 0013) rather than adding a separate
    redundant status column this task's explicit field list didn't ask
    for.
    """

    __tablename__ = "load_test_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    # Serialized JSON: {"seed": int, "duration_seconds": float,
    # "target_rps": float, "burst_schedule": [float, ...], ...}. See
    # `resolvegrid_api.load_test.build_run_config_json`.
    config_json: Mapped[str]
    git_commit: Mapped[str]
    # Populated when the run completes; NULL while in progress. Serialized
    # JSON: aggregate throughput/latency/error-rate/cache-hit summary --
    # the *report*, not the raw per-request rows (those live in the file
    # at raw_results_path, per PLAN_APPROVED.md's "stored raw results"
    # requirement).
    result_summary_json: Mapped[str | None] = mapped_column(default=None)
    # Filesystem path (relative to repo root) to the raw per-request
    # results this run produced, e.g. under eval/results/phase14_load_v1/.
    # NULL until the run has actually written that file.
    raw_results_path: Mapped[str | None] = mapped_column(default=None)
