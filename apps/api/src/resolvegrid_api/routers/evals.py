"""Phase 10 Task 8: read-only admin API surfacing the `EvalRun`/
`EvalCaseResult` rows Task 7's real Arq batch runner
(`resolvegrid_api.eval_worker.run_eval_suite`) persists -- `GET /evals/runs`
(newest-first list, each row annotated with a per-dimension + overall
pass-rate summary) and `GET /evals/runs/{id}` (full per-case breakdown for
one run). Staff/admin-only (`"eval.view"`, added to `packages/authz`'s
`_STAFF_ONLY_ACTIONS`), mirroring `routers/approvals.py`'s
authz-dependency-usage and list-endpoint shape exactly -- see that module
for the established pattern this one follows.

Why staff-only: an eval run's results can surface internal system behavior
(retrieved chunk ids, tool params, approval outcomes, judge verdicts) across
cases exercising arbitrary fixture employees -- never a single employee's
own resource the way a ticket or directory entry is, so (matching
`approval.list`/`approval.decide`'s precedent) a principal with no
admin/department-scoped analyst-or-approver grant is denied outright, never
downgraded to a self-scoped `Decision`.

Pass-rate math: `_pass_rate_summary` groups `EvalCaseResult` rows by
`dimension` and computes `passed_count / total_count` per dimension plus one
overall rate across every case in the run. Division-by-zero is handled
explicitly -- a dimension (or an entire run) with zero graded cases reports
`pass_rate=None` rather than raising `ZeroDivisionError` or silently
reporting a misleading `0.0`/`1.0`.

`details_json` handling: `EvalCaseResult.details_json` is stored as a JSON
string (this codebase's established "JSON-as-text column" convention, same
as `ApprovalRequest.action_params_json` -- see `approvals.py`'s
`_approval_request_to_dict`). `_eval_case_result_to_dict` parses it back into
a real dict via `json.loads` before it ever reaches a response body, so the
frontend never has to `JSON.parse` a nested JSON string itself.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.db import get_db
from resolvegrid_api.deps import get_principal
from resolvegrid_api.models.evaluation import EvalCaseResult, EvalRun
from resolvegrid_authz import Principal, authorize

router = APIRouter(prefix="/evals", tags=["evals"])

# No existing list endpoint in this codebase (GET /approvals, GET /tickets,
# etc.) takes an explicit limit/offset -- each returns its full,
# small-by-construction result set instead. `EvalRun` rows can accumulate
# indefinitely over the life of the project (one row per batch run, run as
# often as CI/local verification runs the suite), so this endpoint is the
# first to need real pagination. A conservative default (recent runs are
# what an admin actually wants to see first) plus a hard upper bound (no
# caller can force an unbounded full-table scan/response) is this task's own
# sensible convention, not a port of an existing precedent.
DEFAULT_LIMIT = 20
MAX_LIMIT = 200


def _rate(passed: int, total: int) -> float | None:
    """`passed / total`, or `None` when `total == 0` -- a dimension (or an
    entire run) with zero graded cases must never raise `ZeroDivisionError`
    nor silently report a misleading `0.0`/`1.0` pass rate for "no data."
    """
    return (passed / total) if total else None


def _pass_rate_summary(results: list[EvalCaseResult]) -> dict:
    """Per-dimension `passed_count`/`total_count`/`pass_rate`, plus one
    overall `passed_count`/`total_count`/`pass_rate` across every case in
    `results` -- the "per-dimension pass-rate summary computed by
    joining/aggregating EvalCaseResult rows grouped by dimension" the plan
    doc's Task 8 section asks for.
    """
    by_dimension: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_dimension.setdefault(result.dimension, {"passed_count": 0, "total_count": 0})
        bucket["total_count"] += 1
        if result.passed:
            bucket["passed_count"] += 1

    dimensions = {
        dimension: {
            "passed_count": counts["passed_count"],
            "total_count": counts["total_count"],
            "pass_rate": _rate(counts["passed_count"], counts["total_count"]),
        }
        for dimension, counts in sorted(by_dimension.items())
    }
    total_passed = sum(counts["passed_count"] for counts in by_dimension.values())
    total_count = sum(counts["total_count"] for counts in by_dimension.values())
    return {
        "dimensions": dimensions,
        "overall_passed_count": total_passed,
        "overall_total_count": total_count,
        "overall_pass_rate": _rate(total_passed, total_count),
    }


def _eval_run_to_dict(row: EvalRun) -> dict:
    return {
        "id": row.id,
        "dataset_version": row.dataset_version,
        "prompt_version": row.prompt_version,
        "workflow_version": row.workflow_version,
        "retriever_version": row.retriever_version,
        "reranker_version": row.reranker_version,
        "embeddings_version": row.embeddings_version,
        "generation_model_version": row.generation_model_version,
        "judge_version": row.judge_version,
        "tool_schema_version": row.tool_schema_version,
        "git_commit": row.git_commit,
        "status": row.status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "error_message": row.error_message,
    }


def _eval_case_result_to_dict(row: EvalCaseResult) -> dict:
    return {
        "id": row.id,
        "case_id": row.case_id,
        "dimension": row.dimension,
        "passed": row.passed,
        "score": row.score,
        "grader_type": row.grader_type,
        # Parsed back to a real dict -- see module docstring's
        # "details_json handling" section. `None` when the grader recorded
        # no structured details at all (the column is nullable).
        "details": json.loads(row.details_json) if row.details_json is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/runs")
def list_eval_runs(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict]:
    """Newest-first `EvalRun` list, each row annotated with `summary` (see
    `_pass_rate_summary`) computed from that run's own `EvalCaseResult`
    rows. Paginated via `limit`/`offset` (see module-level constants for the
    default/max).
    """
    decision = authorize(principal, "eval.view")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    runs = session.scalars(
        select(EvalRun).order_by(EvalRun.started_at.desc(), EvalRun.id.desc()).offset(offset).limit(limit)
    ).all()
    if not runs:
        return []

    # One query for every result row across every returned run (grouped
    # in Python below), rather than N per-run queries -- this endpoint's
    # whole point is a multi-run list view.
    run_ids = [run.id for run in runs]
    results = session.scalars(
        select(EvalCaseResult).where(EvalCaseResult.eval_run_id.in_(run_ids))
    ).all()
    results_by_run: dict[int, list[EvalCaseResult]] = {}
    for result in results:
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    return [
        {**_eval_run_to_dict(run), "summary": _pass_rate_summary(results_by_run.get(run.id, []))}
        for run in runs
    ]


@router.get("/runs/{eval_run_id}")
def get_eval_run(
    eval_run_id: int,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict:
    """Full per-case breakdown for one `EvalRun`: the run's own fields (full
    version tuple, status, timestamps), its pass-rate `summary`, and every
    `EvalCaseResult` row belonging to it (`case_id`, `dimension`, `passed`,
    `score`, `grader_type`, `details` parsed back to a real dict).
    """
    decision = authorize(principal, "eval.view")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    run = session.get(EvalRun, eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="eval run not found")

    results = session.scalars(
        select(EvalCaseResult).where(EvalCaseResult.eval_run_id == eval_run_id).order_by(EvalCaseResult.id)
    ).all()

    return {
        **_eval_run_to_dict(run),
        "summary": _pass_rate_summary(results),
        "case_results": [_eval_case_result_to_dict(result) for result in results],
    }
