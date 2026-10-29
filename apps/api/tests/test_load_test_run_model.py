"""Phase 14 Task 1 schema smoke test: `LoadTestRun` round-trip, nullable
-until-complete fields, and the real `create_load_test_run`/
`complete_load_test_run` helpers in `resolvegrid_api.load_test`.

Mirrors `test_evaluation_models.py`'s shape (migration 0013's own
schema-only smoke test): proof the new table from migration 0022 can be
created, committed (flushed within the transactional `db_session`
fixture), and queried back with the expected nullable-until-done
semantics -- no business logic beyond that, per this task's scope.
"""

from resolvegrid_api.load_test import (
    NHPPConfig,
    build_run_config_json,
    complete_load_test_run,
    create_load_test_run,
)
from resolvegrid_api.models import LoadTestRun


def test_load_test_run_starts_with_null_completion_fields(db_session):
    config = NHPPConfig(duration_seconds=60.0, base_rate_rps=0.01, peak_rate_rps=0.05)
    config_json = build_run_config_json(seed=42, config=config)

    run = create_load_test_run(db_session, config_json=config_json, git_commit="a" * 40)

    fetched = db_session.get(LoadTestRun, run.id)
    assert fetched is not None
    assert fetched.config_json == config_json
    assert fetched.git_commit == "a" * 40
    assert fetched.started_at is not None
    assert fetched.completed_at is None
    assert fetched.result_summary_json is None
    assert fetched.raw_results_path is None


def test_complete_load_test_run_sets_completion_fields(db_session):
    config = NHPPConfig(duration_seconds=60.0, base_rate_rps=0.01, peak_rate_rps=0.05)
    run = create_load_test_run(db_session, config_json=build_run_config_json(seed=1, config=config))

    complete_load_test_run(
        db_session,
        run,
        result_summary_json='{"p95_latency_ms": 420}',
        raw_results_path="eval/results/phase14_load_v1/run_1.jsonl",
    )

    fetched = db_session.get(LoadTestRun, run.id)
    assert fetched.completed_at is not None
    assert fetched.result_summary_json == '{"p95_latency_ms": 420}'
    assert fetched.raw_results_path == "eval/results/phase14_load_v1/run_1.jsonl"


def test_create_load_test_run_defaults_git_commit_to_the_real_current_commit(db_session):
    config = NHPPConfig(duration_seconds=60.0, base_rate_rps=0.01, peak_rate_rps=0.05)
    run = create_load_test_run(db_session, config_json=build_run_config_json(seed=1, config=config))

    # Real `git rev-parse HEAD` output: 40 hex chars, or "unknown" if this
    # somehow isn't a git checkout -- never empty/None.
    assert run.git_commit
    assert run.git_commit == "unknown" or len(run.git_commit) == 40
