"""load test run

Phase 14 Task 1 schema: `LoadTestRun`, a reproducibility manifest for one
execution of the new asyncio load-test harness
(`resolvegrid_api.load_test`), per `PLAN_APPROVED.md`'s schema note for
this phase and the phase plan doc's exact field list (`docs/superpowers/
plans/2026-10-29-phase14-performance-load-testing.md`, Task 1). Schema +
migration only -- no business logic.

See `resolvegrid_api.models.load_test.LoadTestRun`'s own docstring for the
full reasoning: why this is a manifest, not a metrics table (`AgentRun`/
`Span`/`ModelCall` already own every real per-request number); why
`config_json`/`result_summary_json` are plain `String` columns rather
than a native JSON/JSONB type; and why `completed_at` is nullable instead
of a separate status column.

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0022'
down_revision: Union[str, None] = '0021'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('load_test_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('config_json', sa.String(), nullable=False),
    sa.Column('git_commit', sa.String(), nullable=False),
    sa.Column('result_summary_json', sa.String(), nullable=True),
    sa.Column('raw_results_path', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('load_test_run')
