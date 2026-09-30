"""agent run cache hit

Phase 12 Task 3: adds `agent_run.cache_hit` (boolean, default False), the
dedicated field this task's own response cache (`resolvegrid_api.
response_cache`) uses to honestly mark an `AgentRun` row set as "served
entirely from the exact-response cache, the real graph never ran" --
chosen over encoding this into `AgentRun.status` (which stays "running" |
"completed" | "error" for both hit and miss, unchanged, so every existing
status-based query/assertion continues to mean what it already meant) or
into `Span.detail_json` per-row (that field is a small, optional, per-Span
structured extra -- a run-level property like "was this whole run served
from cache" belongs on the run itself, queryable with a single indexed-free
boolean check, not by parsing JSON out of one of the run's four Span rows).

`server_default='false'` (not merely a Python-side `default=False`) so
every pre-existing row -- all of which predate this cache entirely and were
genuinely NOT cache hits -- backfills to the correct, honest value with no
separate UPDATE pass, matching this column's own non-nullable contract.

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0017'
down_revision: Union[str, None] = '0016'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'agent_run',
        sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column('agent_run', 'cache_hit')
