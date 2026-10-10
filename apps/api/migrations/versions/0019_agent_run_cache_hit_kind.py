"""agent run cache hit kind

Phase 12 Task 5: adds `agent_run.cache_hit_kind` (nullable string) --
distinguishes WHICH cache mechanism served a run: "exact" (Task 3's
Redis-backed `response_cache`), "semantic" (this task's pgvector-backed
`semantic_cache`), or NULL (a genuine real-graph run, including every row
written before this column existed). `cache_hit` (migration 0017) is left
unchanged and stays the simple boolean "was this run served from ANY cache"
signal -- this is an additive, finer-grained column on top of it, not a
replacement (see `models/agent.py`'s `AgentRun.cache_hit_kind` docstring for
the full "why a second column, not a widened cache_hit" reasoning).

No `server_default` needed (unlike `cache_hit`'s `server_default=false()`):
NULL is already the correct, honest backfill value for every pre-existing
row -- none of them could have been a "semantic" hit (this cache didn't
exist yet) or needed this finer distinction at all.

Revision ID: 0019
Revises: 0018
Create Date: 2026-10-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0019'
down_revision: Union[str, None] = '0018'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'agent_run',
        sa.Column('cache_hit_kind', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('agent_run', 'cache_hit_kind')
