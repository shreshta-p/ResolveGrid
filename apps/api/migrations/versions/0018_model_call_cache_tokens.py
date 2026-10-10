"""model call cache tokens

Phase 12 Task 4: adds `model_call.cache_creation_input_tokens`/
`model_call.cache_read_input_tokens`, the real Anthropic provider-side
prompt-prefix-caching token counts LiteLLM passes through from a real
`cloud-primary` completion's `usage` object (distinct from this phase's
other, Redis-backed application caches -- see docs/superpowers/plans/
2026-09-29-phase12-caching.md's Task 4 section). Mirrors `trace_id`'s
Phase 11 Task 1 migration (0014) for shape/style: a new column, NOT NULL
with a server-side default so every pre-existing row backfills to the
correct, honest value (0 -- no row written before this migration could
possibly have real cache token data to report) with no separate UPDATE
pass required.

Revision ID: 0018
Revises: 0017
Create Date: 2026-10-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0018'
down_revision: Union[str, None] = '0017'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'model_call',
        sa.Column('cache_creation_input_tokens', sa.Integer(), nullable=False, server_default='0'),
    )
    op.add_column(
        'model_call',
        sa.Column('cache_read_input_tokens', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('model_call', 'cache_read_input_tokens')
    op.drop_column('model_call', 'cache_creation_input_tokens')
