"""model call trace id

Phase 11 Task 1: adds `model_call.trace_id`, the real OTel trace id (32
lowercase hex chars) captured by the new shared `model_call_logging.
log_completion()` helper, so a ModelCall row can later be correlated to a
real trace in Langfuse (Task 5). Nullable -- pre-existing rows have none and
are never backfilled.

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0014'
down_revision: Union[str, None] = '0013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_call', sa.Column('trace_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_call', 'trace_id')
