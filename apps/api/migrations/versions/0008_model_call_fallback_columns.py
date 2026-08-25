"""model call fallback columns

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-25 12:17:12.682329

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0008'
down_revision: Union[str, None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default=sa.false() so existing model_call rows (all written
    # before LiteLLM fallback routing existed) backfill cleanly to False
    # rather than being left NULL.
    op.add_column('model_call', sa.Column('fallback_occurred', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('model_call', sa.Column('serving_model_group', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_call', 'serving_model_group')
    op.drop_column('model_call', 'fallback_occurred')
