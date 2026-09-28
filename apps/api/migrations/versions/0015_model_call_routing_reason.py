"""model call routing reason

Phase 11 Task 4: adds `model_call.routing_reason`, a free-text record of
which real policy input drove a call's `model=` choice (e.g.
"risk_level=high" for a `chat.compose_response` call routed to
`cloud-primary` -- see `routing.py`'s module docstring for the real routing
policy this records, and `models/telemetry.py`'s `ModelCall.routing_reason`
docstring for why this is judged genuinely not redundant with the existing
`provider`/`model`/`fallback_occurred`/`serving_model_group` columns).
Nullable -- every call site without a real routing policy yet (ticket
summarization, classify_intent, judge/eval closures) never sets it, and
pre-existing rows are never backfilled.

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0015'
down_revision: Union[str, None] = '0014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_call', sa.Column('routing_reason', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_call', 'routing_reason')
