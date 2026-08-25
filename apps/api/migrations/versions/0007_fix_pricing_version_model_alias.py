"""fix pricing_version model to match the LiteLLM-facing alias

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-24 06:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0007'
down_revision: Union[str, None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 0006 seeded this row with model="qwen3:14b" -- the raw underlying Ollama
    # tag. But llm_gateway.complete()'s CompletionResult.model always carries
    # whatever `model` value the caller requested (default "local-qwen3",
    # LiteLLM's own alias for that model in infra/litellm/config.yaml), and
    # LiteLLM's own OpenAI-compatible response JSON echoes that same alias
    # back too (confirmed via a real call: {"model":"local-qwen3", ...}) --
    # there is no path by which "qwen3:14b" is ever what a ModelCall row's
    # `model` column actually holds. So the pricing lookup in
    # apps/api/src/resolvegrid_api/routers/tickets.py's
    # _current_pricing_version() could never match this row in real (non-
    # mocked) traffic, silently leaving pricing_version_id NULL on every real
    # call. Financially inert today (Ollama is $0/$0 either way) but breaks
    # PricingVersion's own traceability intent -- found in code review,
    # fixed forward rather than editing the already-applied 0006.
    op.execute("UPDATE pricing_version SET model = 'local-qwen3' WHERE provider = 'ollama' AND model = 'qwen3:14b'")


def downgrade() -> None:
    op.execute("UPDATE pricing_version SET model = 'qwen3:14b' WHERE provider = 'ollama' AND model = 'local-qwen3'")
