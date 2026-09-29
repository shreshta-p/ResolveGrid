"""seed anthropic (cloud-primary) pricing version

Phase 11 Task 7 (closing task): addresses the honest gap flagged in Task 4's
own SECURITY.md write-up -- until now, no `PricingVersion` row existed for
`provider="anthropic"`, so every real `cloud-primary` `ModelCall` row
(ids 491-493 during Task 4's own verification) computed
`estimated_cost_usd=0.0` regardless of real spend, silently under-reporting
this application's own cost ledger for the only paid provider this phase
actually reaches. LiteLLM's own `/key/info` `spend` figure remained the
authoritative number in the meantime (Task 4's real verification cited it
directly), but that's a workaround, not a fix -- this migration is the fix.

Seeded row: `provider="anthropic", model="cloud-primary"` -- matching the
`ModelCall.model` column's established convention (the LiteLLM-facing
`model_name` alias from `infra/litellm/config.yaml`, e.g. "cloud-primary",
never the raw underlying provider model id -- the exact same convention
migration 0007 already fixed for the `ollama`/`local-qwen3` row, and the
only identifier `current_pricing_version`'s lookup in
`model_call_logging.py` ever actually queries by).

Pricing verified against Anthropic's real, current published rate for
`claude-haiku-4-5-20251001` (confirmed via a live web fetch of
https://claude.com/pricing on 2026-09-28, not assumed from training data):
$1.00 / 1M input tokens, $5.00 / 1M output tokens -- i.e. $0.001 / 1k input,
$0.005 / 1k output. `cloud-fallback` (OpenAI `gpt-4o-mini`) is deliberately
NOT seeded here -- no real `cloud-fallback` completion has ever actually
succeeded in this environment (Task 4's real verification found
`OPENAI_API_KEY` has zero real credit, confirmed via a direct probe against
api.openai.com), so seeding a pricing row for a path that has never
produced a real `ModelCall` row would be speculative, not evidence-based;
add it the same way (a new migration) whenever `cloud-fallback` first
actually serves a real completion.

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0016'
down_revision: Union[str, None] = '0015'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pricing_version = sa.table(
        'pricing_version',
        sa.column('provider', sa.String()),
        sa.column('model', sa.String()),
        sa.column('input_cost_per_1k_tokens_usd', sa.Float()),
        sa.column('output_cost_per_1k_tokens_usd', sa.Float()),
    )
    op.bulk_insert(
        pricing_version,
        [
            {
                'provider': 'anthropic',
                'model': 'cloud-primary',
                'input_cost_per_1k_tokens_usd': 0.001,
                'output_cost_per_1k_tokens_usd': 0.005,
            },
        ],
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM pricing_version WHERE provider = 'anthropic' AND model = 'cloud-primary'"
    )
