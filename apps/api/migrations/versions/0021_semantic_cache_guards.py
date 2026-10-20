"""semantic cache guards

Phase 12 Task 5 code-review follow-up. Two independent, real schema
guards, both responding to the same review:

1. `ix_semantic_response_cache_created_at` -- a real btree index on
   `semantic_response_cache.created_at`. Before this migration, the ONLY
   index on this table was the `(scope_sig, kb_version)` composite
   (migration 0020), which supports a real LOOKUP's equality filters but
   gives a cleanup pass that deletes purely by age (`DELETE ... WHERE
   created_at < cutoff`, with no `scope_sig`/`kb_version` predicate -- the
   real shape `resolvegrid_api.semantic_cache.purge_expired_semantic_
   cache_entries` uses) nothing to use but a full table scan, which would
   itself become slower as the exact same unbounded-growth problem the
   cleanup pass exists to fix got worse. This closes that gap BEFORE the
   growth problem has a chance to compound it.

2. `ck_agent_run_cache_hit_kind_requires_cache_hit` -- a real DB `CHECK`
   constraint: `cache_hit_kind IS NULL OR cache_hit = true`. Before this
   migration, nothing but convention (both real write sites in `routers/
   chat.py` happening to set both fields together) prevented a future
   write path from setting an inconsistent combination (e.g. `cache_hit=
   false` with a non-null `cache_hit_kind` -- "which cache kind served a
   run that wasn't a cache hit at all" is nonsensical). Making this
   structurally impossible at the DB layer, not just "currently true by
   convention", is strictly cheaper than relying on every future writer
   remembering the invariant.

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-09 00:00:00.000002

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0021'
down_revision: Union[str, None] = '0020'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_semantic_response_cache_created_at',
        'semantic_response_cache',
        ['created_at'],
        unique=False,
    )
    op.create_check_constraint(
        'ck_agent_run_cache_hit_kind_requires_cache_hit',
        'agent_run',
        'cache_hit_kind IS NULL OR cache_hit = true',
    )


def downgrade() -> None:
    op.drop_constraint('ck_agent_run_cache_hit_kind_requires_cache_hit', 'agent_run', type_='check')
    op.drop_index('ix_semantic_response_cache_created_at', table_name='semantic_response_cache')
