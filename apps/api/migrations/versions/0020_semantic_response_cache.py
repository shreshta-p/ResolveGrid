"""semantic response cache

Phase 12 Task 5: adds `semantic_response_cache` -- the new table storing
`(scope_sig, kb_version, query_embedding vector, answer/citations_json/
caption/intent/risk_level, created_at)` rows that `resolvegrid_api.semantic_
cache` queries by real pgvector `<=>` cosine distance, scoped to the same
`(scope_sig, kb_version)` authz/KB-version boundary Task 2's retrieval cache
and Task 3's response cache already establish. See `models/knowledge.py`'s
`SemanticResponseCache` docstring for the full schema-design rationale
(why Postgres not Redis, why no ANN index, why no TTL column, why inline
JSON citations rather than a separate response table).

The composite `(scope_sig, kb_version)` index is the security-critical
piece: every real lookup's `WHERE` clause filters on it BEFORE any
similarity ordering happens, so a candidate row from a different scope can
never even be considered, let alone matched.

`vector` extension already enabled (migration 0010) -- not re-enabled here.

Revision ID: 0020
Revises: 0019
Create Date: 2026-10-09 00:00:00.000001

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = '0020'
down_revision: Union[str, None] = '0019'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'semantic_response_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scope_sig', sa.String(), nullable=False),
        sa.Column('kb_version', sa.Integer(), nullable=False),
        sa.Column('query_embedding', Vector(768), nullable=False),
        sa.Column('answer', sa.String(), nullable=False),
        sa.Column('citations_json', sa.String(), nullable=False),
        sa.Column('caption', sa.String(), nullable=False),
        sa.Column('intent', sa.String(), nullable=False),
        sa.Column('risk_level', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_semantic_response_cache_scope_kb',
        'semantic_response_cache',
        ['scope_sig', 'kb_version'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_semantic_response_cache_scope_kb', table_name='semantic_response_cache')
    op.drop_table('semantic_response_cache')
