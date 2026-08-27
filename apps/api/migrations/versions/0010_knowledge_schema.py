"""knowledge schema

Adds the Phase 7 Task 1 knowledge/RAG schema: document, document_version,
chunk, embedding, ingestion_run. Enables the `vector` extension (the
pgvector/pgvector:pg16 image already ships it -- this just turns it on for
this database, per infra/docker-compose.yml) and creates an HNSW index on
embedding.vector for cosine-distance ANN search.

Autogenerate also proposed dropping the `checkpoints`/`checkpoint_blobs`/
`checkpoint_writes`/`checkpoint_migrations` tables -- those are LangGraph's
own AsyncPostgresSaver tables (Phase 6), not part of this codebase's
SQLAlchemy Base.metadata, so autogenerate always flags them as "removed".
They are intentionally left untouched here; this migration only lists the
tables it actually adds.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-27 02:42:16.015174

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision: str = '0010'
down_revision: Union[str, None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table('document',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source_type', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('url', sa.String(), nullable=True),
    sa.Column('publisher', sa.String(), nullable=True),
    sa.Column('retrieved_at', sa.DateTime(), nullable=True),
    sa.Column('effective_date', sa.DateTime(), nullable=True),
    sa.Column('checksum', sa.String(), nullable=False),
    sa.Column('license', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('supersedes_document_id', sa.Integer(), nullable=True),
    sa.Column('access_scope_tags', postgresql.ARRAY(sa.String()), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['supersedes_document_id'], ['document.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('ingestion_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('parser_version', sa.String(), nullable=False),
    sa.Column('chunking_version', sa.String(), nullable=False),
    sa.Column('embedding_version', sa.String(), nullable=False),
    sa.Column('documents_processed', sa.Integer(), nullable=False),
    sa.Column('chunks_created', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('error_message', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('document_version',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('parser_version', sa.String(), nullable=False),
    sa.Column('chunking_version', sa.String(), nullable=False),
    sa.Column('content_hash', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_version_document_id', 'document_version', ['document_id'], unique=False)
    op.create_table('chunk',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_version_id', sa.Integer(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('text', sa.String(), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('parent_chunk_id', sa.Integer(), nullable=True),
    sa.Column('metadata_json', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_version_id'], ['document_version.id'], ),
    sa.ForeignKeyConstraint(['parent_chunk_id'], ['chunk.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_chunk_document_version_id', 'chunk', ['document_version_id'], unique=False)
    op.create_table('embedding',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('chunk_id', sa.Integer(), nullable=False),
    sa.Column('embedding_model', sa.String(), nullable=False),
    sa.Column('embedding_version', sa.String(), nullable=False),
    sa.Column('vector', Vector(768), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['chunk_id'], ['chunk.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_embedding_chunk_id', 'embedding', ['chunk_id'], unique=False)
    # HNSW index for cosine-distance ANN search (pgvector docs:
    # https://github.com/pgvector/pgvector#hnsw -- `vector_cosine_ops` is the
    # operator class for cosine distance; m/ef_construction left at pgvector's
    # own defaults of 16/64 rather than pinned here).
    op.execute(
        'CREATE INDEX ix_embedding_vector_hnsw ON embedding '
        'USING hnsw (vector vector_cosine_ops)'
    )


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS ix_embedding_vector_hnsw')
    op.drop_index('ix_embedding_chunk_id', table_name='embedding')
    op.drop_table('embedding')
    op.drop_index('ix_chunk_document_version_id', table_name='chunk')
    op.drop_table('chunk')
    op.drop_index('ix_document_version_document_id', table_name='document_version')
    op.drop_table('document_version')
    op.drop_table('ingestion_run')
    op.drop_table('document')
    # `vector` extension intentionally left enabled on downgrade -- cheap to
    # leave in place, and nothing else in this migration chain depends on
    # dropping it back out.
