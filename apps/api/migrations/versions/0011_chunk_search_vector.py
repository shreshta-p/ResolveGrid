"""chunk search vector

Adds a Postgres full-text-search column to `chunk` for Phase 7 Task 4
(lexical search + RRF fusion).

Decision, made deliberately rather than left ambiguous (per this task's
instructions): a *stored generated* `tsvector` column with a GIN index,
not an inline `to_tsvector(text)` computed per-query. Reasoning:

- The plan doc calls this phase's corpus "bounded", which would make an
  inline per-query `to_tsvector('english', text)` scan tolerable
  performance-wise (no index needed to stay fast at this scale).
- But a stored+indexed column is barely more code (one migration, one
  ORM column) and is the standard, correct way to do Postgres full-text
  search in a real system -- it keeps `ts_rank_cd` queries index-backed
  (`USING gin`) instead of a sequential scan that recomputes
  `to_tsvector` for every row on every query, which stops being "fine"
  the moment the corpus stops being bounded. Since this phase's stated
  goal is "making lexical search real" (not a throwaway prototype),
  the stored/indexed version was chosen.
- `GENERATED ALWAYS AS (...) STORED` is used (not a trigger) because
  Postgres 12+ supports generated columns natively and the expression
  (`to_tsvector('english', text)`) is a pure, immutable function of
  another column in the same row -- exactly the case generated columns
  are for. A trigger would be strictly more code for the same effect.
- The DDL is issued via `op.execute` (matching migration 0010's
  precedent for pgvector-specific raw SQL) rather than
  `op.add_column(..., sa.Computed(...))`, because this needs to run as
  an `ALTER TABLE ... ADD COLUMN ... GENERATED ALWAYS AS (...) STORED`
  against a table that may already have rows (Postgres computes the
  generated value for existing rows at ALTER time) -- verified directly
  against the running `resolvegrid-postgres` container (Postgres 16.15)
  before writing this migration, both the bare `ALTER TABLE ... ADD
  COLUMN ... GENERATED ALWAYS AS (...) STORED` form and the equivalent
  `sa.Computed(...)`-driven `CREATE TABLE` form.
- `'english'` is a fixed literal text search configuration (not a
  column reference), matching this phase's `to_tsvector('english', ...)`
  usage already proven in the grounding queries against the live DB --
  a fixed regconfig literal is required for the expression to be
  immutable (a prerequisite for a `STORED` generated column).

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = '0011'
down_revision: Union[str, None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunk ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED NOT NULL"
    )
    op.execute(
        "CREATE INDEX ix_chunk_search_vector ON chunk USING gin (search_vector)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunk_search_vector")
    op.execute("ALTER TABLE chunk DROP COLUMN search_vector")
