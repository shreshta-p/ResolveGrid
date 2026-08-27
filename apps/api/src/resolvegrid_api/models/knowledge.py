from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from resolvegrid_api.models.base import Base

# nomic-embed-text v1.5 -- the embedding model Phase 7's ingestion pipeline
# targets (plan.md's RAG design section, via Ollama's `/api/embed`) -- outputs
# 768-dimensional vectors at its full/default dimensionality (its published
# model card documents Matryoshka support for truncating down to 64/128/256/
# 512, but 768 is the untruncated default this phase uses). Empirically
# confirmed against a real local Ollama 0.32.15 container running
# `nomic-embed-text` (Phase 7 Task 3, see
# `services/retrieval/src/resolvegrid_retrieval/embedder.py`'s
# `NOMIC_EMBED_TEXT_DIM` constant) -- POST /api/embed returned genuine
# 768-length vectors, matching this column's fixed dimension.
EMBEDDING_DIM = 768


class Document(Base):
    """A single source document (public-attributable or synthetic-private).

    See plan.md's "Knowledge and enterprise data" section for the two
    labeled source kinds this schema must support.
    """

    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str]  # "public" | "synthetic_private"
    title: Mapped[str]
    url: Mapped[str | None] = mapped_column(default=None)
    publisher: Mapped[str | None] = mapped_column(default=None)
    retrieved_at: Mapped[datetime | None] = mapped_column(default=None)
    effective_date: Mapped[datetime | None] = mapped_column(default=None)
    checksum: Mapped[str]
    license: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="active")  # "active" | "superseded" | "stale" | "restricted"
    # Real self-referential FK (not a bare int): matches
    # Department.parent_department_id's precedent elsewhere in this codebase
    # for a nullable self-reference to the same table.
    supersedes_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("document.id"), default=None
    )
    # Native Postgres ARRAY, not this codebase's existing "_json"-suffixed
    # opaque-string-blob convention (AuditLog.metadata_json, Span.detail_json):
    # access scope tags are a queryable filter predicate for Phase 7 Task 5's
    # authz-aware retrieval ("baked into the SQL query itself, not applied
    # post-hoc" per docs/superpowers/plans/2026-08-27-phase7-knowledge-retrieval.md),
    # not an opaque blob -- they need a real array column so a future filter
    # can use `@>`/`ANY` (and a GIN index), which a serialized string cannot.
    access_scope_tags: Mapped[list[str]] = mapped_column(ARRAY(String()), default=list)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    supersedes: Mapped["Document | None"] = relationship(remote_side=[id])


class DocumentVersion(Base):
    """One parsed/chunked snapshot of a Document.

    A Document can be re-ingested (new parser/chunking version, or updated
    source content) without losing the history of prior chunk sets --
    Chunk rows point at the DocumentVersion that produced them, not
    directly at the Document.
    """

    __tablename__ = "document_version"
    __table_args__ = (Index("ix_document_version_document_id", "document_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    parser_version: Mapped[str]
    chunking_version: Mapped[str]
    content_hash: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Chunk(Base):
    """One retrieval unit produced from a DocumentVersion."""

    __tablename__ = "chunk"
    __table_args__ = (Index("ix_chunk_document_version_id", "document_version_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_version_id: Mapped[int] = mapped_column(ForeignKey("document_version.id"))
    ordinal: Mapped[int]
    text: Mapped[str]
    token_count: Mapped[int]
    # Real self-referential FK (nullable): reserved for future parent-child
    # retrieval (plan.md's RAG design section lists "parent-child retrieval"
    # as a planned experiment) -- not populated or read by this task.
    parent_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("chunk.id"), default=None)
    # Following this codebase's existing "_json"-suffixed opaque-string-blob
    # convention (AuditLog.before_json/after_json/metadata_json, Span.detail_json):
    # this holds a small JSON-serialized snapshot of the parent Document's
    # effective_date/access_scope_tags *as of chunk-creation time*, so a chunk's
    # authz/freshness context survives independent of later Document edits. A
    # future task may promote the access-scope portion to a real queryable
    # column (mirroring Document.access_scope_tags) if Task 5's authz filter
    # needs it baked directly into the chunk query rather than read after the
    # fact -- out of scope for this schema-only task.
    metadata_json: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    parent_chunk: Mapped["Chunk | None"] = relationship(remote_side=[id])


class Embedding(Base):
    """One vector embedding of a Chunk, tagged with the model/version that produced it."""

    __tablename__ = "embedding"
    __table_args__ = (Index("ix_embedding_chunk_id", "chunk_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunk.id"))
    embedding_model: Mapped[str]
    embedding_version: Mapped[str]
    vector: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class IngestionRun(Base):
    """One ingestion pipeline execution -- records parser/chunk/embed versions and stats.

    No FK to Document/DocumentVersion: a single run processes many documents,
    matching ModelCall's precedent of a run/call-level summary row that
    doesn't need a real per-item FK to be useful (documents_processed/
    chunks_created are aggregate counts, not a joinable list).
    """

    __tablename__ = "ingestion_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    parser_version: Mapped[str]
    chunking_version: Mapped[str]
    embedding_version: Mapped[str]
    documents_processed: Mapped[int] = mapped_column(default=0)
    chunks_created: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="running")  # "running" | "completed" | "error"
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
