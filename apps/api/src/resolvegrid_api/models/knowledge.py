from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
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
    # Postgres full-text-search column (Phase 7 Task 4, migration 0011):
    # a stored generated tsvector, kept in sync with `text` by Postgres
    # itself (not application code), backed by a GIN index
    # (`ix_chunk_search_vector`) so `resolvegrid_api.retrieval.lexical_search`'s
    # `ts_rank_cd` queries are index-backed rather than a per-query
    # sequential `to_tsvector` recompute. See migration 0011's docstring
    # for the stored-vs-inline reasoning. Never written to directly by
    # the ORM -- Postgres computes it -- so it's not passed in any
    # `Chunk(...)` constructor call in this codebase.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

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


class SemanticResponseCache(Base):
    """One semantic (embedding-similarity) response-cache entry (Phase 12
    Task 5, `resolvegrid_api.semantic_cache`) -- the sibling of Task 3's
    Redis-backed EXACT-match response cache (`resolvegrid_api.response_
    cache`), but keyed by cosine-similarity against a real `nomic-embed
    -text` query embedding rather than a SHA-256 hash of the exact message
    text, so a differently-worded-but-semantically-equivalent question can
    still hit.

    Stored in Postgres (not Redis): pgvector is this codebase's established
    vector-search mechanism (`Embedding` above, `retrieval.py`'s `vector_
    search`), and a real `<=>` cosine-distance ORDER BY against a handful of
    rows per `(scope_sig, kb_version)` is simple and fast at this phase's
    real traffic scale -- introducing vector search inside Redis as well
    would be a second, parallel mechanism for no real benefit (see the plan
    doc's Task 5 section, which explicitly prefers extending pgvector over
    that alternative).

    Cached VALUE shape mirrors `response_cache.py`'s Redis value exactly
    (`answer`/`citations`/`caption`/`intent`/`risk_level`) -- stored inline
    as real columns here (`citations` JSON-serialized into `citations_json`,
    following `Chunk.metadata_json`'s established "_json"-suffixed opaque
    -blob convention for a small structured extra) rather than a separate
    FK'd "response" table, since there is no other real row this entry
    would ever need to join against -- an entire independent response table
    would just be this same handful of columns one join away for no benefit.

    `(scope_sig, kb_version)` -- the SAME two real values Task 2's retrieval
    cache and Task 3's response cache key on, reused unchanged (via `compute_
    scope_sig`/`get_kb_version`, never reimplemented) -- are real, queryable
    columns (not folded into an opaque composite string) with a real
    composite index (`ix_semantic_response_cache_scope_kb`) so a lookup's
    `WHERE scope_sig = :scope_sig AND kb_version = :kb_version` is index
    -backed, and so a similarity search's candidate ROWS are restricted to
    the same authz scope and KB version before any distance is ever
    computed -- this is the security-critical boundary (see `semantic_
    cache.py`'s module docstring): two different `scope_sig` values must
    NEVER semantically match each other's cached answers, no matter how
    similar the query text, exactly mirroring Task 2's `scope_sig` boundary.

    No TTL column: unlike Task 1-3's Redis caches (where `EX` is a native
    Redis mechanism), a Postgres row has no built-in expiry -- `semantic_
    cache.py`'s lookup query instead filters `created_at > :cutoff` using
    the same entitlement-staleness-bound reasoning Task 3's `RESPONSE_CACHE_
    TTL_SECONDS` documents (this cache carries the identical content
    -staleness and entitlement-staleness risk one layer up), so an old row
    is simply never looked up again past that bound -- it is not deleted (a
    future cleanup pass could prune `created_at`-expired rows; out of scope
    here, matching the Redis caches' own "stale keys expire via TTL, no
    explicit delete pass" documented strategy).

    Deliberately no ANN (HNSW) index on `query_embedding`: this table's real
    candidate set per `(scope_sig, kb_version)` is small (one organization's
    worth of distinct recent questions within one scope), so a plain
    sequential scan over the already-filtered candidate rows, exactly
    ordered by real `<=>` cosine distance, is both fast enough and exactly
    correct -- an ANN index trades a small, real chance of missing the true
    nearest neighbor for speed at a scale this table doesn't operate at, and
    this is a security/correctness-adjacent lookup (the whole point is a
    precise, honest threshold decision), so exact ordering is the right
    tradeoff here, unlike `Embedding.vector`'s HNSW index (Phase 7), whose
    multi-chunk corpus-wide search genuinely benefits from ANN speed and
    whose stakes are "which chunk ranks higher", not "does this query match
    at all".
    """

    __tablename__ = "semantic_response_cache"
    __table_args__ = (
        Index("ix_semantic_response_cache_scope_kb", "scope_sig", "kb_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_sig: Mapped[str]
    kb_version: Mapped[int]
    query_embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    answer: Mapped[str]
    citations_json: Mapped[str]
    caption: Mapped[str]
    intent: Mapped[str]
    risk_level: Mapped[str]
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
