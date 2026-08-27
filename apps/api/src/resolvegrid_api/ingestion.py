"""Ingestion pipeline: turns raw Markdown source content into persisted
`Document` / `DocumentVersion` / `Chunk` / `Embedding` rows (Phase 7 Task
6). This is the single place that wires together `chunk_markdown` (Task
2), `embed_texts` (Task 3), and `store_chunks_with_embeddings` (Task 3),
plus the `Document`/`DocumentVersion` bookkeeping none of those lower
layers own.

Untagged-synthetic_private safety check (CRITICAL, per Task 5's review and
`docs/DECISION_LOG.md`'s 2026-08-27 "empty access_scope_tags = public"
entry)
--------------------------------------------------------------------------
`resolvegrid_api.retrieval_authz`/`resolvegrid_api.retrieval` treat a
`Document` with an **empty** `access_scope_tags` array as public --
visible to every principal, regardless of `AuthzFilter`. That convention
is correct and intentional for real `source_type="public"` documents, but
it means an ingestion bug that writes a `source_type="synthetic_private"`
document with empty tags is not a validation nuisance -- it is a silent
real security leak (a private document becomes readable by anyone). This
module refuses to write such a document at all: `ingest_document` raises
`IngestionError` before touching the database if
`source_type == "synthetic_private"` and `access_scope_tags` is empty.
The symmetric case is also rejected (a `source_type="public"` document
must not carry access_scope_tags -- a non-empty set would incorrectly
*restrict* a document meant to be visible to everyone), since that's the
same class of metadata/authz mismatch in the other direction.

Idempotent re-ingestion mechanism (documented, per this task's
requirement to decide and document the exact approach)
--------------------------------------------------------------------------
`Document` has no separate slug/external-id column, so this pipeline uses
`Document.title` as the natural key identifying "the same logical
document" across ingestion runs (documented here since nothing on the
model itself pins this down -- see `models/knowledge.py`). On each call:

1. Look up an existing `Document` by `title`.
2. If none exists, create a new `Document` (this call's `source_type`,
   tags, attribution fields, etc. all apply) plus a `DocumentVersion` with
   `content_hash = sha256(raw_markdown)`, then chunk + embed + store as
   normal.
3. If a `Document` with that title already exists, first re-validate this
   call's `source_type`/`access_scope_tags` against what's already
   persisted on it -- a mismatch raises `IngestionError` rather than
   silently reusing the existing (possibly differently-classified)
   `Document` row. Found by review: without this check, a title collision
   between two genuinely different documents (e.g. one public+untagged,
   one synthetic_private+tagged) would silently discard whichever call's
   classification lost the race to create the `Document` row first --
   exactly the untagged-private-document leak this module exists to
   prevent, reachable through re-ingestion rather than fresh creation.
   Then, check for an existing `DocumentVersion` on it with the *same*
   `content_hash`. If found, this is an idempotent no-op re-run of
   identical source content: return that existing `DocumentVersion`
   immediately, without calling the chunker, the embedder, or
   `store_chunks_with_embeddings` again -- so re-running ingestion over
   unchanged source content never creates duplicate `Chunk`/`Embedding`
   rows.
4. If a `Document` with that title exists but `content_hash` differs
   (the source content actually changed), a *new* `DocumentVersion` is
   created under the *existing* `Document` and re-chunked/re-embedded --
   this is deliberate re-ingestion of updated content, not a duplicate,
   matching `DocumentVersion`'s own docstring ("Chunk rows point at the
   DocumentVersion that produced them, not directly at the Document").
   The existing `Document`'s own metadata (source_type, tags, attribution)
   is left untouched in this case -- only its first ingestion sets those
   fields; this function does not update `Document`-level metadata on a
   content-changed re-ingestion, which is a deliberate scope limit, not
   an oversight.
"""

import hashlib
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.knowledge_store import store_chunks_with_embeddings
from resolvegrid_api.models.knowledge import Document, DocumentVersion
from resolvegrid_retrieval.chunker import chunk_markdown
from resolvegrid_retrieval.embedder import embed_texts

_VALID_SOURCE_TYPES = ("public", "synthetic_private")


class IngestionError(ValueError):
    """Raised when `ingest_document`'s input violates a safety invariant
    (an untagged `synthetic_private` document, a tagged `public` document,
    or an unrecognized `source_type`) -- callers should only ever need to
    catch this one exception type from this module, not a raw
    `ValueError`/`AssertionError`, mirroring `EmbeddingError`'s convention
    in `resolvegrid_retrieval.embedder`.
    """


def _content_hash(raw_markdown: str) -> str:
    return hashlib.sha256(raw_markdown.encode("utf-8")).hexdigest()


def ingest_document(
    session: Session,
    *,
    title: str,
    source_type: str,
    raw_markdown: str,
    parser_version: str,
    chunking_version: str,
    embedding_model: str,
    embedding_version: str,
    access_scope_tags: list[str] | None = None,
    url: str | None = None,
    publisher: str | None = None,
    retrieved_at: datetime | None = None,
    effective_date: datetime | None = None,
    license: str | None = None,
    status: str = "active",
    supersedes_document_id: int | None = None,
) -> DocumentVersion:
    """Ingest one document's raw Markdown content, end to end.

    Raises `IngestionError` (before any database write) if:
      - `source_type` isn't `"public"` or `"synthetic_private"`;
      - `source_type == "synthetic_private"` and `access_scope_tags` is
        empty (see module docstring -- this is the critical leak-
        prevention check);
      - `source_type == "public"` and `access_scope_tags` is non-empty
        (the symmetric metadata/authz mismatch).

    Returns the resulting `DocumentVersion` -- either newly created (fresh
    document, or updated content on an existing one), or the pre-existing
    one from an earlier identical-content run (see module docstring's
    idempotency section). Does not commit -- callers control the
    transaction boundary, matching `store_chunks_with_embeddings`'s and
    this codebase's `db_session`/`raw_db_session` fixture split.
    """
    if source_type not in _VALID_SOURCE_TYPES:
        raise IngestionError(
            f"source_type must be one of {_VALID_SOURCE_TYPES}, got {source_type!r}"
        )

    tags = list(access_scope_tags) if access_scope_tags else []

    if source_type == "synthetic_private" and not tags:
        raise IngestionError(
            f"refusing to ingest synthetic_private document {title!r} with empty "
            "access_scope_tags -- an empty access_scope_tags array is treated as "
            "PUBLIC (visible to every principal) by resolvegrid_api.retrieval's "
            "authz-aware queries (see retrieval_authz.py's module docstring), so "
            "ingesting an untagged synthetic_private document would silently leak "
            "it to everyone. Pass a non-empty access_scope_tags (e.g. a "
            "normalize_department_tag()-normalized department name)."
        )

    if source_type == "public" and tags:
        raise IngestionError(
            f"refusing to ingest public document {title!r} with non-empty "
            f"access_scope_tags {tags!r} -- a public document must carry no "
            "access_scope_tags (empty means visible to everyone); a non-empty "
            "set would incorrectly restrict a document meant to be public."
        )

    content_hash = _content_hash(raw_markdown)

    existing_document = session.scalars(
        select(Document).where(Document.title == title)
    ).first()

    if existing_document is not None:
        # Re-validate this call's classification against what's already
        # persisted under this title (found by review: without this check,
        # ingesting a *different*, e.g. synthetic_private+tagged document
        # under a title that already exists as public+untagged would
        # silently keep the OLD document's source_type/access_scope_tags --
        # discarding this call's explicit classification rather than
        # rejecting the collision. Title is this pipeline's only natural
        # key, so a mismatch here means "this looks like a different
        # document reusing an existing title," which must be refused, not
        # silently absorbed into the existing Document's authz metadata).
        if existing_document.source_type != source_type or set(
            existing_document.access_scope_tags
        ) != set(tags):
            raise IngestionError(
                f"document titled {title!r} already exists with "
                f"source_type={existing_document.source_type!r}, "
                f"access_scope_tags={existing_document.access_scope_tags!r}, "
                f"but this call requested source_type={source_type!r}, "
                f"access_scope_tags={tags!r} -- refusing to ingest under the "
                "same title with a different classification (this would "
                "either silently discard the new classification or, if "
                "reversed, silently change an existing document's authz "
                "scope). If this is genuinely the same document being "
                "reclassified, update it explicitly rather than "
                "re-ingesting; if it's a different document, use a "
                "distinct title."
            )
        existing_version = session.scalars(
            select(DocumentVersion).where(
                DocumentVersion.document_id == existing_document.id,
                DocumentVersion.content_hash == content_hash,
            )
        ).first()
        if existing_version is not None:
            # Idempotent no-op: identical content already ingested under
            # this DocumentVersion -- do not re-chunk/re-embed/re-store.
            return existing_version
        document = existing_document
    else:
        document = Document(
            source_type=source_type,
            title=title,
            url=url,
            publisher=publisher,
            retrieved_at=retrieved_at,
            effective_date=effective_date,
            checksum=content_hash,
            license=license,
            status=status,
            supersedes_document_id=supersedes_document_id,
            access_scope_tags=tags,
        )
        session.add(document)
        session.flush()

    version = DocumentVersion(
        document_id=document.id,
        parser_version=parser_version,
        chunking_version=chunking_version,
        content_hash=content_hash,
    )
    session.add(version)
    session.flush()

    chunk_records = chunk_markdown(raw_markdown, parser_version, chunking_version)
    texts = [record.text for record in chunk_records]
    vectors = embed_texts(texts, model=embedding_model)
    store_chunks_with_embeddings(
        session,
        document_version_id=version.id,
        chunk_records=chunk_records,
        vectors=vectors,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )

    return version
