"""Tests for `resolvegrid_api.ingestion` (Phase 7 Task 6).

`test_ingest_document_idempotent_reingestion_does_not_duplicate_chunks`
and `test_end_to_end_seed_corpus_document_is_retrievable_with_scoped_authz`
are deliberate no-mocking integration tests: real `chunk_markdown`, real
`embed_texts` against a live Ollama `nomic-embed-text`, and real Postgres
writes/queries -- matching `test_knowledge_store.py`/`test_retrieval.py`'s
precedent. Requires a running `resolvegrid-ollama` container with
`nomic-embed-text` pulled and a running `resolvegrid-postgres` container
with migrations applied.
"""

from sqlalchemy import func, select

from resolvegrid_api.ingestion import IngestionError, ingest_document
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding
from resolvegrid_api.retrieval import lexical_search, vector_search
from resolvegrid_api.retrieval_authz import AuthzFilter
from resolvegrid_api.seed_corpus import load_seed_corpus

PARSER_VERSION = "test-ingestion-parser-v1"
CHUNKING_VERSION = "test-ingestion-chunk-v1"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_VERSION = "test-v1"


def _chunk_and_embedding_counts(session, document_version_id: int) -> tuple[int, int]:
    chunk_count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_version_id == document_version_id)
    )
    embedding_count = session.scalar(
        select(func.count())
        .select_from(Embedding)
        .join(Chunk, Chunk.id == Embedding.chunk_id)
        .where(Chunk.document_version_id == document_version_id)
    )
    return chunk_count, embedding_count


# ---------------------------------------------------------------------------
# Untagged synthetic_private / tagged public validation (the critical
# leak-prevention check from Task 5's review, per this task's instructions)
# ---------------------------------------------------------------------------


def test_ingest_document_rejects_untagged_synthetic_private_document(db_session):
    try:
        ingest_document(
            db_session,
            title="Test Untagged Private Doc",
            source_type="synthetic_private",
            raw_markdown="# Secret\n\nThis should never be written.",
            access_scope_tags=[],
            parser_version=PARSER_VERSION,
            chunking_version=CHUNKING_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_version=EMBEDDING_VERSION,
        )
        assert False, "expected IngestionError for untagged synthetic_private document"
    except IngestionError as exc:
        assert "synthetic_private" in str(exc)
        assert "access_scope_tags" in str(exc)

    # Nothing must have been written -- not even a Document row -- for the
    # refused document.
    count = db_session.scalar(
        select(func.count()).select_from(Document).where(Document.title == "Test Untagged Private Doc")
    )
    assert count == 0


def test_ingest_document_rejects_untagged_synthetic_private_document_with_none_tags(db_session):
    """access_scope_tags=None (the default) must be treated the same as
    an empty list -- not silently bypass the check.
    """
    try:
        ingest_document(
            db_session,
            title="Test Untagged Private Doc (None tags)",
            source_type="synthetic_private",
            raw_markdown="# Secret\n\nThis should never be written.",
            parser_version=PARSER_VERSION,
            chunking_version=CHUNKING_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_version=EMBEDDING_VERSION,
        )
        assert False, "expected IngestionError"
    except IngestionError:
        pass


def test_ingest_document_rejects_tagged_public_document(db_session):
    try:
        ingest_document(
            db_session,
            title="Test Tagged Public Doc",
            source_type="public",
            raw_markdown="# Public\n\nThis should never be written either.",
            access_scope_tags=["security"],
            parser_version=PARSER_VERSION,
            chunking_version=CHUNKING_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_version=EMBEDDING_VERSION,
        )
        assert False, "expected IngestionError for tagged public document"
    except IngestionError as exc:
        assert "public" in str(exc)


def test_ingest_document_rejects_unknown_source_type(db_session):
    try:
        ingest_document(
            db_session,
            title="Test Unknown Source Type Doc",
            source_type="totally_made_up",
            raw_markdown="# Doc",
            parser_version=PARSER_VERSION,
            chunking_version=CHUNKING_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_version=EMBEDDING_VERSION,
        )
        assert False, "expected IngestionError"
    except IngestionError:
        pass


def test_ingest_document_rejects_reclassification_via_title_collision(db_session):
    """Regression test for a real bypass found in review: ingesting a
    genuinely different, differently-classified document under a title
    that already exists must be refused, not silently absorbed into the
    existing Document's source_type/access_scope_tags. Without this check,
    ingesting a synthetic_private+tagged document under a title that
    already exists as public+untagged silently kept the OLD (public,
    untagged) classification -- exactly the leak
    `test_ingest_document_rejects_untagged_synthetic_private_document`
    exists to prevent, reachable through a second path.
    """
    title = "Test Title Collision Doc"

    ingest_document(
        db_session,
        title=title,
        source_type="public",
        raw_markdown="# Public\n\nOriginal public content under this title.",
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )

    try:
        ingest_document(
            db_session,
            title=title,
            source_type="synthetic_private",
            raw_markdown="# Private\n\nGenuinely different, confidential content.",
            access_scope_tags=["security"],
            parser_version=PARSER_VERSION,
            chunking_version=CHUNKING_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_version=EMBEDDING_VERSION,
        )
        assert False, "expected IngestionError for a reclassified title collision"
    except IngestionError as exc:
        assert "already exists" in str(exc)

    # The original document's classification must be untouched by the
    # rejected call -- still public/untagged, not silently mutated either.
    document = db_session.scalars(select(Document).where(Document.title == title)).first()
    assert document.source_type == "public"
    assert document.access_scope_tags == []


# ---------------------------------------------------------------------------
# Idempotent re-ingestion
# ---------------------------------------------------------------------------


def test_ingest_document_idempotent_reingestion_does_not_duplicate_chunks(db_session):
    raw_markdown = (
        "# Test Idempotency Doc\n\n"
        "## Section One\n\n"
        "This is the first section's content, written for the idempotency test.\n\n"
        "## Section Two\n\n"
        "This is the second section's content, distinct from the first."
    )

    first_version = ingest_document(
        db_session,
        title="Test Idempotency Doc",
        source_type="synthetic_private",
        raw_markdown=raw_markdown,
        access_scope_tags=["security"],
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )
    db_session.flush()

    document_count_after_first = db_session.scalar(
        select(func.count()).select_from(Document).where(Document.title == "Test Idempotency Doc")
    )
    version_count_after_first = db_session.scalar(
        select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == first_version.document_id)
    )
    chunks_after_first, embeddings_after_first = _chunk_and_embedding_counts(db_session, first_version.id)

    assert document_count_after_first == 1
    assert version_count_after_first == 1
    assert chunks_after_first > 0
    assert embeddings_after_first == chunks_after_first

    # Re-run with byte-identical content -- must be a no-op: same
    # DocumentVersion returned, no new Document/DocumentVersion/Chunk/
    # Embedding rows.
    second_version = ingest_document(
        db_session,
        title="Test Idempotency Doc",
        source_type="synthetic_private",
        raw_markdown=raw_markdown,
        access_scope_tags=["security"],
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )
    db_session.flush()

    assert second_version.id == first_version.id

    document_count_after_second = db_session.scalar(
        select(func.count()).select_from(Document).where(Document.title == "Test Idempotency Doc")
    )
    version_count_after_second = db_session.scalar(
        select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == first_version.document_id)
    )
    chunks_after_second, embeddings_after_second = _chunk_and_embedding_counts(db_session, first_version.id)

    assert document_count_after_second == 1
    assert version_count_after_second == 1
    assert chunks_after_second == chunks_after_first
    assert embeddings_after_second == embeddings_after_first


def test_ingest_document_changed_content_creates_new_version_not_a_duplicate(db_session):
    """Distinguishes the idempotent-skip path from real re-ingestion:
    different content under the same title must create a *second*
    DocumentVersion (new chunks/embeddings for the new content), not be
    treated as identical, and must not touch the first version's rows.
    """
    title = "Test Changed Content Doc"

    first_version = ingest_document(
        db_session,
        title=title,
        source_type="synthetic_private",
        raw_markdown="# Test Changed Content Doc\n\nOriginal content, version one.",
        access_scope_tags=["security"],
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )
    db_session.flush()
    first_chunks, _ = _chunk_and_embedding_counts(db_session, first_version.id)

    second_version = ingest_document(
        db_session,
        title=title,
        source_type="synthetic_private",
        raw_markdown="# Test Changed Content Doc\n\nUpdated content, version two, materially different text.",
        access_scope_tags=["security"],
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )
    db_session.flush()

    assert second_version.id != first_version.id
    assert second_version.document_id == first_version.document_id

    version_count = db_session.scalar(
        select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == first_version.document_id)
    )
    assert version_count == 2

    # First version's own chunks are untouched.
    first_chunks_after, _ = _chunk_and_embedding_counts(db_session, first_version.id)
    assert first_chunks_after == first_chunks
    second_chunks, second_embeddings = _chunk_and_embedding_counts(db_session, second_version.id)
    assert second_chunks > 0
    assert second_embeddings == second_chunks


# ---------------------------------------------------------------------------
# End-to-end: a real seed corpus document, ingested through the full
# pipeline, is retrievable under a matching AuthzFilter and NOT retrievable
# under a disjoint one -- the Task 5 adversarial pattern, now proven
# against a genuinely ingested document instead of a hand-constructed one.
# ---------------------------------------------------------------------------


def test_end_to_end_seed_corpus_document_is_retrievable_with_scoped_authz(db_session):
    seed_docs = {doc.title: doc for doc in load_seed_corpus()}
    vpn_policy = seed_docs["Kestrel VPN Access Policy (v2)"]
    assert vpn_policy.access_scope_tags == ["security"]

    version = ingest_document(
        db_session,
        title="Test E2E -- " + vpn_policy.title,
        source_type=vpn_policy.source_type,
        raw_markdown=vpn_policy.raw_markdown,
        access_scope_tags=vpn_policy.access_scope_tags,
        status=vpn_policy.status,
        effective_date=vpn_policy.effective_date,
        parser_version=PARSER_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_version=EMBEDDING_VERSION,
    )
    db_session.flush()

    chunk_ids = set(
        db_session.scalars(select(Chunk.id).where(Chunk.document_version_id == version.id)).all()
    )
    assert len(chunk_ids) > 0

    query_text = "VPN password reset"
    # A real embedding of the query text, so vector_search exercises the
    # genuine embedding space this document was ingested into (not a
    # hand-constructed vector).
    from resolvegrid_retrieval.embedder import embed_texts

    query_vector = embed_texts([query_text], model=EMBEDDING_MODEL)[0]

    matching_filter = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"security"}))
    disjoint_filter = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"people_ops"}))

    vector_matching = vector_search(db_session, query_vector, limit=10, authz_filter=matching_filter)
    lexical_matching = lexical_search(db_session, query_text, limit=10, authz_filter=matching_filter)

    vector_matching_ids = {chunk_id for chunk_id, _ in vector_matching}
    lexical_matching_ids = {chunk_id for chunk_id, _ in lexical_matching}

    # At least one of this document's chunks must actually come back under
    # the matching filter for both signals -- proving real retrievability,
    # not just "the query didn't error."
    assert vector_matching_ids & chunk_ids
    assert lexical_matching_ids & chunk_ids

    vector_disjoint = vector_search(db_session, query_vector, limit=50, authz_filter=disjoint_filter)
    lexical_disjoint = lexical_search(db_session, query_text, limit=50, authz_filter=disjoint_filter)

    vector_disjoint_ids = {chunk_id for chunk_id, _ in vector_disjoint}
    lexical_disjoint_ids = {chunk_id for chunk_id, _ in lexical_disjoint}

    # None of this document's chunks may appear under the disjoint filter
    # -- literally absent, not merely ranked lower.
    assert not (vector_disjoint_ids & chunk_ids)
    assert not (lexical_disjoint_ids & chunk_ids)
