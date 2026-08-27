"""Integration tests for `resolvegrid_api.knowledge_store` (Phase 7 Task 3).

`test_cosine_similarity_ranks_semantically_similar_chunks_closer` is a
deliberate no-mocking smoke test: it calls the real Ollama
`nomic-embed-text` embeddings endpoint (via `resolvegrid_retrieval.embedder`)
and queries real pgvector cosine distance, to prove the full
chunk -> embed -> store -> similarity-query path actually round-trips
correct semantic behavior -- not just that the plumbing doesn't crash.
Requires a running `resolvegrid-ollama` container with `nomic-embed-text`
pulled (`docker exec resolvegrid-ollama ollama pull nomic-embed-text`) and
a running `resolvegrid-postgres` container with migration `0010` applied.
"""

from sqlalchemy import text

from resolvegrid_api.knowledge_store import store_chunks_with_embeddings
from resolvegrid_api.models.knowledge import Document, DocumentVersion
from resolvegrid_retrieval.chunker import ChunkRecord
from resolvegrid_retrieval.embedder import embed_texts


def _make_document_version(session) -> int:
    document = Document(
        source_type="synthetic_private",
        title="Test Document",
        checksum="test-checksum",
    )
    session.add(document)
    session.flush()

    version = DocumentVersion(
        document_id=document.id,
        parser_version="test-parser-v1",
        chunking_version="test-chunk-v1",
        content_hash="test-content-hash",
    )
    session.add(version)
    session.flush()
    return version.id


def test_store_chunks_with_embeddings_writes_aligned_chunk_and_embedding_rows(db_session):
    document_version_id = _make_document_version(db_session)
    chunk_records = [
        ChunkRecord(ordinal=0, text="first chunk text", token_count=3),
        ChunkRecord(ordinal=1, text="second chunk text", token_count=3),
    ]
    vectors = [[0.1] * 768, [0.2] * 768]

    pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=document_version_id,
        chunk_records=chunk_records,
        vectors=vectors,
        embedding_model="nomic-embed-text",
        embedding_version="v1",
    )

    assert len(pairs) == 2
    for (chunk, embedding), record, vector in zip(pairs, chunk_records, vectors):
        assert chunk.id is not None
        assert chunk.document_version_id == document_version_id
        assert chunk.ordinal == record.ordinal
        assert chunk.text == record.text
        assert embedding.id is not None
        assert embedding.chunk_id == chunk.id
        assert embedding.embedding_model == "nomic-embed-text"
        assert list(embedding.vector) == vector


def test_store_chunks_with_embeddings_rejects_mismatched_lengths(db_session):
    document_version_id = _make_document_version(db_session)

    try:
        store_chunks_with_embeddings(
            db_session,
            document_version_id=document_version_id,
            chunk_records=[ChunkRecord(ordinal=0, text="only one chunk", token_count=3)],
            vectors=[[0.1] * 768, [0.2] * 768],
            embedding_model="nomic-embed-text",
            embedding_version="v1",
        )
        assert False, "expected ValueError for mismatched lengths"
    except ValueError:
        pass


def test_cosine_similarity_ranks_semantically_similar_chunks_closer(db_session):
    document_version_id = _make_document_version(db_session)

    texts = [
        "To reset your VPN password, open the VPN client and click Forgot Password.",
        "If you forgot your VPN password, use the VPN client's password reset link.",
        "The office coffee machine on the third floor is out of order until Friday.",
    ]
    vectors = embed_texts(texts)
    assert len(vectors) == 3
    assert all(len(v) == 768 for v in vectors)

    chunk_records = [
        ChunkRecord(ordinal=i, text=t, token_count=len(t.split())) for i, t in enumerate(texts)
    ]
    pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=document_version_id,
        chunk_records=chunk_records,
        vectors=vectors,
        embedding_model="nomic-embed-text",
        embedding_version="v1",
    )
    db_session.flush()

    vpn_a_id = pairs[0][1].id
    vpn_b_id = pairs[1][1].id
    coffee_id = pairs[2][1].id

    def _cosine_distance(a_id: int, b_id: int) -> float:
        row = db_session.execute(
            text(
                "SELECT a.vector <=> b.vector FROM embedding a, embedding b "
                "WHERE a.id = :a_id AND b.id = :b_id"
            ),
            {"a_id": a_id, "b_id": b_id},
        ).first()
        return row[0]

    vpn_to_vpn = _cosine_distance(vpn_a_id, vpn_b_id)
    vpn_a_to_coffee = _cosine_distance(vpn_a_id, coffee_id)
    vpn_b_to_coffee = _cosine_distance(vpn_b_id, coffee_id)

    # Real measured values are asserted in the test report, not just the
    # pass/fail -- see the task's verification requirements.
    print(
        f"cosine distances -- vpn<->vpn: {vpn_to_vpn:.4f}, "
        f"vpn_a<->coffee: {vpn_a_to_coffee:.4f}, vpn_b<->coffee: {vpn_b_to_coffee:.4f}"
    )

    assert vpn_to_vpn < vpn_a_to_coffee
    assert vpn_to_vpn < vpn_b_to_coffee
