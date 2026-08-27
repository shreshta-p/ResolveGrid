"""Tests for `resolvegrid_api.retrieval` (Phase 7 Task 4: lexical search +
RRF fusion).

`test_lexical_search_*` and `test_vector_search_*` are real, no-mocking
integration tests against the live `resolvegrid-postgres` container
(matching `test_knowledge_store.py`'s precedent) -- they exercise the
actual `ts_rank_cd`/`websearch_to_tsquery` and pgvector `<=>` SQL, not a
mocked DB.

`vector_search`'s test uses hand-constructed vectors rather than a real
`embed_texts()` call to Ollama: the thing under test is the SQL query's
cosine-distance ordering, which is fully determined by the vector values
themselves -- hand-constructed vectors with known, exact geometric
relationships (identical / near-identical / orthogonal) prove the query's
correctness precisely and deterministically, without a live-Ollama
dependency or non-deterministic embedding output. (`test_knowledge_store.py`
already covers the real-embedder round trip separately.)

`test_fuse_rrf_matches_hand_computed_scores` is a pure unit test (no DB) --
the expected scores are computed by hand in the test itself (as exact
fractions, then compared with `pytest.approx`), per this task's
requirement that a future implementation bug in the RRF formula would be
caught precisely, not just by an ordering fuzzy-match.
"""

import pytest

from resolvegrid_api.knowledge_store import store_chunks_with_embeddings
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion
from resolvegrid_api.retrieval import fuse_rrf, lexical_search, vector_search
from resolvegrid_retrieval.chunker import ChunkRecord

EMBEDDING_DIM = 768


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


def _make_chunk(session, document_version_id: int, *, ordinal: int, text: str) -> int:
    chunk = Chunk(
        document_version_id=document_version_id,
        ordinal=ordinal,
        text=text,
        token_count=len(text.split()),
    )
    session.add(chunk)
    session.flush()
    return chunk.id


def _basis_vector(dim: int, index: float, magnitude: float = 1.0) -> list[float]:
    """A mostly-zero vector with `magnitude` at position `index` (allows a
    fractional `index` only in the sense that callers pass an int -- kept
    as a plain helper for the hand-constructed vectors below).
    """
    vector = [0.0] * dim
    vector[index] = magnitude
    return vector


# ---------------------------------------------------------------------------
# lexical_search
# ---------------------------------------------------------------------------


def test_lexical_search_ranks_term_dense_chunk_highest(db_session):
    document_version_id = _make_document_version(db_session)

    # Verified directly against the live Postgres container before writing
    # this test (docker exec ... psql ... ts_rank_cd(...)): with the query
    # "VPN password reset", chunk_dense ranks ~0.2083, chunk_sparse ranks
    # ~0.025, and chunk_irrelevant does not match at all (excluded by the
    # `@@` predicate, not just ranked last).
    chunk_dense_id = _make_chunk(
        db_session,
        document_version_id,
        ordinal=0,
        text=(
            "VPN password reset instructions: to reset your VPN password, "
            "use the VPN client's password reset link."
        ),
    )
    chunk_sparse_id = _make_chunk(
        db_session,
        document_version_id,
        ordinal=1,
        text=(
            "For general account issues, contact IT support. VPN access "
            "requires a password reset request via the help desk."
        ),
    )
    _make_chunk(
        db_session,
        document_version_id,
        ordinal=2,
        text="The office coffee machine on the third floor is out of order until Friday.",
    )

    results = lexical_search(db_session, "VPN password reset", limit=10)

    assert results == [(chunk_dense_id, 1), (chunk_sparse_id, 2)]


def test_lexical_search_respects_limit(db_session):
    document_version_id = _make_document_version(db_session)
    chunk_ids = {
        _make_chunk(
            db_session,
            document_version_id,
            ordinal=i,
            text=f"VPN password reset chunk number {i}.",
        )
        for i in range(5)
    }

    results = lexical_search(db_session, "VPN password reset", limit=2)

    # All 5 chunks match equally well (same term set) -- limit=2 must
    # still cap the result count and rank positions, even though which 2
    # of the 5 equally-ranked chunks come back is not asserted here.
    assert len(results) == 2
    assert {chunk_id for chunk_id, _ in results} <= chunk_ids
    assert [rank for _, rank in results] == [1, 2]


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------


def test_vector_search_ranks_nearest_vectors_first(db_session):
    document_version_id = _make_document_version(db_session)

    query_vector = _basis_vector(EMBEDDING_DIM, 0, 1.0)
    # near_vector points almost the same direction as query_vector.
    near_vector = _basis_vector(EMBEDDING_DIM, 0, 0.9)
    near_vector[1] = 0.1
    # far_vector is orthogonal to query_vector (cosine distance 1.0).
    far_vector = _basis_vector(EMBEDDING_DIM, 1, 1.0)

    chunk_records = [
        ChunkRecord(ordinal=0, text="near chunk", token_count=2),
        ChunkRecord(ordinal=1, text="far chunk", token_count=2),
    ]
    pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=document_version_id,
        chunk_records=chunk_records,
        vectors=[near_vector, far_vector],
        embedding_model="test-model",
        embedding_version="v1",
    )
    near_chunk_id = pairs[0][0].id
    far_chunk_id = pairs[1][0].id

    results = vector_search(db_session, query_vector, limit=2)

    assert results == [(near_chunk_id, 1), (far_chunk_id, 2)]


def test_vector_search_respects_limit(db_session):
    document_version_id = _make_document_version(db_session)
    query_vector = _basis_vector(EMBEDDING_DIM, 0, 1.0)

    vectors = [_basis_vector(EMBEDDING_DIM, 0, 1.0 - i * 0.01) for i in range(5)]
    chunk_records = [
        ChunkRecord(ordinal=i, text=f"chunk {i}", token_count=2) for i in range(5)
    ]
    store_chunks_with_embeddings(
        db_session,
        document_version_id=document_version_id,
        chunk_records=chunk_records,
        vectors=vectors,
        embedding_model="test-model",
        embedding_version="v1",
    )

    results = vector_search(db_session, query_vector, limit=3)

    assert len(results) == 3
    assert [rank for _, rank in results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# fuse_rrf (pure unit test, no DB)
# ---------------------------------------------------------------------------


def test_fuse_rrf_matches_hand_computed_scores():
    # Hand-constructed ranked lists, no DB involved.
    vector_results = [(1, 1), (2, 2), (3, 3)]
    lexical_results = [(2, 1), (4, 2)]
    k = 60

    # Hand-computed expected scores, score(chunk) = sum of 1/(k+rank)
    # over each list containing it:
    #   chunk 1: vector rank 1            -> 1/61
    #   chunk 2: vector rank 2 + lex rank 1 -> 1/62 + 1/61
    #   chunk 3: vector rank 3            -> 1/63
    #   chunk 4: lexical rank 2           -> 1/62
    expected_scores = {
        1: 1 / 61,
        2: 1 / 62 + 1 / 61,
        3: 1 / 63,
        4: 1 / 62,
    }
    # Expected order (descending score): 2 > 1 > 4 > 3
    # since 1/61 > 1/62 > 1/63 individually, and chunk 2's combined score
    # (~0.032522) exceeds every single-list score.
    assert expected_scores[2] > expected_scores[1] > expected_scores[4] > expected_scores[3]

    results = fuse_rrf(vector_results, lexical_results, k=k)

    assert [chunk_id for chunk_id, _ in results] == [2, 1, 4, 3]
    for chunk_id, score in results:
        assert score == pytest.approx(expected_scores[chunk_id])


def test_fuse_rrf_default_k_is_60():
    # Single-list, single-item case isolates the default k directly:
    # score = 1/(k+1). With the documented default k=60, that's 1/61.
    results = fuse_rrf([(42, 1)], [])
    assert results == [(42, pytest.approx(1 / 61))]


def test_fuse_rrf_orders_by_score_descending_with_id_tiebreak():
    # Two chunks with identical scores (both rank 1 in disjoint lists) --
    # tie-break must be deterministic (chunk_id ascending), not
    # incidental input order.
    results = fuse_rrf([(20, 1)], [(10, 1)], k=60)
    assert [chunk_id for chunk_id, _ in results] == [10, 20]


def test_fuse_rrf_empty_inputs_returns_empty():
    assert fuse_rrf([], [], k=60) == []


# ---------------------------------------------------------------------------
# Integration: RRF actually combines both signals
# ---------------------------------------------------------------------------


def test_fuse_rrf_combines_lexical_and_vector_signals(db_session):
    """A chunk that ranks moderately on *both* lexical and vector search
    should outrank chunks that rank #1 on only one of the two signals --
    proving fusion is really combining both, not just picking one.
    """
    document_version_id = _make_document_version(db_session)

    query_vector = _basis_vector(EMBEDDING_DIM, 0, 1.0)

    # chunk_lexical_only: best lexical match (verified rank ~0.2083 for
    # this exact text against "VPN password reset", same fixture as
    # test_lexical_search_ranks_term_dense_chunk_highest above), but its
    # vector is orthogonal to the query vector (cosine distance 1.0 --
    # worst possible, excluded once vector limit=2 is applied).
    lexical_only_vector = _basis_vector(EMBEDDING_DIM, 1, 1.0)

    # chunk_vector_only: identical vector to the query (cosine distance
    # 0 -- best possible vector match), but its text has zero lexical
    # overlap with the query terms at all (no VPN/password/reset match).
    vector_only_vector = _basis_vector(EMBEDDING_DIM, 0, 1.0)

    # chunk_both: moderate on *both* signals -- weaker lexical match
    # (verified rank ~0.025, i.e. rank 2 behind chunk_lexical_only) and a
    # near-but-not-exact vector match (rank 2 behind chunk_vector_only).
    both_vector = _basis_vector(EMBEDDING_DIM, 0, 0.9)
    both_vector[1] = 0.1

    chunk_records = [
        ChunkRecord(
            ordinal=0,
            text=(
                "VPN password reset instructions: to reset your VPN password, "
                "use the VPN client's password reset link."
            ),
            token_count=10,
        ),
        ChunkRecord(
            ordinal=1,
            text="The office coffee machine on the third floor is out of order until Friday.",
            token_count=10,
        ),
        ChunkRecord(
            ordinal=2,
            text=(
                "For general account issues, contact IT support. VPN access "
                "requires a password reset request via the help desk."
            ),
            token_count=10,
        ),
    ]
    vectors = [lexical_only_vector, vector_only_vector, both_vector]
    pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=document_version_id,
        chunk_records=chunk_records,
        vectors=vectors,
        embedding_model="test-model",
        embedding_version="v1",
    )
    lexical_only_id = pairs[0][0].id
    vector_only_id = pairs[1][0].id
    both_id = pairs[2][0].id

    lexical_results = lexical_search(db_session, "VPN password reset", limit=2)
    vector_results = vector_search(db_session, query_vector, limit=2)

    # Sanity-check the two input lists actually isolate one signal each,
    # before asserting anything about the fusion.
    assert lexical_results == [(lexical_only_id, 1), (both_id, 2)]
    assert vector_results == [(vector_only_id, 1), (both_id, 2)]

    fused = fuse_rrf(vector_results, lexical_results, k=60)
    fused_ids_in_order = [chunk_id for chunk_id, _ in fused]

    # both_id appears in both lists (rank 2 in each: score = 2/62) and
    # must outrank lexical_only_id and vector_only_id, each of which
    # appears in only one list (rank 1 in that list: score = 1/61) --
    # 2/62 (~0.032258) > 1/61 (~0.016393).
    assert fused_ids_in_order[0] == both_id
    assert set(fused_ids_in_order[1:]) == {lexical_only_id, vector_only_id}

    scores = dict(fused)
    assert scores[both_id] == pytest.approx(2 / 62)
    assert scores[lexical_only_id] == pytest.approx(1 / 61)
    assert scores[vector_only_id] == pytest.approx(1 / 61)
    assert scores[both_id] > scores[lexical_only_id]
    assert scores[both_id] > scores[vector_only_id]
