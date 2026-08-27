"""Tests for `resolvegrid_api.retrieval` (Phase 7 Task 4: lexical search +
RRF fusion; Phase 7 Task 5: authz-aware filtering + sufficiency check).

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
from resolvegrid_api.models.org import Department, Employee
from resolvegrid_api.retrieval import (
    DEFAULT_MIN_SCORE_THRESHOLD,
    assess_sufficiency,
    fuse_rrf,
    lexical_search,
    vector_search,
)
from resolvegrid_api.retrieval_authz import AuthzFilter, build_authz_filter, normalize_department_tag
from resolvegrid_authz import Principal, RoleGrant
from resolvegrid_retrieval.chunker import ChunkRecord

EMBEDDING_DIM = 768

# Task 4's existing tests (lexical/vector/fusion) aren't exercising authz --
# they pass this shared "see everything" filter so Task 5's now-required
# `authz_filter` parameter doesn't change what those tests are actually
# asserting about. Task 5's own tests below construct real, restrictive
# `AuthzFilter`s instead.
UNRESTRICTED = AuthzFilter(unrestricted=True)


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

    results = lexical_search(db_session, "VPN password reset", limit=10, authz_filter=UNRESTRICTED)

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

    results = lexical_search(db_session, "VPN password reset", limit=2, authz_filter=UNRESTRICTED)

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

    results = vector_search(db_session, query_vector, limit=2, authz_filter=UNRESTRICTED)

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

    results = vector_search(db_session, query_vector, limit=3, authz_filter=UNRESTRICTED)

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

    lexical_results = lexical_search(db_session, "VPN password reset", limit=2, authz_filter=UNRESTRICTED)
    vector_results = vector_search(db_session, query_vector, limit=2, authz_filter=UNRESTRICTED)

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


# ---------------------------------------------------------------------------
# build_authz_filter (Phase 7 Task 5)
# ---------------------------------------------------------------------------


def _make_department(session, name: str) -> Department:
    department = Department(name=name)
    session.add(department)
    session.flush()
    return department


def _make_employee(session, *, department_id: int | None, email_suffix: str) -> Employee:
    from datetime import datetime

    employee = Employee(
        display_name="Test Employee",
        email=f"test-{email_suffix}@example.com",
        title="Engineer",
        hire_date=datetime(2020, 1, 1),
        timezone="UTC",
        department_id=department_id,
    )
    session.add(employee)
    session.flush()
    return employee


def test_build_authz_filter_admin_is_unrestricted(db_session):
    principal = Principal(employee_id=1, roles=(RoleGrant(role="admin", scope="global"),))

    result = build_authz_filter(principal, db_session)

    assert result.unrestricted is True


def test_build_authz_filter_department_scoped_grant_maps_to_department_tags(db_session):
    # Distinct names from seed.py's real `_DEPARTMENTS` list -- Department.name
    # is unique, and this test runs against the live (seeded) DB, so reusing
    # a real seeded name would collide.
    engineering = _make_department(db_session, "Test Retrieval Engineering Dept")
    security = _make_department(db_session, "Test Retrieval Security Dept")
    principal = Principal(
        employee_id=2,
        roles=(
            RoleGrant(role="analyst", scope="department", scope_id=engineering.id),
            RoleGrant(role="approver", scope="department", scope_id=security.id),
        ),
    )

    result = build_authz_filter(principal, db_session)

    assert result.unrestricted is False
    assert result.allowed_tags == frozenset(
        {"test_retrieval_engineering_dept", "test_retrieval_security_dept"}
    )


def test_build_authz_filter_self_scope_falls_back_to_home_department(db_session):
    department = _make_department(db_session, "Test Retrieval People Dept")
    employee = _make_employee(db_session, department_id=department.id, email_suffix="self1")
    principal = Principal(employee_id=employee.id, roles=())

    result = build_authz_filter(principal, db_session)

    assert result.unrestricted is False
    assert result.allowed_tags == frozenset({"test_retrieval_people_dept"})


def test_build_authz_filter_self_scope_with_no_department_is_empty(db_session):
    employee = _make_employee(db_session, department_id=None, email_suffix="self2")
    principal = Principal(employee_id=employee.id, roles=())

    result = build_authz_filter(principal, db_session)

    assert result.unrestricted is False
    assert result.allowed_tags == frozenset()


def test_build_authz_filter_unknown_employee_fails_closed(db_session):
    principal = Principal(employee_id=999_999, roles=())

    result = build_authz_filter(principal, db_session)

    assert result.unrestricted is False
    assert result.allowed_tags == frozenset()


def test_normalize_department_tag():
    assert normalize_department_tag("Platform Engineering") == "platform_engineering"
    assert normalize_department_tag("HR") == "hr"
    assert normalize_department_tag("  Security  ") == "security"


# ---------------------------------------------------------------------------
# assess_sufficiency (Phase 7 Task 5)
# ---------------------------------------------------------------------------


def test_assess_sufficiency_empty_results_is_insufficient():
    result = assess_sufficiency([])

    assert result.sufficient is False
    assert result.result_count == 0
    assert result.top_score is None


def test_assess_sufficiency_top_score_above_threshold_is_sufficient():
    fused = [(1, DEFAULT_MIN_SCORE_THRESHOLD * 2), (2, 0.001)]

    result = assess_sufficiency(fused)

    assert result.sufficient is True
    assert result.result_count == 2
    assert result.top_score == pytest.approx(DEFAULT_MIN_SCORE_THRESHOLD * 2)


def test_assess_sufficiency_top_score_below_threshold_is_insufficient():
    fused = [(1, DEFAULT_MIN_SCORE_THRESHOLD / 2)]

    result = assess_sufficiency(fused)

    assert result.sufficient is False
    assert result.top_score == pytest.approx(DEFAULT_MIN_SCORE_THRESHOLD / 2)


def test_assess_sufficiency_boundary_score_exactly_at_threshold_is_sufficient():
    # score == threshold must count as sufficient (>=), not just >.
    fused = [(1, DEFAULT_MIN_SCORE_THRESHOLD)]

    result = assess_sufficiency(fused)

    assert result.sufficient is True


def test_assess_sufficiency_boundary_score_just_below_threshold_is_insufficient():
    fused = [(1, DEFAULT_MIN_SCORE_THRESHOLD - 1e-9)]

    result = assess_sufficiency(fused)

    assert result.sufficient is False


def test_assess_sufficiency_respects_custom_min_results():
    # A high-scoring single result is still insufficient if the caller
    # demands corroboration from >=2 results.
    fused = [(1, 1.0)]

    result = assess_sufficiency(fused, min_results=2)

    assert result.sufficient is False
    assert result.result_count == 1


def test_assess_sufficiency_respects_custom_min_score_threshold():
    fused = [(1, 0.5)]

    result = assess_sufficiency(fused, min_score_threshold=0.9)

    assert result.sufficient is False


# ---------------------------------------------------------------------------
# Adversarial authz-filter leakage test (Phase 7 Task 5 / Task 8 exit
# criteria) -- the single most important test in this task.
# ---------------------------------------------------------------------------


def test_adversarial_authz_filter_blocks_cross_scope_leakage(db_session):
    """A principal/filter scoped to ONE document's access tag must get ZERO
    chunks back from a DISJOINT-scoped document -- not "ranked lower",
    literally absent from both `vector_search`'s and `lexical_search`'s
    result sets.

    To make a filtering bug (e.g. an accidentally no-op WHERE clause)
    impossible to pass accidentally, the unauthorized (HR-scoped) chunk is
    engineered to be the single BEST possible match for the query on BOTH
    signals: its embedding is set to the exact query vector (cosine
    distance 0 -- the best possible vector match) and its text is set to
    the exact query string (guaranteed best lexical rank). A sanity check
    below proves that, unfiltered, this HR chunk really does win both
    searches outright -- so the later assertions that it's absent once
    the filter is scoped to "engineering" only are proof the filter is
    doing real work, not an accident of ranking.
    """
    engineering_doc = Document(
        source_type="synthetic_private",
        title="Engineering Internal Doc",
        checksum="adversarial-eng-checksum",
        access_scope_tags=["engineering"],
    )
    hr_doc = Document(
        source_type="synthetic_private",
        title="HR Internal Doc",
        checksum="adversarial-hr-checksum",
        access_scope_tags=["hr"],
    )
    db_session.add_all([engineering_doc, hr_doc])
    db_session.flush()

    engineering_version = DocumentVersion(
        document_id=engineering_doc.id,
        parser_version="v1",
        chunking_version="v1",
        content_hash="adversarial-eng-hash",
    )
    hr_version = DocumentVersion(
        document_id=hr_doc.id,
        parser_version="v1",
        chunking_version="v1",
        content_hash="adversarial-hr-hash",
    )
    db_session.add_all([engineering_version, hr_version])
    db_session.flush()

    query_text = "quarterly headcount budget planning figures"
    query_vector = _basis_vector(EMBEDDING_DIM, 0, 1.0)

    # Engineering chunk: a weak, unrelated match on both signals.
    engineering_pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=engineering_version.id,
        chunk_records=[
            ChunkRecord(ordinal=0, text="Deployment runbook for the API gateway service.", token_count=8)
        ],
        # Orthogonal to query_vector -- the worst possible vector match.
        vectors=[_basis_vector(EMBEDDING_DIM, 1, 1.0)],
        embedding_model="test-model",
        embedding_version="v1",
    )
    engineering_chunk_id = engineering_pairs[0][0].id

    # HR chunk: the single best possible match on both signals.
    hr_pairs = store_chunks_with_embeddings(
        db_session,
        document_version_id=hr_version.id,
        chunk_records=[ChunkRecord(ordinal=0, text=query_text, token_count=len(query_text.split()))],
        # Identical to query_vector -- cosine distance 0, best possible match.
        vectors=[query_vector],
        embedding_model="test-model",
        embedding_version="v1",
    )
    hr_chunk_id = hr_pairs[0][0].id

    # Sanity check: unfiltered, the HR chunk really does win both searches
    # outright -- this is what proves the later leakage assertions mean
    # something (a no-op filter would let this chunk straight through).
    sanity_vector = vector_search(db_session, query_vector, limit=10, authz_filter=UNRESTRICTED)
    sanity_lexical = lexical_search(db_session, query_text, limit=10, authz_filter=UNRESTRICTED)
    assert sanity_vector[0] == (hr_chunk_id, 1)
    assert sanity_lexical[0] == (hr_chunk_id, 1)

    # A principal/filter scoped to "engineering" only -- disjoint from the
    # HR document's "hr" tag.
    engineering_only_filter = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"engineering"}))

    vector_results = vector_search(db_session, query_vector, limit=10, authz_filter=engineering_only_filter)
    lexical_results = lexical_search(db_session, query_text, limit=10, authz_filter=engineering_only_filter)

    vector_result_ids = {chunk_id for chunk_id, _ in vector_results}
    lexical_result_ids = {chunk_id for chunk_id, _ in lexical_results}

    # The HR chunk must be LITERALLY ABSENT, not merely ranked lower.
    assert hr_chunk_id not in vector_result_ids
    assert hr_chunk_id not in lexical_result_ids
    # And the only chunk that ever *is* visible is the engineering-scoped one.
    assert vector_result_ids == {engineering_chunk_id}
    assert lexical_result_ids <= {engineering_chunk_id}

    # Symmetric check: an "hr"-only filter must exclude the engineering
    # chunk just as strictly.
    hr_only_filter = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"hr"}))
    vector_results_hr_scope = vector_search(db_session, query_vector, limit=10, authz_filter=hr_only_filter)
    vector_result_ids_hr_scope = {chunk_id for chunk_id, _ in vector_results_hr_scope}
    assert engineering_chunk_id not in vector_result_ids_hr_scope
    assert vector_result_ids_hr_scope == {hr_chunk_id}


def test_adversarial_authz_filter_is_baked_into_sql_where_clause(db_session):
    """Confirms, by inspecting the actual SQL text sent to Postgres (not
    just the returned rows), that `vector_search`/`lexical_search` really
    do fold the authz predicate into the query's own `WHERE`/`JOIN`
    clauses -- per the plan doc's explicit "baked into the SQL query
    itself, not applied post-hoc" requirement for this task.
    """
    from sqlalchemy import event

    executed_statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        executed_statements.append(statement)

    connection = db_session.get_bind()
    event.listen(connection, "before_cursor_execute", _capture)
    try:
        vector_search(db_session, _basis_vector(EMBEDDING_DIM, 0, 1.0), limit=5, authz_filter=UNRESTRICTED)
        lexical_search(db_session, "test query", limit=5, authz_filter=UNRESTRICTED)
    finally:
        event.remove(connection, "before_cursor_execute", _capture)

    vector_sql = next(s for s in executed_statements if "<=>" in s)
    lexical_sql = next(s for s in executed_statements if "ts_rank_cd" in s)

    for sql in (vector_sql, lexical_sql):
        lowered = sql.lower()
        assert "document.access_scope_tags" in lowered
        assert "join document_version" in lowered
        assert "join document " in lowered or lowered.rstrip().endswith("join document")
