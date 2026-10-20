"""Tests for `resolvegrid_api.semantic_cache` (Phase 12 Task 5: semantic
-response cache + mandatory cross-scope leakage and kb_version-isolation
tests).

Real Postgres throughout (nothing here mocks the DB away), matching this
codebase's established philosophy (`test_retrieval_cache.py`'s module
docstring, Task 2). The threshold-correctness test uses REAL embeddings
from the real running Ollama `nomic-embed-text` model (not fixture
vectors) -- this is the one test that must prove the actual empirically
-chosen `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` constant correctly separates
real near-duplicate rewordings from real different-topic questions, so it
needs the real model, not a stand-in. The scope/kb_version isolation tests
below use fixed synthetic vectors instead (deliberately IDENTICAL vectors
across the two sides of each isolation test -- the worst case, zero cosine
distance -- so a bug in the scope/kb_version filtering itself, not merely
in the threshold, is what these tests actually prove) since those tests
are about the SQL `WHERE` boundary, not about embedding quality, and fixed
vectors make them fast and deterministic.
"""

import json

import pytest
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.orm import Session

from resolvegrid_api.models.knowledge import SemanticResponseCache
from resolvegrid_api.semantic_cache import (
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SEMANTIC_CACHE_TTL_SECONDS,
    lookup_semantic_cache,
    purge_expired_semantic_cache_entries,
    store_semantic_cache_entry,
)
from resolvegrid_retrieval.embedder import DEFAULT_EMBEDDING_MODEL, embed_texts

_TEST_SCOPE_PREFIX = "test-semantic-cache:"


def _clear_test_rows(session: Session) -> None:
    session.execute(
        delete(SemanticResponseCache).where(
            SemanticResponseCache.scope_sig.like(f"{_TEST_SCOPE_PREFIX}%")
        )
    )
    session.commit()


@pytest.fixture
def semantic_session(raw_db_session):
    _clear_test_rows(raw_db_session)
    yield raw_db_session
    _clear_test_rows(raw_db_session)


def _fixed_vector(seed: float, dim: int = 768) -> list[float]:
    """A deterministic, fixed-length fake embedding -- used only for the
    scope/kb_version isolation tests below, which need two sides to embed
    to the IDENTICAL vector (not a realistic embedding, just a fixed
    768-float list matching the real `Vector(768)` column's dimension).
    """
    return [seed] * dim


def test_lookup_is_a_real_miss_before_any_store(semantic_session):
    result = lookup_semantic_cache(
        semantic_session,
        query_embedding=_fixed_vector(0.01),
        scope_sig=f"{_TEST_SCOPE_PREFIX}miss-check",
        kb_version=1,
    )
    assert result is None


def test_store_then_lookup_with_identical_embedding_is_a_real_hit(semantic_session):
    scope_sig = f"{_TEST_SCOPE_PREFIX}hit-check"
    vector = _fixed_vector(0.5)
    value = {
        "answer": "A VPN is a Virtual Private Network.",
        "citations": [{"chunk_id": 1, "document_title": "What Is a VPN"}],
        "caption": "General-knowledge answer",
        "intent": "general_question",
        "risk_level": "low",
    }

    store_semantic_cache_entry(
        semantic_session, query_embedding=vector, scope_sig=scope_sig, kb_version=7, value=value,
    )

    result = lookup_semantic_cache(
        semantic_session, query_embedding=vector, scope_sig=scope_sig, kb_version=7,
    )
    assert result == value


def test_ttl_constant_is_the_documented_15_minutes_short_bound():
    """Real, specific number this module's docstring commits to -- mirrors
    Task 3's `RESPONSE_CACHE_TTL_SECONDS` bound/rationale exactly (see
    `semantic_cache.py`'s module docstring).
    """
    assert SEMANTIC_CACHE_TTL_SECONDS == 15 * 60


def test_stale_row_past_ttl_is_not_served(semantic_session):
    """A row older than `SEMANTIC_CACHE_TTL_SECONDS` must not be served --
    proven by inserting a row with a real, explicitly backdated `created_at`
    (not waiting 15 real minutes) and confirming the lookup treats it as a
    genuine miss.
    """
    scope_sig = f"{_TEST_SCOPE_PREFIX}stale-check"
    vector = _fixed_vector(0.33)
    entry = SemanticResponseCache(
        scope_sig=scope_sig,
        kb_version=1,
        query_embedding=vector,
        answer="stale answer",
        citations_json=json.dumps([]),
        caption="stale caption",
        intent="general_question",
        risk_level="low",
    )
    semantic_session.add(entry)
    semantic_session.commit()
    # Backdate created_at past the TTL bound directly via real SQL (ORM
    # `default`/`server_default` only apply on INSERT).
    semantic_session.execute(
        text(
            "UPDATE semantic_response_cache SET created_at = now() - make_interval(secs => :past) "
            "WHERE id = :id"
        ),
        {"past": SEMANTIC_CACHE_TTL_SECONDS + 60, "id": entry.id},
    )
    semantic_session.commit()

    result = lookup_semantic_cache(
        semantic_session, query_embedding=vector, scope_sig=scope_sig, kb_version=1,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Mandatory security test: cross-scope leakage must be impossible, even in
# the worst case (identical embedding -- zero cosine distance -- used for
# both sides).
# ---------------------------------------------------------------------------


def test_cross_scope_leakage_identical_embedding_never_shared_across_scopes(semantic_session):
    identical_vector = _fixed_vector(0.77)
    scope_sig_a = f"{_TEST_SCOPE_PREFIX}scope-A"
    scope_sig_b = f"{_TEST_SCOPE_PREFIX}scope-B"
    kb_version = 42

    store_semantic_cache_entry(
        semantic_session,
        query_embedding=identical_vector,
        scope_sig=scope_sig_a,
        kb_version=kb_version,
        value={
            "answer": "ENGINEERING-SECRET-ANSWER",
            "citations": [],
            "caption": "c",
            "intent": "general_question",
            "risk_level": "low",
        },
    )

    # Scope B issues the IDENTICAL query embedding -- the worst-case
    # adversarial input (zero cosine distance to scope A's stored entry) --
    # and must still get a real miss: scope A's entry is never even a
    # candidate for scope B's lookup.
    result_b = lookup_semantic_cache(
        semantic_session, query_embedding=identical_vector, scope_sig=scope_sig_b, kb_version=kb_version,
    )
    assert result_b is None

    # Sanity: scope A's OWN lookup with the same embedding is a real hit --
    # proves the miss above is a real scope boundary, not a broken lookup
    # that misses everything.
    result_a = lookup_semantic_cache(
        semantic_session, query_embedding=identical_vector, scope_sig=scope_sig_a, kb_version=kb_version,
    )
    assert result_a is not None
    assert result_a["answer"] == "ENGINEERING-SECRET-ANSWER"


def test_kb_version_isolation_identical_embedding_and_scope_but_old_kb_version(semantic_session):
    """An entry cached under an OLD kb_version must not match after a new
    one, mirroring Task 2's kb_version-isolation convention -- proven with
    the same worst-case identical-embedding input.
    """
    identical_vector = _fixed_vector(0.42)
    scope_sig = f"{_TEST_SCOPE_PREFIX}kbversion-check"

    store_semantic_cache_entry(
        semantic_session,
        query_embedding=identical_vector,
        scope_sig=scope_sig,
        kb_version=100,
        value={
            "answer": "OLD-KB-VERSION-ANSWER",
            "citations": [],
            "caption": "c",
            "intent": "general_question",
            "risk_level": "low",
        },
    )

    # A new kb_version (corpus re-ingested) must not see the old entry.
    result_new = lookup_semantic_cache(
        semantic_session, query_embedding=identical_vector, scope_sig=scope_sig, kb_version=101,
    )
    assert result_new is None

    # The OLD kb_version's own lookup is still a real hit -- proves the
    # miss above is a real kb_version boundary, not a broken lookup.
    result_old = lookup_semantic_cache(
        semantic_session, query_embedding=identical_vector, scope_sig=scope_sig, kb_version=100,
    )
    assert result_old is not None
    assert result_old["answer"] == "OLD-KB-VERSION-ANSWER"


# ---------------------------------------------------------------------------
# Real Postgres-outage degrade
# ---------------------------------------------------------------------------


def test_lookup_postgres_outage_degrades_to_none_instead_of_raising():
    """A real, genuinely unreachable Postgres endpoint (port 1, refuses
    connections immediately) -- not a mock -- proves this cache degrades to
    a real miss rather than propagating and breaking `/chat`.
    """
    unreachable_engine = create_engine(
        "postgresql+psycopg://user:pass@localhost:1/db",
        connect_args={"connect_timeout": 1},
    )
    unreachable_session = Session(unreachable_engine)
    try:
        result = lookup_semantic_cache(
            unreachable_session,
            query_embedding=_fixed_vector(0.1),
            scope_sig="unrestricted",
            kb_version=1,
        )
        assert result is None
    finally:
        unreachable_session.close()
        unreachable_engine.dispose()


def test_store_postgres_outage_is_swallowed_without_raising():
    unreachable_engine = create_engine(
        "postgresql+psycopg://user:pass@localhost:1/db",
        connect_args={"connect_timeout": 1},
    )
    unreachable_session = Session(unreachable_engine)
    try:
        store_semantic_cache_entry(
            unreachable_session,
            query_embedding=_fixed_vector(0.1),
            scope_sig="unrestricted",
            kb_version=1,
            value={"answer": "x", "citations": [], "caption": "c", "intent": "i", "risk_level": "low"},
        )  # must not raise
    finally:
        unreachable_session.close()
        unreachable_engine.dispose()


# ---------------------------------------------------------------------------
# Empirical threshold-tuning proof: REAL embeddings from the real running
# Ollama nomic-embed-text model, proving SEMANTIC_CACHE_SIMILARITY_THRESHOLD
# (0.15) correctly separates real near-duplicate rewordings (must be a HIT)
# from real different-topic questions (must be a MISS). The real numbers
# this threshold was derived from are cited in semantic_cache.py's own
# module docstring.
# ---------------------------------------------------------------------------


def test_empirical_threshold_matches_near_duplicate_questions(semantic_session):
    scope_sig = f"{_TEST_SCOPE_PREFIX}threshold-match"
    original = "What is a firewall?"
    reworded = "Can you explain what a firewall is?"

    original_vector = embed_texts([original], model=DEFAULT_EMBEDDING_MODEL)[0]
    store_semantic_cache_entry(
        semantic_session,
        query_embedding=original_vector,
        scope_sig=scope_sig,
        kb_version=1,
        value={
            "answer": "A firewall is a network security device.",
            "citations": [],
            "caption": "c",
            "intent": "general_question",
            "risk_level": "low",
        },
    )

    reworded_vector = embed_texts([reworded], model=DEFAULT_EMBEDDING_MODEL)[0]
    result = lookup_semantic_cache(
        semantic_session, query_embedding=reworded_vector, scope_sig=scope_sig, kb_version=1,
    )
    assert result is not None
    assert result["answer"] == "A firewall is a network security device."


def test_empirical_threshold_rejects_different_topic_question(semantic_session):
    scope_sig = f"{_TEST_SCOPE_PREFIX}threshold-reject"
    original = "What is a firewall?"
    different_topic = "How do I request VPN access?"

    original_vector = embed_texts([original], model=DEFAULT_EMBEDDING_MODEL)[0]
    store_semantic_cache_entry(
        semantic_session,
        query_embedding=original_vector,
        scope_sig=scope_sig,
        kb_version=1,
        value={
            "answer": "A firewall is a network security device.",
            "citations": [],
            "caption": "c",
            "intent": "general_question",
            "risk_level": "low",
        },
    )

    different_vector = embed_texts([different_topic], model=DEFAULT_EMBEDDING_MODEL)[0]
    result = lookup_semantic_cache(
        semantic_session, query_embedding=different_vector, scope_sig=scope_sig, kb_version=1,
    )
    assert result is None


def test_threshold_constant_is_the_documented_empirically_chosen_value():
    assert SEMANTIC_CACHE_SIMILARITY_THRESHOLD == 0.15


# ---------------------------------------------------------------------------
# purge_expired_semantic_cache_entries (Phase 12 Task 5 code-review
# follow-up): the real DELETE that bounds this table's row count, closing
# the gap flagged on review -- a `created_at` lookup filter alone never
# deletes anything, so without this function the table grows unbounded.
# ---------------------------------------------------------------------------


def test_purge_deletes_only_rows_older_than_the_ttl_cutoff(semantic_session):
    scope_sig = f"{_TEST_SCOPE_PREFIX}purge-check"

    old_entry = SemanticResponseCache(
        scope_sig=scope_sig, kb_version=1, query_embedding=_fixed_vector(0.11),
        answer="old", citations_json=json.dumps([]), caption="c",
        intent="general_question", risk_level="low",
    )
    fresh_entry = SemanticResponseCache(
        scope_sig=scope_sig, kb_version=1, query_embedding=_fixed_vector(0.22),
        answer="fresh", citations_json=json.dumps([]), caption="c",
        intent="general_question", risk_level="low",
    )
    semantic_session.add_all([old_entry, fresh_entry])
    semantic_session.commit()
    # Captured as plain ints BEFORE any further commit -- `purge_expired_
    # semantic_cache_entries` below commits internally, which (with
    # SQLAlchemy's default expire_on_commit) expires these ORM instances'
    # attributes; re-accessing `old_entry.id` afterward would try to
    # refresh-SELECT a row that's genuinely gone by then, raising
    # `ObjectDeletedError` -- plain python ints sidestep that entirely.
    old_id, fresh_id = old_entry.id, fresh_entry.id

    # Backdate ONLY the old entry past the real TTL bound via real SQL
    # (mirrors test_stale_row_past_ttl_is_not_served's identical technique)
    # -- the fresh entry's created_at is left at its real, just-inserted
    # server_default=now() value.
    semantic_session.execute(
        text(
            "UPDATE semantic_response_cache SET created_at = now() - make_interval(secs => :past) "
            "WHERE id = :id"
        ),
        {"past": SEMANTIC_CACHE_TTL_SECONDS + 60, "id": old_id},
    )
    semantic_session.commit()

    deleted_count = purge_expired_semantic_cache_entries(semantic_session)
    assert deleted_count >= 1  # at least our one old row (a shared dev DB may have others)

    remaining_ids = set(
        semantic_session.scalars(
            select(SemanticResponseCache.id).where(SemanticResponseCache.id.in_([old_id, fresh_id]))
        ).all()
    )
    assert old_id not in remaining_ids  # purged -- past the TTL cutoff
    assert fresh_id in remaining_ids  # kept -- still within the TTL window


def test_purge_does_not_delete_a_fresh_unexpired_row(semantic_session):
    scope_sig = f"{_TEST_SCOPE_PREFIX}purge-nothing-expired"
    store_semantic_cache_entry(
        semantic_session, query_embedding=_fixed_vector(0.33), scope_sig=scope_sig, kb_version=1,
        value={"answer": "x", "citations": [], "caption": "c", "intent": "i", "risk_level": "low"},
    )

    # A real, freshly-inserted row must not be purged -- real purge activity
    # on THIS test's own row must be zero (other pre-existing stale rows in
    # a shared dev DB may still be purged by the same call, hence checking
    # this test's own row survived rather than asserting a global count).
    purge_expired_semantic_cache_entries(semantic_session)

    survived = semantic_session.scalar(
        select(SemanticResponseCache.id).where(
            SemanticResponseCache.scope_sig == scope_sig, SemanticResponseCache.kb_version == 1,
        )
    )
    assert survived is not None


def test_purge_postgres_outage_degrades_to_zero_instead_of_raising():
    unreachable_engine = create_engine(
        "postgresql+psycopg://user:pass@localhost:1/db", connect_args={"connect_timeout": 1}
    )
    unreachable_session = Session(unreachable_engine)
    try:
        result = purge_expired_semantic_cache_entries(unreachable_session)
        assert result == 0
    finally:
        unreachable_session.close()
        unreachable_engine.dispose()
