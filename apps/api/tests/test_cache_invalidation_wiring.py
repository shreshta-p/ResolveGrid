"""Task 6 (`docs/superpowers/plans/2026-09-29-phase12-caching.md`): the real,
non-mocked proof that re-running ingestion for real -- not merely passing a
different literal `kb_version` integer, which `test_retrieval_cache.py`'s
`test_different_kb_version_gets_an_independent_cache_entry` and
`test_response_cache.py`'s `test_different_kb_versions_get_independent_entries`
already cover -- genuinely bumps `get_kb_version`'s real return value via a
real new completed `IngestionRun` row, and that a retrieval/response cache
entry populated under the OLD `kb_version` is consequently never looked up
again under the NEW one. The next identical query under the new kb_version is
a real MISS that triggers a real fresh `compute_fn()` call (retrieval) / a
real fallthrough to running the graph again (response), not a stale hit.

This does NOT re-implement or duplicate `get_kb_version`/`compute_scope_sig`/
the cache GET/SET plumbing -- it composes three already-tested real pieces
(`IngestionRun` row creation, `get_kb_version`, and each cache module's own
real lookup/store functions) in the one combination neither cache's own test
module exercises: a REAL kb_version bump via a REAL `IngestionRun` row, not a
hand-picked literal.

Semantic cache (Task 5): per this task's own instructions, checked first via
`grep -n "kb_version" apps/api/tests/test_semantic_cache.py` before writing
anything here -- `test_kb_version_isolation_identical_embedding_and_scope_but_
old_kb_version` (that module) already proves the identical property for
`semantic_cache.lookup_semantic_cache`/`store_semantic_cache_entry` (using
the worst case for a false hit: an IDENTICAL query embedding and scope_sig,
differing only in kb_version) -- not duplicated here.

The stale OLD-kb_version entry is also confirmed to still be *present* (a
real hit under its own original key) after the bump, proving the "old keys
are simply never looked up again, not explicitly deleted" stale-handling
strategy both cache modules' docstrings document, rather than assuming it.
"""

from sqlalchemy import delete, select

from resolvegrid_api.cache import REDIS_URL
from resolvegrid_api.models.knowledge import IngestionRun
from resolvegrid_api.response_cache import (
    lookup_cached_response,
    store_cached_response,
)
from resolvegrid_api.retrieval_cache import get_kb_version, get_or_compute_retrieval

import redis
import pytest


def _clear_namespaces(client: redis.Redis) -> None:
    for pattern in ("cache:retrieval:*", "cache:response:*"):
        for key in client.scan_iter(match=pattern):
            client.delete(key)


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fail fast (real connection check) if Redis isn't up
    _clear_namespaces(client)
    yield client
    _clear_namespaces(client)


def _complete_real_ingestion_run(session) -> IngestionRun:
    """Create and commit one real `status="completed"` `IngestionRun` row --
    the real event `get_kb_version` watches for (per `retrieval_cache.py`'s
    module docstring: `select(func.max(IngestionRun.id)).where(status ==
    "completed")`). This is deliberately a bare row insert, not a full
    `run_seed_corpus_ingestion()` call -- the real corpus-ingestion pipeline
    is already covered end-to-end by `test_ingestion_worker.py`; what THIS
    test needs is only the real, minimal event that bumps the kb_version
    stamp, so the cache-invalidation assertion isn't entangled with (and
    slowed down by) a real full ingestion run.
    """
    run = IngestionRun(
        parser_version="cache-invalidation-test-v1",
        chunking_version="cache-invalidation-test-v1",
        embedding_version="cache-invalidation-test-v1",
        status="completed",
    )
    session.add(run)
    session.commit()
    return run


def test_retrieval_cache_entry_under_old_kb_version_is_not_looked_up_after_real_ingestion_run(
    raw_db_session, redis_client
):
    scope_sig = "unrestricted"
    query_text = "cache invalidation wiring: what is a firewall?"

    old_kb_version = get_kb_version(raw_db_session)

    old_calls = {"count": 0}

    def compute_old():
        old_calls["count"] += 1
        return {"chunks": [], "sufficient": False, "context_block": "old-kb-version-result"}

    # Populate the cache under the OLD kb_version -- a real miss, real compute.
    first = get_or_compute_retrieval(
        query_text=query_text,
        kb_version=old_kb_version,
        scope_sig=scope_sig,
        redis_client=redis_client,
        compute_fn=compute_old,
    )
    assert old_calls["count"] == 1
    assert first["context_block"] == "old-kb-version-result"

    # Confirm it's a real, repeatable hit under the SAME (old) kb_version
    # before bumping anything -- establishes the baseline this test's
    # invalidation assertion is actually about.
    second = get_or_compute_retrieval(
        query_text=query_text,
        kb_version=old_kb_version,
        scope_sig=scope_sig,
        redis_client=redis_client,
        compute_fn=compute_old,
    )
    assert old_calls["count"] == 1  # still 1: real cache hit, compute_fn not called again
    assert second["context_block"] == "old-kb-version-result"

    new_run = None
    try:
        # Real event: re-running ingestion creates a new completed
        # IngestionRun row, for real bumping the real kb_version stamp.
        new_run = _complete_real_ingestion_run(raw_db_session)
        new_kb_version = get_kb_version(raw_db_session)
        assert new_kb_version == new_run.id
        assert new_kb_version > old_kb_version

        new_calls = {"count": 0}

        def compute_new():
            new_calls["count"] += 1
            return {"chunks": [], "sufficient": False, "context_block": "new-kb-version-result"}

        # The IDENTICAL query_text/scope_sig, looked up under the NEW
        # (post-ingestion) kb_version, must be a genuine MISS -- the old
        # entry is keyed differently and is never found -- triggering a
        # real fresh compute, not a stale hit of "old-kb-version-result".
        third = get_or_compute_retrieval(
            query_text=query_text,
            kb_version=new_kb_version,
            scope_sig=scope_sig,
            redis_client=redis_client,
            compute_fn=compute_new,
        )
        assert new_calls["count"] == 1, "expected a real recompute under the new kb_version, not a stale hit"
        assert third["context_block"] == "new-kb-version-result"

        # The OLD entry is still a real, live hit under its OWN original
        # key -- proving the stale-handling strategy is "never looked up
        # under the new key", not "explicitly deleted on invalidation".
        still_old = get_or_compute_retrieval(
            query_text=query_text,
            kb_version=old_kb_version,
            scope_sig=scope_sig,
            redis_client=redis_client,
            compute_fn=compute_old,
        )
        assert old_calls["count"] == 1  # unchanged: still a real hit, not recomputed
        assert still_old["context_block"] == "old-kb-version-result"
    finally:
        if new_run is not None:
            raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id == new_run.id))
            raw_db_session.commit()


def test_response_cache_entry_under_old_kb_version_is_not_looked_up_after_real_ingestion_run(
    raw_db_session, redis_client
):
    scope_sig = "unrestricted"
    prompt_template_version = "1"
    message_text = "cache invalidation wiring: what is a VPN?"

    old_kb_version = get_kb_version(raw_db_session)

    # Before the bump: a real miss, then a real store, then a real hit --
    # under the OLD kb_version.
    assert (
        lookup_cached_response(
            message_text=message_text,
            kb_version=old_kb_version,
            scope_sig=scope_sig,
            prompt_template_version=prompt_template_version,
            redis_client=redis_client,
        )
        is None
    )
    store_cached_response(
        message_text=message_text,
        kb_version=old_kb_version,
        scope_sig=scope_sig,
        prompt_template_version=prompt_template_version,
        redis_client=redis_client,
        value={"answer": "old-kb-version-answer"},
    )
    assert lookup_cached_response(
        message_text=message_text,
        kb_version=old_kb_version,
        scope_sig=scope_sig,
        prompt_template_version=prompt_template_version,
        redis_client=redis_client,
    ) == {"answer": "old-kb-version-answer"}

    new_run = None
    try:
        new_run = _complete_real_ingestion_run(raw_db_session)
        new_kb_version = get_kb_version(raw_db_session)
        assert new_kb_version == new_run.id
        assert new_kb_version > old_kb_version

        # The identical message/scope/template, looked up under the NEW
        # kb_version, must be a genuine miss -- `routers/chat.py`'s real
        # caller treats this as "fall through and run the real graph",
        # never a stale hit of "old-kb-version-answer".
        assert (
            lookup_cached_response(
                message_text=message_text,
                kb_version=new_kb_version,
                scope_sig=scope_sig,
                prompt_template_version=prompt_template_version,
                redis_client=redis_client,
            )
            is None
        )

        # The OLD entry is still a real, live hit under its own original
        # key -- stale handling is "never looked up under the new key",
        # not an explicit delete-on-invalidation pass.
        assert lookup_cached_response(
            message_text=message_text,
            kb_version=old_kb_version,
            scope_sig=scope_sig,
            prompt_template_version=prompt_template_version,
            redis_client=redis_client,
        ) == {"answer": "old-kb-version-answer"}
    finally:
        if new_run is not None:
            raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id == new_run.id))
            raw_db_session.commit()
