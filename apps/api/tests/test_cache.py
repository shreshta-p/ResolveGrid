"""Tests for `resolvegrid_api.cache` (Phase 12 Task 1: embedding cache).

Real Redis, not mocked: every test in this file hits the actual
`resolvegrid-redis` dev container (`infra/docker-compose.yml`, host port
6380) via `REDIS_URL`, matching this codebase's established philosophy of
testing against real infra (`apps/api/tests/conftest.py`'s `db_session`/
`raw_db_session` fixtures hit a real Postgres, never a mocked Session).
"""

import hashlib
import json

import pytest
import redis

from resolvegrid_api.cache import (
    EMBEDDING_CACHE_TTL_SECONDS,
    REDIS_URL,
    get_or_compute_embeddings,
)


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fail fast (real connection check) if Redis isn't up
    yield client
    # Clean up every key this test file's own cache namespace might have
    # written, so tests don't leak state into each other or later runs.
    for key in client.scan_iter(match="cache:embedding:test-model*"):
        client.delete(key)


def test_cache_key_shape(redis_client):
    """`cache:embedding:{model}:{sha256(text)}` -- the exact key shape the
    plan doc's cross-cutting design section specifies.
    """
    text = "what is a firewall?"
    model = "test-model-keyshape"
    calls = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    get_or_compute_embeddings(
        [text], model=model, redis_client=redis_client, embed_fn=fake_embed
    )

    expected_key = f"cache:embedding:{model}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    raw = redis_client.get(expected_key)
    assert raw is not None
    assert json.loads(raw) == [0.1, 0.2, 0.3]


def test_second_call_is_a_cache_hit_and_does_not_recompute(redis_client):
    """The real point of this cache: an identical (model, text) pair must
    only ever invoke the real embed function once. This is proven with a
    real call-counting spy, not "should work" reasoning.
    """
    model = "test-model-hit"
    text = "what is a VPN?"
    calls = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 2.0] for _ in texts]

    first = get_or_compute_embeddings(
        [text], model=model, redis_client=redis_client, embed_fn=fake_embed
    )
    second = get_or_compute_embeddings(
        [text], model=model, redis_client=redis_client, embed_fn=fake_embed
    )

    assert first == [[1.0, 2.0]]
    assert second == [[1.0, 2.0]]
    # Only ONE real call to the embed function across both invocations --
    # the second was served entirely from Redis.
    assert calls == [[text]]


def test_ttl_is_set_on_written_key(redis_client):
    model = "test-model-ttl"
    text = "some stable ttl-checking text"

    get_or_compute_embeddings(
        [text], model=model, redis_client=redis_client, embed_fn=lambda ts: [[9.9] for _ in ts]
    )

    key = f"cache:embedding:{model}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    ttl = redis_client.ttl(key)
    assert 0 < ttl <= EMBEDDING_CACHE_TTL_SECONDS


def test_different_texts_get_independent_cache_entries(redis_client):
    model = "test-model-multi"
    texts = ["text one", "text two", "text three"]
    calls = []

    def fake_embed(batch: list[str]) -> list[list[float]]:
        calls.append(list(batch))
        return [[float(len(t))] for t in batch]

    first = get_or_compute_embeddings(
        texts, model=model, redis_client=redis_client, embed_fn=fake_embed
    )
    assert first == [[len(t)] for t in texts]
    assert calls == [texts]  # one batch call, all misses

    # Now mix: one previously-cached text plus one brand-new text. Only the
    # new text should trigger a real embed call.
    calls.clear()
    mixed = ["text one", "a brand new text"]
    second = get_or_compute_embeddings(
        mixed, model=model, redis_client=redis_client, embed_fn=fake_embed
    )
    assert second[0] == [len("text one")]
    assert second[1] == [len("a brand new text")]
    assert calls == [["a brand new text"]]

    for key in redis_client.scan_iter(match=f"cache:embedding:{model}:*"):
        redis_client.delete(key)


def test_empty_texts_list_returns_empty_without_touching_redis_or_embed_fn(redis_client):
    def fake_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_fn must not be called for an empty texts list")

    result = get_or_compute_embeddings(
        [], model="test-model-empty", redis_client=redis_client, embed_fn=fake_embed
    )
    assert result == []


def test_redis_outage_degrades_to_direct_embed_fn_instead_of_raising():
    """A real, genuinely unreachable Redis endpoint -- not a mock -- proves
    the degrade-on-outage path actually works: this is a cache, not a hard
    dependency, so a real `redis.exceptions.ConnectionError` from a dead
    Redis must never propagate out of `get_or_compute_embeddings` and take
    down retrieval/ingestion. Port 1 on loopback refuses connections
    immediately (no listener can bind under 1024 without privilege), so this
    fails fast without needing a mock or a long timeout.
    """
    unreachable_client = redis.Redis(
        host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1
    )
    calls = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[42.0] for _ in texts]

    result = get_or_compute_embeddings(
        ["some text", "another text"],
        model="test-model-outage",
        redis_client=unreachable_client,
        embed_fn=fake_embed,
    )

    # The real embed function was actually called (proving this isn't just a
    # swallowed exception returning garbage) and its real output is what
    # comes back, in order.
    assert result == [[42.0], [42.0]]
    assert calls == [["some text", "another text"]]


def test_redis_write_back_failure_still_returns_computed_embeddings():
    """If the read (MGET) succeeds against a real, reachable Redis but the
    write-back pipeline fails (simulated here via a client that reads from
    the real dev Redis but whose `pipeline()` is swapped for one pointed at
    an unreachable endpoint), the real freshly-computed embeddings from
    `embed_fn` must still be returned -- only the cache write is skipped.
    """
    real_client = redis.Redis.from_url(REDIS_URL)
    real_client.ping()

    class _BrokenPipelineClient:
        """Wraps a real, working client for `mget` but returns a pipeline
        bound to an unreachable Redis for `pipeline()` -- a real failure
        surface (Redis died between the read and the write), not a mock of
        `get_or_compute_embeddings`'s own logic.
        """

        def mget(self, keys):
            return real_client.mget(keys)

        def pipeline(self):
            return redis.Redis(
                host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1
            ).pipeline()

    calls = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[7.0] for _ in texts]

    result = get_or_compute_embeddings(
        ["text never cached before, write-back test"],
        model="test-model-writeback",
        redis_client=_BrokenPipelineClient(),
        embed_fn=fake_embed,
    )

    assert result == [[7.0]]
    assert calls == [["text never cached before, write-back test"]]
