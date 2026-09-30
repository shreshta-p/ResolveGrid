"""Tests for `resolvegrid_api.response_cache` (Phase 12 Task 3: exact
-response cache for `/chat`).

Real Redis throughout, matching this codebase's established philosophy
(`test_cache.py`/`test_retrieval_cache.py`'s module docstrings, Tasks 1/2)
-- nothing here mocks Redis away. These tests exercise the module's own
GET/SET plumbing (key shape, hit/miss, TTL, cross-key isolation, real
-outage degrade); `test_chat_api.py`'s cache tests separately exercise the
full real `/chat` integration (real graph skip, real AgentRun/Span/
ModelCall row shapes).
"""

import hashlib
import json

import pytest
import redis

from resolvegrid_api.cache import REDIS_URL
from resolvegrid_api.response_cache import (
    RESPONSE_CACHE_TTL_SECONDS,
    lookup_cached_response,
    store_cached_response,
)


def _clear_response_cache_namespace(client: redis.Redis) -> None:
    for key in client.scan_iter(match="cache:response:*"):
        client.delete(key)


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fail fast (real connection check) if Redis isn't up
    _clear_response_cache_namespace(client)
    yield client
    _clear_response_cache_namespace(client)


_SAMPLE_VALUE = {
    "answer": "A VPN is a Virtual Private Network.",
    "citations": [{"chunk_id": 1, "document_title": "What Is a VPN"}],
    "caption": "General-knowledge answer",
    "intent": "general_question",
    "risk_level": "low",
}


def test_cache_key_shape(redis_client):
    """`cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:
    {sha256(message_text)}` -- the CORRECTED key shape (no `risk_level`
    component -- see module docstring's amendment note).
    """
    message_text = "what is a firewall?"
    kb_version = 3
    scope_sig = "unrestricted"
    prompt_template_version = "1"

    store_cached_response(
        message_text=message_text, kb_version=kb_version, scope_sig=scope_sig,
        prompt_template_version=prompt_template_version, redis_client=redis_client,
        value=_SAMPLE_VALUE,
    )

    digest = hashlib.sha256(message_text.encode("utf-8")).hexdigest()
    expected_key = f"cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:{digest}"
    raw = redis_client.get(expected_key)
    assert raw is not None
    assert json.loads(raw) == _SAMPLE_VALUE


def test_lookup_is_a_real_miss_before_any_store(redis_client):
    result = lookup_cached_response(
        message_text="never stored before, response cache miss test",
        kb_version=1, scope_sig="unrestricted", prompt_template_version="1",
        redis_client=redis_client,
    )
    assert result is None


def test_store_then_lookup_is_a_real_hit_with_the_full_value(redis_client):
    kwargs = dict(
        message_text="what is a VPN?", kb_version=5, scope_sig="tags:it_support",
        prompt_template_version="1",
    )
    store_cached_response(**kwargs, redis_client=redis_client, value=_SAMPLE_VALUE)
    result = lookup_cached_response(**kwargs, redis_client=redis_client)
    assert result == _SAMPLE_VALUE
    # The classified intent/risk_level ride along in the VALUE (per this
    # task's key-shape correction), not just answer/citations/caption.
    assert result["intent"] == "general_question"
    assert result["risk_level"] == "low"


def test_ttl_is_set_on_written_key(redis_client):
    kwargs = dict(
        message_text="ttl-checking message", kb_version=0, scope_sig="unrestricted",
        prompt_template_version="1",
    )
    store_cached_response(**kwargs, redis_client=redis_client, value=_SAMPLE_VALUE)
    digest = hashlib.sha256(kwargs["message_text"].encode("utf-8")).hexdigest()
    key = f"cache:response:0:unrestricted:1:{digest}"
    ttl = redis_client.ttl(key)
    assert 0 < ttl <= RESPONSE_CACHE_TTL_SECONDS


def test_ttl_is_the_documented_15_minutes_short_bound():
    """Real, specific number this task's docstring commits to -- not just
    "some short TTL" -- since the entitlement-staleness bound it names
    only means something concrete if the number is pinned down and
    regression-tested.
    """
    assert RESPONSE_CACHE_TTL_SECONDS == 15 * 60


def test_different_kb_versions_get_independent_entries(redis_client):
    base_kwargs = dict(
        message_text="same message, different kb_version",
        scope_sig="unrestricted", prompt_template_version="1",
    )
    store_cached_response(**base_kwargs, kb_version=1, redis_client=redis_client, value={"answer": "v1"})
    store_cached_response(**base_kwargs, kb_version=2, redis_client=redis_client, value={"answer": "v2"})

    assert lookup_cached_response(**base_kwargs, kb_version=1, redis_client=redis_client) == {"answer": "v1"}
    assert lookup_cached_response(**base_kwargs, kb_version=2, redis_client=redis_client) == {"answer": "v2"}


def test_different_scope_sigs_get_independent_entries(redis_client):
    """The security-critical isolation dimension: two different scope_sig
    values (as if two employees with genuinely different resolved
    AuthzFilters) must never collide, even for the identical message and
    kb_version.
    """
    base_kwargs = dict(
        message_text="same message, different scope",
        kb_version=1, prompt_template_version="1",
    )
    store_cached_response(
        **base_kwargs, scope_sig="tags:it_support", redis_client=redis_client,
        value={"answer": "IT Support answer"},
    )
    store_cached_response(
        **base_kwargs, scope_sig="tags:finance", redis_client=redis_client,
        value={"answer": "Finance answer"},
    )

    assert lookup_cached_response(
        **base_kwargs, scope_sig="tags:it_support", redis_client=redis_client
    ) == {"answer": "IT Support answer"}
    assert lookup_cached_response(
        **base_kwargs, scope_sig="tags:finance", redis_client=redis_client
    ) == {"answer": "Finance answer"}


def test_different_prompt_template_versions_get_independent_entries(redis_client):
    """Proves a future compose-prompt wording change (bumping
    `PROMPT_TEMPLATE_VERSION`) organically busts old entries instead of
    serving a stale-worded cached answer.
    """
    base_kwargs = dict(message_text="same message, different template version", kb_version=1, scope_sig="unrestricted")
    store_cached_response(
        **base_kwargs, prompt_template_version="1", redis_client=redis_client, value={"answer": "old wording"}
    )
    assert lookup_cached_response(
        **base_kwargs, prompt_template_version="2", redis_client=redis_client
    ) is None
    assert lookup_cached_response(
        **base_kwargs, prompt_template_version="1", redis_client=redis_client
    ) == {"answer": "old wording"}


def test_different_message_texts_get_independent_entries(redis_client):
    common_kwargs = dict(kb_version=1, scope_sig="unrestricted", prompt_template_version="1")
    store_cached_response(
        message_text="first message", **common_kwargs, redis_client=redis_client, value={"answer": "first"}
    )
    store_cached_response(
        message_text="second message", **common_kwargs, redis_client=redis_client, value={"answer": "second"}
    )
    assert lookup_cached_response(
        message_text="first message", **common_kwargs, redis_client=redis_client
    ) == {"answer": "first"}
    assert lookup_cached_response(
        message_text="second message", **common_kwargs, redis_client=redis_client
    ) == {"answer": "second"}


def test_lookup_redis_outage_degrades_to_none_instead_of_raising():
    """A real, genuinely unreachable Redis endpoint -- not a mock --
    proves the degrade-on-outage path actually works: a real
    `redis.exceptions.ConnectionError` from a dead Redis must never
    propagate out of `lookup_cached_response` (which would take down
    `/chat`). Port 1 on loopback refuses connections immediately (no
    listener can bind under 1024 without privilege), so this fails fast
    without needing a mock or a long timeout -- mirrors `test_cache.py`'s
    identical real-outage proof for Task 1.
    """
    unreachable_client = redis.Redis(
        host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1
    )
    result = lookup_cached_response(
        message_text="doesn't matter, redis is unreachable",
        kb_version=1, scope_sig="unrestricted", prompt_template_version="1",
        redis_client=unreachable_client,
    )
    assert result is None


def test_store_redis_outage_is_swallowed_without_raising():
    """A write-back failure (real unreachable Redis) must never raise out
    of `store_cached_response` -- `routers/chat.py` calls this right
    before returning a real, already-computed response; a raised
    exception here would turn a cache-population failure into a broken
    `/chat` response, which this cache must never do.
    """
    unreachable_client = redis.Redis(
        host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1
    )
    store_cached_response(
        message_text="doesn't matter, redis is unreachable",
        kb_version=1, scope_sig="unrestricted", prompt_template_version="1",
        redis_client=unreachable_client, value=_SAMPLE_VALUE,
    )  # must not raise
