"""Redis-backed embedding cache (Phase 12 Task 1, `docs/superpowers/plans/
2026-09-29-phase12-caching.md`'s cross-cutting design section and Task 1
spec).

Reuses the existing shared `redis` service (`infra/docker-compose.yml`, host
port 6380 -- already used by `ingestion_worker.py`'s Arq job queue), not a
new Redis instance, per that section's explicit reuse decision: cache keys
are namespaced (`cache:embedding:...`) and the cached data (embedding
vectors of arbitrary text) has no confidentiality requirement stricter than
what Arq's own job payloads already put through the same container.

Authz-scope note: unlike the retrieval/response/semantic caches later tasks
in this phase add, an embedding cache entry has NO authz dimension at all --
the embedding vector of a fixed string is the same for every caller and
reveals nothing an unauthorized party couldn't already produce themselves by
embedding the same public string via Ollama directly. So the key is
globally shareable, unscoped, exactly as the plan doc's cross-cutting design
section specifies.

Dependency-direction note: `services/retrieval` (`embedder.py`) must never
import Redis or `resolvegrid_api` -- confirmed by reading `embedder.py`
first (per this task's instructions): `embed_texts` is a bare, directly
-called function (not a DI closure like `CompleteFn`/`RetrieveFn`
elsewhere in this codebase), so the cache wraps it at each real `apps/api`
call site (`agent_retrieval.py`'s query-time embed, `ingestion.py`'s
ingestion-time embed) via `get_or_compute_embeddings`'s injected `embed_fn`
parameter, rather than threading a new cache parameter through
`services/retrieval` itself.
"""

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator

import redis
from opentelemetry import metrics

# Same env var name/default `ingestion_worker.py` already uses for Arq's
# connection (`infra/docker-compose.yml`'s `redis` service, "6380:6379").
# Kept as its own module-level constant rather than imported from
# `ingestion_worker` -- that module documents (see its own module docstring)
# that it re-declares this rather than importing `DATABASE_URL`-style from a
# shared module, and this cache module has no other reason to depend on
# `ingestion_worker`.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# `nomic-embed-text` is the only embedding model this codebase uses
# (confirmed via `embedder.py`'s `DEFAULT_EMBEDDING_MODEL` and
# `ingestion_worker.py`'s `EMBEDDING_MODEL` constant -- both point at the
# same string, no other model name appears anywhere in `apps/api` or
# `services/retrieval`). The embedding of a fixed piece of text under a
# fixed model never goes stale the way retrieved KB *content* does (that's
# Task 2's problem, not this one) -- the only reason to ever evict an entry
# early is Redis memory pressure, not correctness. 7 days is long enough
# that any text queried/ingested more than once within a normal work week
# gets the full benefit, short enough that Redis doesn't accumulate
# unbounded history for text nobody ever asks about again (e.g. one-off
# ingested documents from a corpus re-edit that superseded them).
EMBEDDING_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

_redis_client: redis.Redis | None = None
# Guards the check-then-set below against a real race: two threads (e.g. two
# concurrent first requests in the same process, since apps/api serves
# requests off a threadpool -- FastAPI's default sync-endpoint behavior)
# both observing `_redis_client is None` before either assigns it. Without
# this lock that race is real but low-impact (each thread would construct
# its own independently-valid `redis.Redis` instance -- `redis-py` clients
# are just connection-pool handles, not global state -- so the only actual
# cost would be one harmless extra client object, immediately discarded by
# whichever thread loses); the lock is added anyway since it's a one-line
# fix that removes the ambiguity entirely rather than leaving it as an
# accepted-but-unaddressed race.
_redis_client_lock = threading.Lock()


def get_redis_client() -> redis.Redis:
    """A shared, process-wide Redis client, lazily created on first use.

    Mirrors `db.py`'s `session_factory()`/`_engine` pattern: callers outside
    FastAPI's request scope (`agent_retrieval.py`'s `retrieve_for_agent`,
    built once at startup; `ingestion.py`'s `ingest_document`, called from
    Arq worker jobs in a separate process) cannot use a per-request
    `Depends(...)`, so they call this directly instead.
    """
    global _redis_client
    if _redis_client is None:
        with _redis_client_lock:
            if _redis_client is None:  # re-check: another thread may have
                # already built it while this thread was waiting on the lock.
                _redis_client = redis.Redis.from_url(REDIS_URL)
    return _redis_client


def get_redis() -> Iterator[redis.Redis]:
    """FastAPI dependency: `Depends(get_redis)`, mirroring `db.py`'s
    `Depends(get_db)` shape for request-scoped endpoints that need direct
    Redis access.
    """
    yield get_redis_client()


# Real OTel Meter-based counter (not a fabricated counter object) --
# `resolvegrid_telemetry.init_metrics` sets the process-wide MeterProvider
# (called from `main.py`'s lifespan); `metrics.get_meter` here returns a
# proxy meter that binds to that real provider once it's set, exactly like
# `opentelemetry.trace.get_tracer`'s equivalent proxy-tracer behavior --
# safe to create at import time even before `init_metrics` has run.
_meter = metrics.get_meter("resolvegrid_api.cache")
_cache_lookups_counter = _meter.create_counter(
    "cache_lookups_total",
    description="Cache lookups by cache kind and hit/miss result",
)


def _embedding_cache_key(model: str, text: str) -> str:
    """`cache:embedding:{model}:{sha256(text)}` -- the exact key shape the
    plan doc's cross-cutting design section specifies.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"cache:embedding:{model}:{digest}"


def get_or_compute_embeddings(
    texts: list[str],
    *,
    model: str,
    redis_client: redis.Redis,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    """Return one embedding vector per `texts` entry, in the same order,
    serving every entry already cached under `cache:embedding:{model}:
    {sha256(text)}` from Redis and computing (via `embed_fn`, batched into
    one call for every real cache miss) only the entries that aren't.

    `embed_fn` is the injected real embedder (`resolvegrid_retrieval.
    embedder.embed_texts`, or a closure over it that pins `model` -- see
    `agent_retrieval.py`/`ingestion.py`'s call sites) -- this function never
    imports `services/retrieval` itself, keeping the dependency direction
    the plan doc requires (the cache lives in `apps/api`; the library stays
    unaware Redis exists).

    Increments the real `cache_lookups_total{cache="embedding",
    result="hit"|"miss"}` counter once per input text (not once per batch),
    so a caller asking for 3 texts with 1 cache hit and 2 misses records
    exactly 1 hit + 2 misses, matching what actually happened per-text.

    `texts=[]` returns `[]` immediately without any Redis round trip or
    `embed_fn` call, mirroring `embed_texts`'s own empty-input contract.

    Any exception `embed_fn` raises (e.g. `EmbeddingError`) propagates
    unchanged -- callers keep whatever degrade-on-failure behavior they
    already had around the raw `embed_texts` call (e.g. `agent_retrieval.
    py`'s degrade-to-lexical-only-search on `EmbeddingError`).

    Redis outage handling: this is a cache, not a hard dependency -- a real
    Redis outage must never take down retrieval/ingestion, which have zero
    other reason to depend on Redis being up. If the initial `MGET` read
    fails (`redis.exceptions.RedisError` -- covers `ConnectionError`,
    `TimeoutError`, etc.), there is no way to know which entries were
    actually cached, so this degrades to computing every input via
    `embed_fn` directly (recording one `result="error"` lookup per text --
    an outage must stay visible in the same telemetry a hit/miss would be,
    not silently vanish from the metric) and returns those, skipping the
    cache read AND write entirely for this call. If the read succeeds but
    the later pipelined write-back fails (e.g. Redis dies mid-call), the
    already-correctly-computed `embed_fn` results for this call are still
    returned as-is (nothing is lost) -- only the write-back is skipped, so
    a future call simply won't get this entry's benefit yet; the "miss"
    telemetry already recorded above for those entries remains accurate
    (they genuinely were misses), so no separate telemetry is needed for a
    write-back failure specifically.
    """
    if not texts:
        return []

    keys = [_embedding_cache_key(model, text) for text in texts]

    try:
        cached_raw = redis_client.mget(keys)
    except redis.exceptions.RedisError:
        for _ in texts:
            _cache_lookups_counter.add(1, {"cache": "embedding", "result": "error"})
        return embed_fn(texts)

    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    for index, raw in enumerate(cached_raw):
        if raw is not None:
            results[index] = json.loads(raw)
            _cache_lookups_counter.add(1, {"cache": "embedding", "result": "hit"})
        else:
            miss_indices.append(index)
            _cache_lookups_counter.add(1, {"cache": "embedding", "result": "miss"})

    if miss_indices:
        miss_texts = [texts[index] for index in miss_indices]
        computed = embed_fn(miss_texts)
        pipe = redis_client.pipeline()
        for index, vector in zip(miss_indices, computed):
            results[index] = vector
            pipe.set(keys[index], json.dumps(vector), ex=EMBEDDING_CACHE_TTL_SECONDS)
        try:
            pipe.execute()
        except redis.exceptions.RedisError:
            # Write-back failed -- see docstring above: `results` is already
            # fully populated from `computed` by the loop above, so this is
            # a no-op other than not persisting to Redis for next time.
            pass

    return results  # type: ignore[return-value]  # every index is now filled
