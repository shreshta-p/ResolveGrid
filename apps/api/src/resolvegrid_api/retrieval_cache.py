"""Redis-backed retrieval-RESULT cache (Phase 12 Task 2, `docs/superpowers/
plans/2026-09-29-phase12-caching.md`'s cross-cutting design section and
Task 2 spec).

Distinct module from `resolvegrid_api.cache` (Task 1's embedding cache) on
purpose, per this task's own instructions: this codebase's established
convention is one module per cache *concern*, not one grab-bag module for
every cache kind -- the embedding cache has NO authz dimension at all (see
`cache.py`'s module docstring), while this one is entirely ABOUT the authz
dimension (`scope_sig` below), so folding both into one file would blur
exactly the distinction future readers most need to see clearly. This
module still reuses `cache.py`'s Redis client singleton
(`get_redis_client()`) and its shared `cache_lookups_total` OTel counter
(`get_cache_lookups_counter()`) rather than creating parallel ones.

What this caches: the FULL dict `resolvegrid_api.agent_retrieval.
retrieve_for_agent` returns (`{"chunks": [...], "sufficient": bool,
"context_block": str}`) -- a cache hit skips vector_search, lexical_search,
RRF fusion, reranking, status-adjustment, dedup, AND token-budget assembly
entirely, not just the embedding step Task 1 already covers.

Key shape: `cache:retrieval:{kb_version}:{scope_sig}:{sha256(query_text)}`,
exactly as the plan doc's cross-cutting design section specifies.

Authorization-scope signature (the security-critical piece -- see the plan
doc's cross-cutting design section for the full rationale): two different
employees with the identical resolved `AuthzFilter` (e.g. two analysts in
the same department) MUST share a cache entry -- that is the actual
efficiency point of role/department-scoped caching the master plan asks
for, not a bug -- while two employees with different resolved scopes MUST
NEVER share one. `compute_scope_sig` below is the ONLY place this signature
is computed, so every cache key that touches retrieval-authorized content
derives it the same way.

kb_version: `IngestionRun` already gets a new `status="completed"` row
every time the corpus changes for real (Phase 7) -- reusing `select(func.
max(IngestionRun.id)).where(IngestionRun.status == "completed")` as a
version stamp needs no new schema column, and re-ingesting the corpus
organically invalidates every previously-cached entry (old keys are simply
never looked up again under the new kb_version; they expire via TTL rather
than requiring an explicit delete pass -- the plan doc's documented,
accepted stale-handling strategy).
"""

import hashlib
import json

import redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from resolvegrid_api.cache import get_cache_lookups_counter
from resolvegrid_api.models.knowledge import IngestionRun
from resolvegrid_api.retrieval_authz import AuthzFilter

# Retrieved *rankings* can shift even without a kb_version bump -- a rerank/
# dedup/status-adjustment code change (e.g. a future tuning of
# `resolvegrid_retrieval.reranker`/`status_adjustment.py`, or a Document's
# `status` flipping to "superseded" between calls, per `agent_retrieval.py`'s
# own status-adjustment step) doesn't touch `IngestionRun`, so a long TTL
# here would happily keep serving a cached ranking that no longer reflects
# either of those. 2 hours is chosen deliberately, not an arbitrary round
# number: it's long enough that repeated queries within one real
# troubleshooting session or a single workday's worth of `/chat` traffic on
# the same topic get the full benefit (the actual efficiency point this
# cache exists for), while being short enough that a same-day rerank/status
# change or ranking-relevant edit is only ever stale for a bounded couple of
# hours rather than persisting for the embedding cache's 7-day horizon. It
# is deliberately shorter than the embedding cache's TTL (an embedding of
# fixed text never goes stale; a *ranking* of chunks can) and looser than
# Task 3's response cache (15-30 min, per the plan doc, since that one also
# carries the tighter entitlement-staleness bound named in the cross-cutting
# design section) since a stale retrieval ranking alone -- unlike a stale
# *composed answer* -- still only ever surfaces chunks the CURRENT scope_sig
# was already authorized to see; it can be behind on ranking freshness, but
# never on authorization.
RETRIEVAL_CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 hours


def compute_scope_sig(authz_filter: AuthzFilter) -> str:
    """The canonical, deterministic scope signature the plan doc's
    cross-cutting design section defines:

        "unrestricted" if authz_filter.unrestricted
        else "tags:" + ",".join(sorted(authz_filter.allowed_tags))

    Deliberately derived from the real resolved `AuthzFilter`
    (`retrieval_authz.build_authz_filter`'s output), never from a raw
    `principal.employee_id` -- keying on the employee id would be correct
    but would defeat the department/role-scoped cache SHARING the master
    plan explicitly asks for; omitting scope entirely would leak across
    authz boundaries. `sorted(...)` makes the signature independent of
    `allowed_tags`' iteration order (it's a `frozenset`, whose iteration
    order is not guaranteed stable), so two `AuthzFilter`s built from the
    same logical set of tags always produce byte-identical signatures.
    """
    if authz_filter.unrestricted:
        return "unrestricted"
    return "tags:" + ",".join(sorted(authz_filter.allowed_tags))


def get_kb_version(session: Session) -> int:
    """`select(func.max(IngestionRun.id)).where(IngestionRun.status ==
    "completed")`, per the plan doc's cross-cutting design section -- a
    real query against `IngestionRun`, never a hardcoded/assumed value.

    Returns 0 (never `None`) when no ingestion run has ever completed yet
    (a genuinely empty, freshly-migrated database) -- 0 is still a
    perfectly valid, stable version stamp for "nothing has ever been
    ingested", and keeps this function's return type a plain `int` rather
    than forcing every caller to separately handle `int | None`.
    """
    return session.scalar(
        select(func.max(IngestionRun.id)).where(IngestionRun.status == "completed")
    ) or 0


def _retrieval_cache_key(kb_version: int, scope_sig: str, query_text: str) -> str:
    """`cache:retrieval:{kb_version}:{scope_sig}:{sha256(query_text)}` --
    the exact key shape the plan doc's cross-cutting design section
    specifies.
    """
    digest = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    return f"cache:retrieval:{kb_version}:{scope_sig}:{digest}"


def get_or_compute_retrieval(
    *,
    query_text: str,
    kb_version: int,
    scope_sig: str,
    redis_client: redis.Redis,
    compute_fn,
) -> dict:
    """Return the cached `retrieve_for_agent`-shaped dict for
    `(kb_version, scope_sig, query_text)` if present, else call
    `compute_fn()` (the real, expensive hybrid-search/rerank/dedup/budget
    pipeline -- a zero-argument closure, mirroring `agent_retrieval.py`'s
    call site: it already has `session`/`authz_filter`/`query_text` in
    scope, so there's nothing left to inject as a parameter here) and cache
    its result.

    Increments the real, shared `cache_lookups_total{cache="retrieval",
    result="hit"|"miss"|"error"}` counter (`resolvegrid_api.cache.
    get_cache_lookups_counter()` -- the SAME counter object Task 1's
    embedding cache uses, not a second one) exactly once per call.

    Redis-outage handling mirrors Task 1's `get_or_compute_embeddings`
    exactly (same resilience precedent this codebase already established):
    a cache is never allowed to take down `/chat` just because Redis is
    down.
      - If the `GET` itself fails (`redis.exceptions.RedisError`), there is
        no way to know whether this entry was cached, so this degrades to
        calling `compute_fn()` directly, records one `result="error"`
        lookup (an outage must stay visible in telemetry, not silently
        vanish from the metric), and returns the real computed result,
        skipping the cache write entirely for this call.
      - If the read succeeds but is a genuine cache miss, `compute_fn()` is
        called for the real result, which is returned regardless of
        whether the follow-up write-back succeeds; a write-back failure
        (Redis dying mid-call) is swallowed (nothing is lost -- the
        already-correctly-computed real result is still returned; the
        "miss" telemetry already recorded remains accurate since it
        genuinely was a miss).
    """
    key = _retrieval_cache_key(kb_version, scope_sig, query_text)
    counter = get_cache_lookups_counter()

    try:
        raw = redis_client.get(key)
    except redis.exceptions.RedisError:
        counter.add(1, {"cache": "retrieval", "result": "error"})
        return compute_fn()

    if raw is not None:
        counter.add(1, {"cache": "retrieval", "result": "hit"})
        return json.loads(raw)

    counter.add(1, {"cache": "retrieval", "result": "miss"})
    result = compute_fn()
    try:
        redis_client.set(key, json.dumps(result), ex=RETRIEVAL_CACHE_TTL_SECONDS)
    except redis.exceptions.RedisError:
        # Write-back failed -- `result` is already the real, freshly
        # computed value; only persisting it to Redis for next time is
        # skipped. See docstring above.
        pass
    return result
