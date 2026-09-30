"""Redis-backed exact-response cache for `/chat` (Phase 12 Task 3, `docs/
superpowers/plans/2026-09-29-phase12-caching.md`'s cross-cutting design
section and this task's own, AMENDED section -- read that amendment before
this module: the plan doc's originally-written key shape included
`risk_level`, which is wrong (it's an OUTPUT of `classify_intent`, a node
INSIDE the graph a cache hit must skip entirely, so it cannot be part of a
key computed BEFORE the graph runs). The corrected key shape below drops
it; `docs/DECISION_LOG.md`'s 2026-09-30 entry records the full reasoning.

Distinct module from `resolvegrid_api.cache` (Task 1's embedding cache) and
`resolvegrid_api.retrieval_cache` (Task 2's retrieval-result cache), per
this codebase's established one-module-per-cache-*concern* convention (see
`retrieval_cache.py`'s own docstring for why that convention exists). This
module reuses `retrieval_cache.py`'s `compute_scope_sig`/`get_kb_version`
directly (imported, never reimplemented -- both caches share the identical
authz-scope-signature and KB-version-stamp mechanisms, per the plan doc's
cross-cutting design section) and `cache.py`'s shared Redis client
singleton/`cache_lookups_total` OTel counter, exactly like `retrieval_cache.
py` already does.

Key shape (corrected): `cache:response:{kb_version}:{scope_sig}:
{prompt_template_version}:{sha256(message_text)}`. `prompt_template_version`
is `resolvegrid_agent_orchestration.PROMPT_TEMPLATE_VERSION` (Phase 12
Task 3's addition to `graph.py`) -- a future edit to either compose-prompt
template organically busts every previously-cached answer instead of
silently continuing to serve one worded against a template that no longer
exists.

Cached value shape: `{"answer": str, "citations": list[dict], "caption":
str, "intent": str, "risk_level": str}` -- the full response payload
`routers/chat.py` returns to callers PLUS the classified `intent`/
`risk_level` (per the key-shape correction above: these ride along in the
VALUE, computed once by the real graph on a miss, so a later hit can read
them back -- e.g. to derive the cache-hit `ModelCall` row's `model`/
`routing_reason` via the exact same `select_model_for_risk_level` policy a
real call would have used -- without ever needing them to perform the
lookup itself).

Two separate functions (`lookup_cached_response`/`store_cached_response`),
NOT one `get_or_compute_response(..., compute_fn)` wrapper like Task 2's
`get_or_compute_retrieval` -- a deliberate, small deviation from that
task's exact shape, not an inconsistency: `routers/chat.py`'s real miss
path is not a small, swappable "compute the value" closure the way
`retrieve_for_agent`'s pipeline is for Task 2 -- it's the entire existing
`chat()` handler body (a real `AgentRun`/four-`Span`-row write, exception
handling that itself writes an error-shaped `AgentRun` row, real per-node
graph timing, citation/caption assembly). Folding all of that into a
`compute_fn` closure passed into a generic wrapper would force that
substantial, already-tested control flow to live awkwardly inside a
one-line closure instead of the router where it belongs, for no real
benefit -- `chat()` already has exactly two points where cache behavior
needs to hook in (a GET right after `authz_filter`/`kb_version`/`scope_sig`
are resolved, and a SET right before the final successful return), so two
plain functions map onto those two points directly. The real hit/miss/
error telemetry and Redis-outage-degrade CONVENTIONS are still followed
exactly (see each function's docstring below) -- only the single-call
shape differs.

TTL: 15 minutes (900s) -- the SHORT end of the plan doc's stated 15-30
minute range, chosen deliberately, not split down the middle. This is the
one cache in this phase where the cross-cutting design section's
"entitlement/role-change staleness" bound genuinely matters (Task 2's
retrieval cache is bounded by ranking freshness instead, and Task 1's
embedding cache has no authz dimension at all to go stale) -- if an admin
revokes a role grant, a cached response entry keyed by that employee's OLD
`scope_sig` remains servable to them until it expires. Erring toward the
tighter end of the stated range favors closing that real security-relevant
staleness window sooner over the marginal extra cache-hit-rate the looser
end would buy; 15 minutes is still long enough to cover the actual repeat
-question pattern this cache exists for (a user asking the same/an
identically-worded question again within one short troubleshooting
session) without meaningfully undermining the point of caching at all.
"""

import hashlib
import json

import redis

from resolvegrid_api.cache import get_cache_lookups_counter

# Deliberately SHORTER than Task 2's `RETRIEVAL_CACHE_TTL_SECONDS` (2 hours)
# and far shorter than Task 1's `EMBEDDING_CACHE_TTL_SECONDS` (7 days) -- see
# module docstring for why THIS cache specifically carries the tighter
# entitlement-staleness bound the plan doc's cross-cutting design section
# names.
RESPONSE_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


def _response_cache_key(
    kb_version: int, scope_sig: str, prompt_template_version: str, message_text: str
) -> str:
    """`cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:
    {sha256(message_text)}` -- the corrected key shape (see module
    docstring's amendment note): `risk_level` is deliberately NOT part of
    this key.
    """
    digest = hashlib.sha256(message_text.encode("utf-8")).hexdigest()
    return f"cache:response:{kb_version}:{scope_sig}:{prompt_template_version}:{digest}"


def lookup_cached_response(
    *,
    message_text: str,
    kb_version: int,
    scope_sig: str,
    prompt_template_version: str,
    redis_client: redis.Redis,
) -> dict | None:
    """Real Redis `GET` only -- returns the cached value dict on a real
    hit, `None` on a real miss OR a real Redis outage (this is a cache,
    never a hard dependency: a `redis.exceptions.RedisError` here must
    never take down `/chat` -- it degrades to `None`, which `routers/chat.
    py`'s caller treats identically to a genuine miss, i.e. it runs the
    real graph, exactly the "never breaks /chat, falls through to running
    the real graph" behavior this task requires).

    Increments the real, shared `cache_lookups_total{cache="response",
    result="hit"|"miss"|"error"}` counter (`resolvegrid_api.cache.
    get_cache_lookups_counter()` -- the SAME counter object Tasks 1/2 use)
    exactly once per call, mirroring `retrieval_cache.get_or_compute_
    retrieval`'s identical telemetry convention.
    """
    key = _response_cache_key(kb_version, scope_sig, prompt_template_version, message_text)
    counter = get_cache_lookups_counter()

    try:
        raw = redis_client.get(key)
    except redis.exceptions.RedisError:
        counter.add(1, {"cache": "response", "result": "error"})
        return None

    if raw is not None:
        counter.add(1, {"cache": "response", "result": "hit"})
        return json.loads(raw)

    counter.add(1, {"cache": "response", "result": "miss"})
    return None


def store_cached_response(
    *,
    message_text: str,
    kb_version: int,
    scope_sig: str,
    prompt_template_version: str,
    redis_client: redis.Redis,
    value: dict,
) -> None:
    """Real Redis `SET ... EX RESPONSE_CACHE_TTL_SECONDS` only -- called by
    `routers/chat.py` after a real graph run completes successfully, to
    populate the entry a future identical `(kb_version, scope_sig,
    message_text)` lookup will find.

    A write-back failure (`redis.exceptions.RedisError` -- Redis dies
    between the earlier `lookup_cached_response` miss and this call) is
    swallowed, mirroring Task 1/2's identical write-back-failure handling:
    nothing real is lost -- the caller already has its real, freshly
    -computed response and returns it regardless; only this call's future
    cache benefit is skipped, exactly as if this call had simply lost the
    race to populate the entry.
    """
    key = _response_cache_key(kb_version, scope_sig, prompt_template_version, message_text)
    try:
        redis_client.set(key, json.dumps(value), ex=RESPONSE_CACHE_TTL_SECONDS)
    except redis.exceptions.RedisError:
        pass
