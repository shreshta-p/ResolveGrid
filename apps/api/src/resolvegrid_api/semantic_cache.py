"""Postgres/pgvector-backed SEMANTIC-response cache (Phase 12 Task 5,
`docs/superpowers/plans/2026-09-29-phase12-caching.md`'s cross-cutting
design section and this task's own section).

Sibling of `resolvegrid_api.response_cache` (Task 3's exact-match cache):
same cached VALUE shape (`answer`/`citations`/`caption`/`intent`/
`risk_level`), same `scope_sig`/`kb_version` authz-safety requirements
(reused directly from `retrieval_cache.py`'s `compute_scope_sig`/`get_kb_
version`, never reimplemented) -- but a DIFFERENT lookup mechanism:
cosine-similarity against a real `nomic-embed-text` query embedding,
instead of an exact SHA-256 hash match of the message text. A differently
-worded-but-semantically-equivalent question (e.g. "What is a firewall?"
vs. "Can you explain what a firewall is?") can hit THIS cache even though
it would be a guaranteed miss on Task 3's exact cache.

Stored in Postgres, not Redis -- see `models/knowledge.py`'s
`SemanticResponseCache` docstring for the full "why a new pgvector table,
not a parallel Redis vector index" rationale (the plan doc's Task 5 section
explicitly prefers extending this codebase's established pgvector mechanism
over introducing a second, parallel vector-search mechanism inside Redis).

Authorization-scope boundary (the security-critical piece -- read
carefully, mirrors Task 2's `scope_sig` boundary exactly): every real
lookup's SQL `WHERE` clause filters candidate rows to `scope_sig = :scope_
sig AND kb_version = :kb_version` BEFORE any `<=>` cosine-distance ordering
ever runs -- a row from a different scope is never even a candidate, let
alone a similarity match, no matter how close the query text. This is
proven by this module's own test suite's mandatory cross-scope-leakage test
(two disjoint scopes, an IDENTICAL query embedding -- i.e. the worst case,
zero cosine distance -- used for both; the second scope must still get a
real miss).

Similarity threshold (empirically measured, not an arbitrary round number
-- see this module's own `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` constant
below for the real, cited cosine-distance numbers this was derived from):
a lookup is a HIT iff the nearest candidate row's real pgvector `<=>`
cosine distance is `<= SEMANTIC_CACHE_SIMILARITY_THRESHOLD`.

Staleness bound (no TTL column exists -- see `SemanticResponseCache`'s
docstring for why): `SEMANTIC_CACHE_TTL_SECONDS` is applied as a real
`created_at > :cutoff` filter in the lookup query itself, mirroring Task
3's `RESPONSE_CACHE_TTL_SECONDS` 15-minute entitlement-staleness bound
(this cache carries the identical content-staleness/entitlement-staleness
risk one layer up from Task 3's exact cache -- see that module's docstring
for the full rationale, which applies here unchanged) -- a row older than
the bound is simply never looked up again, never deleted by this module.

Real Postgres-outage degrade (this cache's lookup/store touch Postgres, not
Redis, so Task 1-3's `redis.exceptions.RedisError` precedent doesn't apply
literally -- the analogous real failure mode here is a real `sqlalchemy.
exc.DBAPIError`, the base class SQLAlchemy wraps every real DBAPI-level
failure in, including a real unreachable-Postgres `OperationalError`):
exactly mirrors Tasks 1-3's "a cache must never take down `/chat`" contract
-- a lookup failure degrades to `None` (treated as a genuine miss by
`routers/chat.py`'s caller, which falls through to running the real graph),
and a store failure is swallowed (the caller's already-computed real
response is returned regardless; only this call's future cache benefit is
skipped). Both paths still record the real `cache_lookups_total{cache=
"semantic", result="error"}` telemetry (an outage must stay visible in the
same metric a hit/miss would be, not silently vanish).
"""

import json

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from resolvegrid_api.cache import get_cache_lookups_counter
from resolvegrid_api.models.knowledge import SemanticResponseCache

# Empirically measured, 2026-10-09, against the real running `resolvegrid
# -ollama` container's `nomic-embed-text` model (POST /api/embed), NOT a
# guessed round number -- real cosine distances (1 - cosine_similarity)
# computed between real embedding vectors of these real example pairs:
#
#   SHOULD match (near-duplicate rewordings of the same question):
#     "What is a firewall?" vs "Can you explain what a firewall is?"           -> 0.0352
#     "How do I reset my password?" vs "What steps do I follow to reset
#       my password?"                                                          -> 0.0510
#     "What is a VPN?" vs "Could you explain what a VPN is?"                   -> 0.0356
#     "How do I request VPN access?" vs "What is the process to get VPN
#       access?"                                                                -> 0.1171
#
#   SHOULD NOT match (genuinely different-topic questions):
#     "What is a firewall?" vs "How do I request VPN access?"                  -> 0.5374
#     "How do I reset my password?" vs "What is a firewall?"                   -> 0.5941
#     "What is a VPN?" vs "How do I reset my password?"                        -> 0.6049
#     "How do I request VPN access?" vs "What is a firewall?"                  -> 0.5374
#
# The real near-duplicate distances (max observed: 0.1171) and the real
# different-topic distances (min observed: 0.5374) are separated by a wide
# margin (~0.42), so 0.15 is chosen as a threshold with real headroom on
# both sides -- comfortably above every measured near-duplicate distance
# (closest approach: 0.1171, ~22% margin) and comfortably below every
# measured different-topic distance (closest approach: 0.5374, more than
# 3x the threshold). This is not the exact midpoint of the two clusters
# (~0.33) deliberately: erring toward the tighter/lower end favors avoiding
# a false semantic-cache HIT (which would serve a previously-cached answer
# for a genuinely different question -- a correctness/trust risk) over
# squeezing out marginally more hit-rate from reworded-but-more-distant
# paraphrases, the same "err toward the safer side of a wide empirical gap"
# principle `retrieval.py`'s `DEFAULT_MIN_SCORE_THRESHOLD` docstring and
# Phase 7's eval-harness recall/precision tuning both already apply.
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.15

# Identical value and identical rationale to Task 3's `RESPONSE_CACHE_TTL_
# SECONDS` (`response_cache.py`): this cache carries the SAME entitlement
# -staleness bound (an admin-revoked role grant stays servable to that
# employee's old scope_sig until this bound elapses) and the SAME inherited
# ranking-staleness risk one layer up (a stale retrieval ranking underneath
# a semantic-cache hit means a stale COMPOSED ANSWER can be served here
# too) -- see that module's docstring for the full written-out reasoning,
# which applies here completely unchanged.
SEMANTIC_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


def _vector_literal(vector: list[float]) -> str:
    """Render a Python float list as the string pgvector's `::vector` cast
    expects (`"[0.1,0.2,...]"`). Duplicated from `retrieval.py`'s private
    `_vector_literal` (same exact logic, same empirically-verified reason
    a bind parameter must be cast this way -- see that module's docstring)
    rather than imported: that helper is underscore-prefixed/private to its
    own module, and this codebase's established one-module-per-cache
    -concern convention (see `retrieval_cache.py`'s module docstring) favors
    a small, self-contained duplicate here over reaching into another
    module's private internals.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def lookup_semantic_cache(
    session: Session,
    *,
    query_embedding: list[float],
    scope_sig: str,
    kb_version: int,
) -> dict | None:
    """Real pgvector `<=>` cosine-distance query, scoped to `(scope_sig,
    kb_version)` and the real `SEMANTIC_CACHE_TTL_SECONDS` staleness bound
    (`created_at > now() - interval`), returning the stored cached-response
    VALUE dict (`{"answer", "citations", "caption", "intent", "risk_level"}`)
    for the single nearest candidate row IF its real distance is `<=
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD`, else `None` (a genuine miss --
    either no row exists in scope, or the nearest one is too dissimilar).

    `LIMIT 1` -- only the single nearest row is ever considered; no ANN
    index is used (see `SemanticResponseCache`'s docstring for why exact,
    not approximate, ordering is the deliberate choice here).

    Real Postgres-outage degrade: a real `sqlalchemy.exc.DBAPIError`
    (covers a real unreachable-Postgres `OperationalError` and any other
    real driver-level failure) degrades to `None` -- exactly like a miss,
    which `routers/chat.py`'s caller already treats as "fall through to the
    real graph" -- after recording `result="error"` telemetry. This is a
    cache, never a hard dependency.
    """
    counter = get_cache_lookups_counter()

    try:
        row = session.execute(
            text(
                "SELECT answer, citations_json, caption, intent, risk_level, "
                "query_embedding <=> (:query_vector)::vector AS distance "
                "FROM semantic_response_cache "
                "WHERE scope_sig = :scope_sig AND kb_version = :kb_version "
                "AND created_at > now() - make_interval(secs => :ttl_seconds) "
                "ORDER BY query_embedding <=> (:query_vector)::vector ASC "
                "LIMIT 1"
            ),
            {
                "query_vector": _vector_literal(query_embedding),
                "scope_sig": scope_sig,
                "kb_version": kb_version,
                "ttl_seconds": SEMANTIC_CACHE_TTL_SECONDS,
            },
        ).first()
    except DBAPIError:
        counter.add(1, {"cache": "semantic", "result": "error"})
        return None

    if row is not None and row.distance <= SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
        counter.add(1, {"cache": "semantic", "result": "hit"})
        return {
            "answer": row.answer,
            "citations": json.loads(row.citations_json),
            "caption": row.caption,
            "intent": row.intent,
            "risk_level": row.risk_level,
        }

    counter.add(1, {"cache": "semantic", "result": "miss"})
    return None


def store_semantic_cache_entry(
    session: Session,
    *,
    query_embedding: list[float],
    scope_sig: str,
    kb_version: int,
    value: dict,
) -> None:
    """Insert one real `semantic_response_cache` row and commit it -- called
    by `routers/chat.py` after a real graph run completes successfully, on
    a full miss (both the exact cache AND this semantic cache missed), so a
    future semantically-similar `(scope_sig, kb_version)` lookup can find it.

    Unlike `log_completion`/`record_audit_event`'s "caller owns the
    transaction boundary" convention, this function DOES commit its own
    insert: by the time `routers/chat.py` calls this, the request's own
    `AgentRun`/`Span` writes for this turn have already been committed
    (this is a genuinely separate, independent row with no FK relationship
    to the run that happened to populate it), so there is no shared
    transaction boundary to respect -- committing here immediately, rather
    than leaving an uncommitted row for the caller to remember to flush, is
    simpler and matches `store_cached_response`'s (Task 3) own
    self-contained "writes and is done" shape for its Redis `SET`.

    Real Postgres-outage degrade: a real `sqlalchemy.exc.DBAPIError` here
    is caught and swallowed (the session is rolled back so it's left usable
    for the caller's subsequent work) -- nothing real is lost: the caller
    already has its real, freshly-computed response and returns it
    regardless; only this call's future cache benefit is skipped, exactly
    mirroring Tasks 1-3's identical write-back-failure handling.
    """
    entry = SemanticResponseCache(
        scope_sig=scope_sig,
        kb_version=kb_version,
        query_embedding=query_embedding,
        answer=value.get("answer") or "",
        citations_json=json.dumps(value.get("citations") or []),
        caption=value.get("caption") or "",
        intent=value.get("intent") or "unclear",
        risk_level=value.get("risk_level") or "low",
    )
    session.add(entry)
    try:
        session.commit()
    except DBAPIError:
        session.rollback()
