"""Document-status-aware rank adjustment of reranked candidates (Phase 8
Task 7 Part B).

Problem this exists to fix, confirmed by `docs/EXPERIMENT_REGISTRY.md`'s
Phase 8 Task 6 entry: reranking alone flipped one of three VPN-policy
distractor cases in the golden eval set, ranking the deprecated
`Kestrel VPN Access Policy (v1, deprecated)` chunk above the current
`Kestrel VPN Access Policy (v2)` chunk for a "what client software should
I install" query. Root cause (confirmed by reading the actual scored
chunk text, not guessed): this is not an information gap -- v2's own
chunk text does state the current/legacy distinction -- the cross-encoder
still favored v1 by a thin, non-robust margin (0.989 vs 0.980, both
saturated near the top of the sigmoid range), apparently weighting v1's
exact version-string specificity over v2's in-context recency cue.

`Document.status` (`"active"` | `"superseded"` | `"stale"` | `"restricted"`,
`apps/api/src/resolvegrid_api/models/knowledge.py`) already correctly
labels this exact pair in the live DB (v1 is `"superseded"`, v2 is
`"active"`) and was consulted by nothing in the rerank/dedup/context-
budget pipeline before this task.

Architectural placement, decided (not the first option reached for):
`Document.status` lives in `apps/api`'s Postgres schema -- reading it
requires a DB query, and `services/retrieval` has no DB dependency at
all (see `dedup.py`'s module docstring for the same "no DB dependency in
this package" reasoning, applied there to embedding vectors instead of
`Document.status`). This module follows that exact precedent: it takes
`superseded_chunk_ids` as a plain `frozenset[int]`/`set[int]` the caller
(`apps/api`'s `agent_retrieval.py`/`eval_retrieval.py`, which already
queries `Chunk`/`Document` for text/titles) resolves via a trivial
`Document.status == "superseded"` check on data it already fetched --
mirroring `dedup()`'s "attach data the caller already has, no new data
source" composability convention. This keeps `services/retrieval`
dependency-free for this feature (no new import at all) while keeping
`apps/api` the sole owner of DB access, matching every other module in
this package.

Mechanism, decided: a hard score penalty, not a soft multiplicative
discount. `rerank()`'s output is a sigmoid-activated probability in
`[0, 1]` (see `reranker.py`'s docstring) -- `DEFAULT_SUPERSEDED_PENALTY =
1.0` is subtracted from every superseded candidate's score. Since no
non-superseded candidate's score can exceed `1.0` and no superseded
candidate's *adjusted* score can exceed `0.0`, this is a strict
guarantee: whenever both a superseded and a non-superseded candidate are
present, the non-superseded one always sorts first, regardless of how
thin or wide the original rerank-score gap was. This was chosen over a
smaller/multiplicative penalty (e.g. "halve superseded scores") on
purpose: Task 6's flip was a *thin, non-robust* margin (0.989 vs 0.980,
a 0.009 gap) -- a mitigation tuned only to survive that specific gap
would be exactly the kind of fragile, case-specific fix this task's plan
explicitly warns against ("a real engineering decision, not a rubber-
stamp"). A full-range penalty is correct for *any* margin the
cross-encoder might produce, not just the one measured so far, and costs
nothing extra in code complexity.

This is deprioritization, not filtering: a superseded chunk is never
removed from the candidate list, only pushed below every non-superseded
one in this ranking pass. It can still survive `dedup()` and make it into
the final budgeted context (e.g. if no non-superseded chunk addresses the
same point, or there's budget room after all non-superseded chunks are
included) -- an outdated-but-present chunk with an honest low rank is
strictly better than silently deleting real content the user might still
legitimately need (e.g. "what was the OLD VPN policy"), matching this
whole phase's "erring toward under-collapsing/under-hiding is the safer
failure mode" philosophy already established in `dedup.py`.

Where this sits in the pipeline (decided in Task 7's wiring): after
`rerank()`, before `dedup()`. `dedup()` re-sorts its input by `(-score,
chunk_id)` internally (its own documented "defensive" convention) --
this module's score adjustment must happen *before* `dedup()` sees the
candidates, or `dedup()`'s resort would simply undo any pure
reordering that didn't also change the underlying score. Applying the
penalty directly to the score (rather than just reordering the list) is
what makes this composition correct: `dedup()`'s own resort naturally
preserves this module's intended order, because it sorts by the exact
same (now-penalized) score.
"""

from __future__ import annotations

# See module docstring's "Mechanism, decided" section for why this is a
# full [0, 1]-range-exceeding constant rather than a smaller/multiplicative
# discount -- a superseded candidate's adjusted score can never exceed 0.0,
# and a non-superseded candidate's score (rerank()'s sigmoid output) can
# never be below 0.0, so this guarantees non-superseded always sorts first
# whenever both are present, regardless of the original score gap.
DEFAULT_SUPERSEDED_PENALTY = 1.0


def apply_status_adjustment(
    ranked_candidates: list[tuple[int, str, float]],
    *,
    superseded_chunk_ids: frozenset[int] | set[int],
    penalty: float = DEFAULT_SUPERSEDED_PENALTY,
) -> list[tuple[int, str, float]]:
    """Deprioritize candidates whose document is `superseded`, and
    re-sort `ranked_candidates` (rerank()'s `(chunk_id, text, score)`
    shape with text re-attached, the same input shape `dedup()` takes)
    best-first by the adjusted score.

    Every candidate whose `chunk_id` is in `superseded_chunk_ids` has
    `penalty` subtracted from its score before sorting; every other
    candidate's score is unchanged. Returns all candidates (nothing is
    dropped -- see module docstring's "deprioritization, not filtering"
    section), sorted by `(-adjusted_score, chunk_id)` -- the same
    best-first, deterministic-tie-break convention `rerank()`/`fuse_rrf()`/
    `dedup()` all already use.

    `ranked_candidates=[]` returns `[]`. `superseded_chunk_ids=frozenset()`
    (no superseded candidates in this batch) is a no-op re-sort by the
    original scores -- safe and cheap to call unconditionally, no need for
    a caller to special-case "no superseded chunks in this batch."
    """
    adjusted = [
        (
            chunk_id,
            text,
            score - penalty if chunk_id in superseded_chunk_ids else score,
        )
        for chunk_id, text, score in ranked_candidates
    ]
    return sorted(adjusted, key=lambda c: (-c[2], c[0]))
