"""Deterministic citation verification (Phase 8 Task 4).

Scope note: this module is a standalone, testable capability, same as
`reranker.py`/`dedup.py`/`context_budget.py` before it. It is
deliberately **not** wired into the agent graph
(`services/agent-orchestration`), `apps/api`, or `/chat` yet -- that is
Phase 8 Task 7. `compose_response`'s prompt template and the actual
`finalize`/routing consequence of a failed verification are read-only
investigation inputs here, not things this module or task changes.

Citation format, confirmed against the real prompt template (not
assumed) -- `services/agent-orchestration/.../graph.py`,
`_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE`:

    "...cite it inline using that exact bracketed id (for example:
    '...per the VPN policy [chunk:123].')"

and `_build_context_block`, which labels every context entry the model
is shown with exactly `[chunk:<id>] (from "<title>"):`. So the model is
instructed to reproduce the literal token `[chunk:<digits>]` -- lowercase
`chunk`, a single colon, no space, digits only for the id, e.g.
`[chunk:123]`. `_CITATION_PATTERN` below matches exactly that literal
form. This is a deliberate choice, not an oversight: a differently-cased
variant (`[Chunk:123]`) or a malformed one (`[chunk 123]`, `[chunk:abc]`)
is not a citation the model was ever instructed to produce, so treating
it as "not a citation" (silently ignored, not counted as verified or
fabricated) is the correct behavior for a *deterministic* verifier
checking compliance with a known, fixed instruction -- not a parser bug
to be made more lenient. (If real-world model output drift away from the
exact instructed format becomes an observed problem, that is a prompt-
engineering/model-behavior finding for a later task, not a reason to
loosen this regex to guess at intent.)

Return shape, decided (not a bare bool) -- per this task's brief, quoting
the plan doc's stage 11/12 language ("citation verification... safe
answer, abstention, or escalation"): a caller making a real routing
decision needs to know *which* citations are fabricated (to strip just
those, or to decide the whole answer is untrustworthy), not just whether
verification passed overall. So `verify_citations` returns a
`VerificationResult` with:

  - `citations`: every parsed `Citation` occurrence, in the order it
    appears in `answer_text`, each carrying `chunk_id`, whether it
    `verified`, and its exact matched span (`start`/`end` character
    offsets into `answer_text`, plus the raw matched substring) -- enough
    for a caller to do exact in-place string surgery (e.g. strip only the
    fabricated citation markers, leaving the surrounding prose and valid
    citations untouched) rather than a blunt whole-answer reject.
  - `all_verified`: `True` iff every parsed citation was verified.
    Vacuously `True` when there are zero citations at all -- a
    general-knowledge answer citing nothing is a valid, unremarkable
    case per this task's brief, not a failure (there is nothing to have
    gotten wrong).
  - `verified_chunk_ids`/`fabricated_chunk_ids`: unique chunk ids across
    all occurrences, in order of first appearance -- a convenience
    dedup of `citations` for a caller that only cares "which ids", not
    "how many times and where."

The critical semantic this module exists to enforce, per the task brief
and `dedup.py`'s prior precedent for documenting an easy-to-get-wrong
distinction: verification checks "was this chunk id in `valid_chunk_ids`
-- the exact set the model was shown for *this* context" -- never "does
a chunk with this id exist anywhere in the corpus, ever." A citation to
a real, persisted `Chunk.id` that simply wasn't part of *this* answer's
context (e.g. it was reranked/deduped/budgeted out, or belongs to an
entirely different retrieval) is still fabricated for this answer and
must be flagged `verified=False` -- the model cited something it was
never shown, which is exactly as untrustworthy as citing an id that does
not exist at all. `valid_chunk_ids` is intentionally the caller's
responsibility to construct correctly (e.g. from
`context_budget.ContextBlock.chunk_ids` -- the authoritative "what did
the model actually see" list, not a DB-wide id set) -- this module has
no database access and no opinion on where the set comes from, only on
membership within it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches exactly the literal citation token the model is instructed to
# produce -- see module docstring. `\d+` (not `\w+` or a permissive
# id pattern) deliberately excludes non-numeric ids like "[chunk:abc]"
# from matching at all, so malformed syntax is simply not recognized as
# a citation (neither verified nor fabricated) rather than crashing the
# parser or being coerced into some guessed chunk id.
_CITATION_PATTERN = re.compile(r"\[chunk:(\d+)\]")


@dataclass(frozen=True)
class Citation:
    """One parsed `[chunk:<id>]` occurrence in an answer's text.

    `raw_text`/`start`/`end` are the exact matched substring and its
    character offsets into the `answer_text` that was verified -- enough
    for a caller to do exact in-place string surgery (e.g.
    `answer_text[:c.start] + answer_text[c.end:]` to strip one fabricated
    citation marker) without re-parsing.
    """

    chunk_id: int
    verified: bool
    raw_text: str
    start: int
    end: int


@dataclass(frozen=True)
class VerificationResult:
    """The result of verifying every citation in one answer's text
    against the set of chunk ids actually present in the context the
    model was given. See module docstring for the full reasoning behind
    this shape and the "shown to the model, not exists anywhere" (not)
    semantic it deliberately enforces.
    """

    citations: list[Citation] = field(default_factory=list)
    all_verified: bool = True
    verified_chunk_ids: list[int] = field(default_factory=list)
    fabricated_chunk_ids: list[int] = field(default_factory=list)


def verify_citations(answer_text: str, valid_chunk_ids: set[int]) -> VerificationResult:
    """Parse every `[chunk:<id>]` citation out of `answer_text` and
    classify each as verified (its id is in `valid_chunk_ids`) or
    fabricated (it is not).

    `answer_text=""` or an answer with no citations at all returns an
    empty `VerificationResult` with `all_verified=True` -- zero citations
    is a valid, unremarkable case (e.g. a general-knowledge answer with
    nothing to cite), not something to flag. Malformed citation-like
    syntax (`[chunk:abc]`, `[chunk 5]` with no colon, `[Chunk:5]` wrong
    case) is simply not matched by `_CITATION_PATTERN` -- it is invisible
    to this parser, never counted as either verified or fabricated, and
    never raises.

    `valid_chunk_ids` membership is the *only* thing that determines
    `verified` -- see module docstring's "shown to this context, not
    exists anywhere" distinction. A duplicate citation to the same id
    (the model citing `[chunk:5]` twice) produces two entries in
    `citations` (one per occurrence, since each has its own text span a
    caller may need to strip independently) but only one entry in
    `verified_chunk_ids`/`fabricated_chunk_ids` (first-appearance order).
    """
    citations: list[Citation] = []
    verified_ids: list[int] = []
    fabricated_ids: list[int] = []
    seen_ids: set[int] = set()

    for match in _CITATION_PATTERN.finditer(answer_text):
        chunk_id = int(match.group(1))
        is_verified = chunk_id in valid_chunk_ids
        citations.append(
            Citation(
                chunk_id=chunk_id,
                verified=is_verified,
                raw_text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            (verified_ids if is_verified else fabricated_ids).append(chunk_id)

    return VerificationResult(
        citations=citations,
        all_verified=all(c.verified for c in citations),
        verified_chunk_ids=verified_ids,
        fabricated_chunk_ids=fabricated_ids,
    )
