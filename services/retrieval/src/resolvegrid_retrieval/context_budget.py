"""Token-budget-aware context assembly (Phase 8 Task 3).

Scope note: this module is a standalone, testable capability, same as
`reranker.py` and `dedup.py` before it. It is deliberately **not** wired
into `apps/api/src/resolvegrid_api/retrieval.py`, the agent graph
(`services/agent-orchestration`), or `/chat` yet -- that is Phase 8 Task
7. It replaces the *behavior* of the graph's current
`_build_context_block` (an unbounded `"\n\n".join(...)` over every
retrieved chunk, no budget, no prioritization beyond `fuse_rrf`'s raw
order) with a budgeted equivalent that a later task can drop in, once it
also decides how `rerank()` -> `dedup()`'s output gets titles reattached
for the graph's `RetrievedChunk` shape.

Input/output shape: this module takes `(chunk_id, title, text)` triples,
best-first (i.e. `dedup()`'s `(chunk_id, text, score)` output, in the
order `dedup()` already returns it, with each chunk's document title
re-attached by the caller and the now-redundant `score` dropped -- the
same "trivial dict lookup against data the caller already has, no new
data source" composability precedent `dedup()`'s own docstring documents
for how it consumes `rerank()`'s output). Best-first order is this
module's *only* prioritization signal -- it does not re-score anything
itself, it only decides how much of an already-ranked list fits in a
fixed token budget.

Deviation from this task's illustrative signature, decided (not
overlooked): the task brief sketches `assemble_context(query, chunks,
*, max_tokens=...)`. `query` is dropped here. It was seriously
considered for query-aware extractive truncation of an oversized chunk
(e.g. keep only the sentence closest to the query) -- but the
truncate-vs-skip decision below rejects truncation entirely for a
citation-integrity reason that has nothing to do with query relevance,
so a `query` parameter would sit in this module's signature completely
unused by its actual logic. An unused parameter kept only to match an
illustrative example is worse engineering than a smaller, honest
signature -- see the truncate-vs-skip writeup below for why truncation
(query-aware or not) was rejected.

Token budget, investigated and measured for real in this environment,
2026-08-28 (not assumed from `qwen3:14b`'s marketing context length, and
not taken from web documentation, which disagreed with itself across
sources -- Ollama's own FAQ says 4096, its Modelfile reference says 2048,
and its context-length docs say "picked from available VRAM"):

  This repo's actual running `resolvegrid-ollama` container was queried
  directly rather than guessed about. `docker exec resolvegrid-ollama
  ollama show qwen3:14b` reports the model's own architecture max as
  `context length: 40960` -- but that is the model's theoretical
  capability, not what this deployment actually runs it with: the
  model's Modelfile (`ollama show qwen3:14b --modelfile`) sets no
  `PARAMETER num_ctx` at all, and neither `infra/litellm/config.yaml`
  nor `apps/api/src/resolvegrid_api/llm_gateway.py`'s `complete()`
  request body sets one either (confirmed by inspection of both -- no
  `num_ctx`/`options` key anywhere in the request). So the number that
  actually governs a real `/chat` completion here is whatever Ollama's
  runtime defaults to for this tag on this host, not 40960.

  Measured directly: a real completion call was sent to this
  environment's live Ollama instance (`POST /api/generate`,
  `model=qwen3:14b`), then `docker exec resolvegrid-ollama ollama ps`
  was checked immediately after, while the model was resident. Its
  `CONTEXT` column reported **4096** -- the actual, real context window
  this deployment runs `qwen3:14b` with today, empirically confirmed,
  not inferred from conflicting docs.

  Budget derivation against that real 4096-token ceiling (one
  `compose_response` completion call is a single, independent request --
  `num_ctx` bounds that one call's prompt tokens *plus* its generated
  tokens together, not the whole graph run, since `classify_intent` and
  `compose_response` are separate HTTP calls to the gateway):

    - Fixed instruction-prompt boilerplate (everything in
      `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE` except `context_block` and
      `input_text`): measured at 146 tokens via this repo's own
      `estimate_token_count` against the real template text. Rounded up
      to **200** for headroom.
    - User's question (`input_text`): budgeted generously at **250**
      tokens -- comfortably more than a typical IT-support question, but
      real users do sometimes paste a multi-sentence description.
    - Room for the model's own generated answer: **800** tokens --
      enough for a real multi-sentence answer with several inline
      `[chunk:<id>]` citations, not a one-liner. `num_ctx` covers
      generation too (Ollama's `prompt_eval_count` + `eval_count` both
      count against it), so this has to come out of the same 4096, not
      be treated as free.
    - Reserved subtotal: 200 + 250 + 800 = 1250, leaving 4096 - 1250 =
      2846 raw headroom for retrieved-context text.
    - A further safety margin is applied on top of that raw headroom,
      not just at the edges: `tokenizer.py`'s own docstring documents
      that this repo's word-regex `estimate_token_count` can disagree
      with a real BPE tokenizer by roughly +/-10-20% on ordinary
      English prose, and specifically *under*-counts relative to a real
      subword tokenizer (BPE splits inside longer words into multiple
      sub-word tokens; the regex estimator counts one word as one
      token). Since every reservation above (boilerplate, question,
      context) is computed with the *same* estimator, budgeting right up
      to the raw 2846-token edge risks the real request exceeding 4096
      once actual BPE tokenization is applied.

  `DEFAULT_CONTEXT_TOKEN_BUDGET = 2000` -- comfortably under the raw
  2846-token headroom (leaving ~800 tokens, or ~28%, of margin against
  the estimator's documented undercount risk across all of this
  budget's reserved pieces, not just the context portion), while still
  being a real, useful amount of retrieved context: at this repo's
  chunker's ~300-500-token target chunk size (`chunker.py`), 2000 tokens
  is enough for roughly 4-6 chunks of real retrieved context per answer
  -- a realistic number of sources for a single IT-ops question, not a
  token-starved sliver.

Truncation strategy for an oversized chunk, decided (not the first
option reached for):

  Phase 7's chunker documents that a single paragraph larger than its
  ~300-500-token target still becomes its own, oversized chunk rather
  than being split mid-paragraph (`chunker.py`'s `_group_paragraphs`
  docstring). So a chunk whose formatted entry alone exceeds this
  module's remaining (or even total) budget is a real, if rare, case to
  handle.

  Two options, per this task's brief: truncate the chunk's text to fit,
  or skip it entirely. **This module skips.** Truncating was rejected
  for a citation-integrity reason specific to how this pipeline's
  citations work, not a style preference:

    - The whole point of a `[chunk:<id>]` citation is that it names a
      real, persisted `Chunk` row, and Task 4's `verify_citations` (not
      yet built, but its stated job per the Phase 8 plan doc is
      "every `[chunk:<id>]` citation the model's answer contains must
      map to a chunk actually present in the context it was given") will
      check "was chunk id X present in what the model saw" -- not "was
      *all* of chunk id X's text present." If this module truncated a
      chunk's text before including it, a citation to that chunk would
      look fully valid to that check, while the model may have only seen
      (and be citing) a fragment -- possibly cut off mid-sentence, mid-
      number, or mid-clause. That is a strictly worse failure mode than
      not showing the model that chunk at all: a confidently-cited
      partial fact is more dangerous in an IT-ops/policy context (e.g.
      truncating "...MFA is mandatory for every VPN session, with no
      opt-out" to "...MFA is mandatory for every VPN session" mid-list-
      item, silently losing the "no opt-out" qualifier) than an honest
      "the context doesn't fully answer this" -- which is the fallback
      behavior `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE` already
      explicitly instructs the model to use when context is incomplete.
    - Skipping preserves a simple, exact invariant that composes cleanly
      with Task 4's future needs: every chunk id in
      `ContextBlock.chunk_ids` corresponds to that chunk's complete,
      unmodified, verbatim text in `ContextBlock.text` -- never a
      partial slice a downstream reader would have to know to treat
      differently.
    - The real downside the task brief names -- "skipping risks losing
      the single best-ranked result if it happens to be long" -- is
      mitigated by *not* stopping assembly at the first chunk that
      doesn't fit: `assemble_context` continues past a chunk that would
      overflow the remaining budget and keeps trying subsequent
      (lower-priority) candidates, so one oversized chunk costs only
      itself, not the rest of the budget. This does mean a smaller,
      lower-rerank-score chunk can end up included where a single larger
      higher-score chunk was skipped -- an inherent, accepted tradeoff of
      greedily filling a fixed budget rather than reserving space
      up front for the single best candidate regardless of size.

Chunk-id transparency for citation verification, decided: `ContextBlock`
returns three things, not a bare string a caller would have to
re-parse -- `text` (the assembled block, in the exact
`[chunk:<id>] (from "<title>"):\n<text>` format
`_build_context_block` already produces, joined by blank lines, so a
later wiring task is a clean drop-in swap for the "everything fits"
case), `chunk_ids` (best-first ordered list of chunk ids that actually
made it into `text` -- this is what Task 4's `verify_citations` needs:
the exact set/order of chunks the model was actually shown, not
`dedup()`'s pre-budget candidate list), and `dropped_chunk_ids` (ids
that were excluded for budget reasons, best-first order among
themselves -- not required by this task's brief, but cheap to expose and
useful for logging/telemetry on how often budgeting is actually binding
in practice, without a caller having to diff `chunk_ids` against its own
input list by hand).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resolvegrid_retrieval.tokenizer import estimate_token_count

# See module docstring's "Token budget, investigated and measured for
# real in this environment" section for the full derivation against this
# repo's actual, measured qwen3:14b `num_ctx` (4096, confirmed via
# `docker exec resolvegrid-ollama ollama ps` after a real completion
# call) -- not the model's marketing/architecture max (40960).
DEFAULT_CONTEXT_TOKEN_BUDGET = 2000


def _format_entry(chunk_id: int, title: str, text: str) -> str:
    """Render one chunk as a citation-ready context entry. Mirrors
    `services/agent-orchestration`'s `_build_context_block` entry format
    exactly (`[chunk:<id>] (from "<title>"):\\n<text>`), so this module's
    assembled `text` output matches what that function already produces
    for the "everything fits" case -- the drop-in-replacement requirement
    this task calls for.
    """
    return f'[chunk:{chunk_id}] (from "{title}"):\n{text}'


@dataclass(frozen=True)
class ContextBlock:
    """The result of budgeted context assembly.

    `text`: the assembled context block, ready to substitute into
    `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE`'s `{context_block}` slot.
    Entries are joined with a blank line, matching
    `_build_context_block`'s current `"\\n\\n".join(...)` exactly.

    `chunk_ids`: best-first ordered ids of chunks actually included in
    `text` -- the unambiguous "what did the model actually see" list a
    later citation-verification task needs (see module docstring).

    `dropped_chunk_ids`: ids excluded purely for budget reasons (didn't
    fit in what remained when their turn came), best-first order among
    themselves. Not the same as "irrelevant" -- a dropped chunk may have
    ranked highly; it was excluded because of size/ordering, not
    relevance. Exposed for observability, not required for citation
    verification.
    """

    text: str
    chunk_ids: list[int] = field(default_factory=list)
    dropped_chunk_ids: list[int] = field(default_factory=list)


def assemble_context(
    chunks: list[tuple[int, str, str]],
    *,
    max_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> ContextBlock:
    """Assemble a token-budgeted context block from `chunks` -- a
    best-first (post-rerank, post-dedup) list of `(chunk_id, title,
    text)` triples.

    Walks `chunks` in the given order (the caller's priority order --
    this module does not re-sort; see module docstring for why
    `dedup()`'s output order is exactly what should be passed here) and
    greedily includes each chunk's formatted entry
    (`[chunk:<id>] (from "<title>"):\\n<text>`) if it fits in whatever
    token budget remains, using this repo's `estimate_token_count` (the
    same estimator `chunker.py` uses for its ~300-500-token chunk-sizing
    target -- reused here rather than a second token-counting approach,
    per this task's instructions).

    A chunk that doesn't fit in the *remaining* budget is skipped (not
    truncated -- see module docstring's truncate-vs-skip writeup) and
    assembly continues to the next chunk in priority order, so one
    oversized or budget-exceeding chunk only costs itself, not the rest
    of the pass.

    `chunks=[]` returns an empty `ContextBlock` (`text=""`, both id lists
    empty) without doing any work.
    """
    included_entries: list[str] = []
    included_ids: list[int] = []
    dropped_ids: list[int] = []
    remaining_budget = max_tokens

    for chunk_id, title, text in chunks:
        entry = _format_entry(chunk_id, title, text)
        entry_tokens = estimate_token_count(entry)

        if entry_tokens <= remaining_budget:
            included_entries.append(entry)
            included_ids.append(chunk_id)
            remaining_budget -= entry_tokens
        else:
            dropped_ids.append(chunk_id)

    return ContextBlock(
        text="\n\n".join(included_entries),
        chunk_ids=included_ids,
        dropped_chunk_ids=dropped_ids,
    )
