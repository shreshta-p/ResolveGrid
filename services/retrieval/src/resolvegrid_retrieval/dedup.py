"""Near-duplicate / adjacent-chunk deduplication of reranked candidates
(Phase 8 Task 2).

Scope note: this module is a standalone, testable dedup capability. Like
`reranker.py` before it, it is deliberately **not** wired into
`apps/api/src/resolvegrid_api/retrieval.py`, the agent graph, or `/chat`
yet -- that is a later Phase 8 task (context budgeting needs to exist
first to decide how deduped candidates flow into the context block).

Input/output shape, decided (not assumed) from how this composes with
Task 1's `reranker.rerank()`:

  `rerank()` takes `(chunk_id, text)` pairs and returns `(chunk_id,
  rerank_score)` pairs, best-first -- it drops `text` from its output
  because a rerank score alone is enough for *that* module's job. This
  module needs the text back (to compare candidates for near-duplication)
  and needs to keep the score (to decide which of two near-duplicates
  survives, and so `dedup()`'s own output stays sorted best-first for
  whatever composes with it next -- Task 3's context budgeting). So
  `dedup()` takes `(chunk_id, text, score)` triples: `rerank()`'s output
  with each chunk's text re-attached by the caller (a trivial dict lookup
  against the same `candidates` list `rerank()` was called with -- no new
  data source needed), and returns the same shape with near-duplicates
  removed. This keeps the two modules composable without a shared/new
  dataclass or either one depending on the other's internals.

Similarity method, investigated and decided (not the first option
reached for), 2026-08-28:

  The task brief poses this as embedding cosine similarity vs. a
  text-overlap heuristic. Embedding similarity was seriously considered
  and rejected for *this* package, for a real architectural reason, not
  convenience:

    - Chunks already have embeddings computed and stored at ingestion
      time (`Embedding.vector` in `apps/api`'s Postgres/pgvector schema,
      per `apps/api/src/resolvegrid_api/models/knowledge.py`). Reusing
      those would be free at query time -- but they live in `apps/api`'s
      database, and `services/retrieval` has no SQL/DB dependency at all
      (confirmed: `pyproject.toml` declares only `httpx`, for
      `embedder.py`'s direct Ollama HTTP calls -- no `sqlalchemy`,
      `psycopg`, or `apps/api` import anywhere in this package). Giving
      `dedup()` a DB dependency just to fetch vectors it can't otherwise
      get would break this package's established "plain library, no DB"
      design and would make it untestable without a running Postgres --
      a real regression, not a style nitpick (`reranker.py`'s own
      docstring documents the same discipline being applied to keep
      `sentence-transformers` out of the *core* dependency set).
    - The alternative -- computing *fresh* embeddings for candidates
      inside `dedup()` via `embedder.py`'s `embed_texts()` -- was also
      rejected: it means a real network round-trip to Ollama on every
      `dedup()` call (on top of the round-trip ingestion already paid
      for the exact same text), for a step whose only job is "is this
      candidate too similar to one already kept" among a small
      (~10-20-item) reranked list. That is a real latency/reliability
      cost (another network dependency, another failure mode) for a
      comparison embedding cosine similarity is not even guaranteed to
      get meaningfully more right than a cheaper method on this specific
      sub-problem (see threshold validation below).

  `rerank()`'s own signature is the deciding precedent: it already
  operates on `(chunk_id, text)` pairs, not vectors, despite reranking
  being the step where semantic accuracy matters most in this pipeline.
  A text-based dedup step operating on the same `(chunk_id, text)` shape
  keeps the whole reranked-candidates pipeline (rerank -> dedup -> [Task
  3's budgeting]) consistent in what it passes around, and keeps this
  package dependency-free for this feature (no new import at all, unlike
  `reranker.py`'s optional `sentence-transformers` extra).

  Concretely: word-level (unigram) Jaccard similarity over each
  candidate's normalized, stopword-filtered token set. Word bigrams and
  word-sequence trigram "shingles" (the more sophisticated text-overlap
  techniques, e.g. used in near-duplicate web page detection) were tried
  first and rejected after *measuring* them against real and realistic
  corpus content (see threshold validation below): trigram shingles are
  sensitive to word order/phrasing to the point that even a genuine
  same-fact-reworded pair scored under 0.07 similarity, indistinguishable
  from two genuinely unrelated chunks (also near 0) -- reworded prose
  rarely preserves 3-word runs verbatim, so shingles of that length
  systematically under-detect exactly the realistic near-duplicate case
  this module exists to catch. Plain unigram overlap, which only cares
  "do these two chunks talk about the same specific things" rather than
  "in what order," produced a real, measurable separation between
  reworded-same-fact pairs and superficially-similar-but-different pairs
  (numbers below). A short, fixed English stopword list is stripped
  before comparison so high-frequency function words ("the", "is", "to",
  "and", ...) that are present in nearly every chunk regardless of topic
  don't dilute the signal.

Threshold, validated against real and realistic examples (not picked
arbitrarily), 2026-08-28 -- measured with this module's actual
`jaccard_similarity()`:

  Genuine near-duplicate (the realistic case this task's brief calls
  for: "two internal docs might restate the same policy point"),
  constructed from real corpus content -- one chunk *is*
  `eval/corpus/kestrel-time-off-request.md`'s real "PTO Accrual"
  section; the other is a synthetic FAQ-style restatement of the exact
  same policy (same specific facts -- "40 hours", "calendar year",
  "tenure", "local law" -- reworded around them, the way a second
  internal document restating a known policy point realistically would,
  keeping the load-bearing nouns/numbers and changing sentence
  structure): **similarity = 0.370** (see `test_dedup.py`).

  Genuine near-miss (the important edge case: superficially similar,
  substantively different) -- real corpus content, `kestrel-vpn-policy-
  v1-deprecated.md`'s and `kestrel-vpn-policy-v2.md`'s MFA/password-rules
  sections. These are about as close as two real chunks in this corpus
  get to "topically identical" (same document family, same subsection,
  same vocabulary: VPN, password, multi-factor authentication,
  employees, days) while stating opposite facts (v1: MFA optional,
  180-day/8-char passwords; v2: MFA mandatory with no opt-out, 90-day/
  12-char passwords): **similarity = 0.229** (see `test_dedup.py`). A
  threshold that
  collapsed this pair would silently discard the *current* mandatory-MFA
  policy in favor of (or in addition to) a chunk correctly labeled
  deprecated, or vice versa -- exactly the failure mode this task's
  brief warns a too-loose threshold produces.

  Unrelated control (VPN password reset vs. PTO request, different
  documents and topics entirely): **similarity = 0.031**.

  `DEFAULT_DEDUP_THRESHOLD = 0.30` sits strictly between the near-miss
  ceiling observed (0.229) and the near-duplicate floor observed (0.370),
  closer to the near-miss side deliberately: an incorrectly *kept*
  near-duplicate costs a little context-budget space (Task 3's problem to
  manage) and mild redundancy in what the model reads; an incorrectly
  *collapsed* near-miss can silently delete a real, distinct, possibly
  safety-relevant fact (e.g. dropping the current mandatory-MFA chunk)
  from what the model ever sees. Erring toward under-collapsing is the
  safer failure mode, so the threshold is set nearer the measured
  near-miss ceiling than the measured near-duplicate floor, not centered
  between them.

  Known, documented limitation of this heuristic (not glossed over): a
  *heavily* synonym-substituted paraphrase of the same fact -- one that
  changes essentially every content word, not just sentence structure,
  e.g. rewriting "accrue PTO ... based on tenure ... annual cap" as
  "build up paid time off ... depends on how long you've been with the
  company ... yearly maximum" -- was measured during threshold validation
  at **similarity = 0.172**, *below* the near-miss ceiling above. A
  word-overlap heuristic cannot distinguish "same fact, no shared
  vocabulary" from "different fact" -- that is precisely the semantic
  gap embedding similarity would close, at the dependency/latency cost
  documented above. This module's threshold is tuned for the realistic
  case (enterprise KB restatements typically retain the specific
  terms/numbers that carry the actual information -- a policy's "40
  hours" or "90 days" is what a second document keeps, because that's
  the fact being restated, not filler prose), not the adversarial case
  of a maximally thesaurus-driven paraphrase. If false negatives
  (near-duplicates that survive dedup) turn out to matter in practice
  against real production traffic, that is the concrete, measured
  justification for revisiting this module toward embedding similarity
  later -- not a hypothetical.

Which chunk survives, decided: the higher-rerank-score chunk. `dedup()`
defensively re-sorts its input by `(-score, chunk_id)` -- the same
best-first, deterministic-tie-break convention `rerank()` and
`fuse_rrf()` both already use -- before comparing, so its output is
correct best-first order regardless of whether the caller's input was
already sorted (it will be, in the real `rerank()` -> `dedup()` pipeline,
but `dedup()` should not silently produce wrong "best-first" output for
the one caller that gets that wrong). It then walks candidates in that
order and greedily keeps each one unless it is similar enough (at or
above `threshold`) to a chunk *already kept* -- so ties are broken in
favor of whichever candidate ranks first in best-first order, i.e. the
higher rerank score. This also naturally handles a duplicate cluster of
more than two chunks (all collapse onto the single highest-scoring
survivor) and non-adjacent duplicates (comparison is against every
already-kept survivor, not just the immediately preceding candidate --
"adjacent" in this task's brief describes a common real-world case
[re-chunked/re-ingested near-identical source docs tend to rank near
each other after fusion+reranking], not a positional constraint on what
this function checks).
"""

from __future__ import annotations

import re

# Matches runs of ASCII letters/digits as one word, mirroring
# `tokenizer.py`'s zero-dependency word-regex approach elsewhere in this
# package. Case is normalized by the caller (`_normalize_words` lowercases
# first) so this pattern doesn't need a case-insensitive flag itself.
# Deliberately narrower than tokenizer.py's `\w+` (which also matches
# underscore and non-ASCII word characters via re.UNICODE): dedup only
# needs "is this the same word" for ordinary English prose content, not a
# token-budget estimate, so the extra generality isn't needed here.
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

# A short, fixed list of high-frequency English function words. These
# appear in nearly every chunk regardless of topic (see module docstring)
# and would otherwise inflate the similarity of any two chunks of ordinary
# English prose regardless of actual shared content. This is intentionally
# a small, unsurprising list (not a full NLP stopword corpus, which would
# be a new dependency) -- it only needs to filter the words common enough
# to show up in virtually any two unrelated sentences.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "and", "or", "that", "this", "these",
        "those", "at", "by", "with", "as", "it", "its", "your", "you", "not",
        "no", "up", "into", "any", "per", "from", "if", "than", "then",
        "once", "before", "after", "only", "through", "under", "over",
        "will", "must", "can", "may", "has", "have", "had", "do", "does",
        "did", "so", "such", "there", "their", "they", "them",
    }
)

# See "Threshold, validated against real and realistic examples" above for
# the measured numbers this value was chosen against.
DEFAULT_DEDUP_THRESHOLD = 0.30


def _normalize_words(text: str) -> frozenset[str]:
    """Return `text`'s content words: lowercased, alphanumeric-only
    tokens with stopwords removed, as a set (word *presence*, not
    frequency/position, is what Jaccard similarity compares).
    """
    words = _WORD_PATTERN.findall(text.lower())
    return frozenset(word for word in words if word not in _STOPWORDS)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-overlap similarity between `text_a` and `text_b`, in `[0, 1]`:
    the size of their (stopword-filtered) shared-word set divided by the
    size of their combined word set. `1.0` means identical content-word
    sets; `0.0` means no shared content words at all.

    Two texts that are both empty/entirely stopwords are treated as
    identical (`1.0`) -- there is no content to differ on. A text that is
    empty/stopwords-only compared against a non-empty one returns `0.0`.
    """
    words_a = _normalize_words(text_a)
    words_b = _normalize_words(text_b)

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union


def dedup(
    ranked_candidates: list[tuple[int, str, float]],
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> list[tuple[int, str, float]]:
    """Remove near-duplicate chunks from `ranked_candidates` -- a list of
    `(chunk_id, text, score)` triples (e.g. `rerank()`'s `(chunk_id,
    rerank_score)` output with each chunk's text re-attached by the
    caller; see module docstring for why this shape was chosen).

    Returns the surviving candidates in best-first order (highest `score`
    first, `chunk_id` ascending as a deterministic tie-break -- see module
    docstring for why this reordering happens defensively even though the
    real `rerank()` -> `dedup()` pipeline already produces best-first
    input). When two or more candidates are near-duplicates of each other
    (word-overlap `jaccard_similarity` at or above `threshold`), only the
    highest-scoring one survives.

    `ranked_candidates=[]` returns `[]`. A single candidate is always
    returned unchanged (trivially, nothing to compare it against).
    """
    ordered = sorted(ranked_candidates, key=lambda c: (-c[2], c[0]))

    survivors: list[tuple[int, str, float]] = []
    for candidate in ordered:
        _chunk_id, text, _score = candidate
        is_near_duplicate = any(
            jaccard_similarity(text, kept_text) >= threshold
            for _kept_id, kept_text, _kept_score in survivors
        )
        if not is_near_duplicate:
            survivors.append(candidate)

    return survivors
