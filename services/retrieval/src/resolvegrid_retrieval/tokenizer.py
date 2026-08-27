"""Approximate token counting for chunk sizing.

Chunking needs a token-count *signal* to decide when a section is big
enough to sub-split (target ~300-500 tokens per chunk, see `chunker.py`).
It does not need an exact count matching any one provider's real
tokenizer -- there isn't a single "real" answer here anyway, since this
repo's embedding model (`nomic-embed-text` via Ollama, per
`apps/api/src/resolvegrid_api/models/knowledge.py`'s `EMBEDDING_DIM`
comment) uses its own tokenizer, not OpenAI's.

Tradeoff, written down explicitly (per Phase 7 Task 2's instructions):

  tiktoken (rejected for this package):
    + Exact token counts for OpenAI's cl100k/o200k BPE vocabularies.
    - Those counts are still *not* what `nomic-embed-text` (or whatever
      embedding model a later phase swaps in) actually tokenizes to --
      it would be trading one approximation for a different one, at a
      real cost: a multi-megabyte Rust extension wheel, and a first-use
      path that can reach out to the network to download BPE merge
      tables unless pre-vendored/cached, which is an avoidable
      reliability risk for a small, offline-friendly chunker package.

  Word/punctuation regex count (chosen):
    + Zero dependencies, fully deterministic, offline, and fast.
    + Close enough for a "roughly 300-500 tokens" *budget* decision --
      the chunker only needs to know "is this section big enough to
      need splitting," not a provider-exact count.
    - Will disagree with any specific real tokenizer by roughly
      +/-10-20% on ordinary English prose (BPE tokenizers often split
      inside longer words into sub-word pieces, which this approximation
      does not do), so `token_count` should be read as an estimate, not
      an exact count, by whatever later task persists it to `Chunk.token_count`.

If a later phase needs the real embedding model's exact token count
(e.g. for a hard context-window limit rather than a soft chunk-size
target), that should be computed at embedding time from the actual
model's tokenizer/API, not here.
"""

import re

# Matches runs of "word" characters (letters/digits/underscore, unicode-aware)
# as one token, or any single non-whitespace, non-word character (punctuation)
# as its own token. This roughly mirrors how most subword tokenizers keep
# punctuation as separate tokens from adjacent words, without attempting
# any real subword segmentation.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_token_count(text: str) -> int:
    """Return an approximate token count for `text` (see module docstring).

    Empty/whitespace-only text returns 0.
    """
    if not text or not text.strip():
        return 0
    return len(_TOKEN_PATTERN.findall(text))
