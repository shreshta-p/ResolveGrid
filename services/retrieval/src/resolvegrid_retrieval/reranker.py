"""Cross-encoder reranking of retrieved chunks (Phase 8 Task 1).

Scope note: this module is a standalone, testable reranking capability.
It is deliberately **not** wired into `apps/api/src/resolvegrid_api/
retrieval.py`, the agent graph, or `/chat` yet -- that is a later Phase 8
task, once dedup and context budgeting exist to consume its output
alongside `fuse_rrf`'s.

Model/library choice, investigated and verified in this environment
(2026-08-28), not assumed from the original architecture plan:

  The original plan named `bge-reranker-v2-m3` via `sentence-transformers`
  or `FlagEmbedding`, with `bge-reranker-base` as a fallback "if the
  footprint is too heavy." Both claims were checked for real rather than
  taken on faith:

  Library: `sentence-transformers` vs. `FlagEmbedding`. Both require
  `torch` + `transformers` as transitive dependencies -- there is no way
  to run a real BGE cross-encoder on this hardware without a tensor
  framework. `FlagEmbedding` (BAAI's own package) additionally pulls in
  training-oriented dependencies (`datasets`, `accelerate`, etc.) this
  module has no use for -- it only ever needs inference. `sentence-
  transformers` ships a `CrossEncoder` class that is the standard,
  actively-maintained way to run a `[query, passage] -> relevance score`
  cross-encoder (BGE's model cards on Hugging Face document exactly this
  usage), so it was chosen as the lighter, inference-only path.

  Model: `bge-reranker-v2-m3` vs. `bge-reranker-base`, measured for real
  in this CPU-only dev environment (no CUDA toolchain assumed here --
  `services/retrieval` should not assume GPU the way local Ollama
  inference does; confirmed after install that the resolved `torch` wheel
  is `2.13.0+cpu` with `torch.cuda.is_available() == False`):

    bge-reranker-v2-m3 (568M params, XLM-RoBERTa-large-based):
      - Download: 2.2GB (fp32 safetensors/bin weights cached under
        `~/.cache/huggingface/hub`).
      - First load (incl. download, this environment): ~134s.
      - Inference, 12 candidates, warm (model already resident in
        memory): ~4.08s total, ~340ms/candidate.
    bge-reranker-base (278M params, XLM-RoBERTa-base-based):
      - Download: 1.1GB.
      - First load (incl. download, this environment): ~66s.
      - Inference, 12 candidates, warm: ~0.94s total, ~78ms/candidate.

  `bge-reranker-v2-m3`'s ~4s warm reranking latency for a realistic
  10-12-candidate batch does not fit a `/chat` request's latency budget
  when it is one step among retrieval + LLM generation, not the whole
  budget. `bge-reranker-base` at under 1s for the same batch is
  practical. This module therefore uses `bge-reranker-base` as the
  **default** model -- the original plan's stated fallback turned out to
  be the real choice, not a hedge that never materializes. This is a
  measured tradeoff (v2-m3 is the stronger multilingual model per BAAI's
  published benchmarks; this corpus is English-only internal IT-ops
  content, per `eval/corpus/*.md`, so v2-m3's multilingual strength buys
  nothing here that offsets its ~4x slower CPU inference). `model` is a
  keyword argument on `rerank()` below specifically so a future task can
  override it (e.g. if GPU inference becomes available and v2-m3's
  latency becomes acceptable) without an API change.

  Dependency footprint: see `pyproject.toml`'s `[project.optional-
  dependencies] reranker` comment for why `sentence-transformers` (and
  the torch/transformers/scipy/numpy stack it pulls in -- measured at
  ~200MB across 28 packages via a real `uv add` in this environment) is
  an optional extra rather than a core dependency of this package.

Model loading is cached process-wide (`_MODEL_CACHE` below): the ~1-2s
weight-load-from-local-disk cost (separate from the one-time network
download measured above) is real but wasteful to pay on every `rerank()`
call in a process that reranks many queries, so the loaded `CrossEncoder`
is kept in memory after first use, keyed by model name.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# BGE reranker models are trained/documented for up to 512-token
# query+passage pairs (per BAAI's model card) -- this repo's chunks target
# 300-500 tokens each (see chunker.py), so a 512-token cap on the
# underlying WordPiece/SentencePiece tokenization (not this repo's
# word-regex `estimate_token_count`, a different, coarser measure) is
# enough for one chunk's text plus a short query, with truncation as a
# fail-soft fallback for anything longer rather than an error.
DEFAULT_MAX_LENGTH = 512

# Real measured CPU inference cost (see module docstring) is dominated by
# per-batch overhead, not per-call Python overhead, so the batch size just
# needs to comfortably cover a realistic reranking candidate count (~10-20
# fused chunks per query, per this task's brief) in a single batch rather
# than being tuned further -- sentence-transformers' own default (32) is
# already more than enough headroom and is left unset/unoverridden here.

# Process-wide cache of loaded models, keyed by model name. See module
# docstring for why this exists (avoid re-paying disk-load cost on every
# call) and `clear_model_cache()` below for the test-isolation escape
# hatch.
_MODEL_CACHE: dict[str, object] = {}


class RerankError(Exception):
    """Raised for any failure reranking candidates: the optional
    `sentence-transformers` dependency isn't installed, the model fails
    to load (network failure, invalid/unknown model name, corrupt local
    cache), or inference itself fails. Callers should only ever need to
    catch this one exception type, never a raw `ImportError`/
    `OSError`/whatever `sentence-transformers`/`torch` happen to raise
    internally -- this mirrors `embedder.py`'s `EmbeddingError`
    convention.
    """


def clear_model_cache() -> None:
    """Drop all cached loaded models. Exists for test isolation (forcing
    a fresh `_get_model()` load, e.g. to exercise the model-load-failure
    path with a bad model name after a real model has already been
    loaded and cached under a different name) -- production callers have
    no reason to call this.
    """
    _MODEL_CACHE.clear()


def _get_model(model_name: str):
    """Return the cached `CrossEncoder` for `model_name`, loading and
    caching it first if necessary. Raises `RerankError` if
    `sentence-transformers` isn't installed or the model fails to load.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankError(
            "sentence-transformers is not installed -- reranker.py "
            "requires the optional 'reranker' extra "
            "(`resolvegrid-retrieval[reranker]`); see pyproject.toml for "
            "why it isn't a core dependency of this package"
        ) from exc

    logger.info("loading reranker model '%s' (first use in this process)", model_name)
    try:
        model = CrossEncoder(model_name, max_length=DEFAULT_MAX_LENGTH)
    except Exception as exc:  # noqa: BLE001 -- sentence-transformers/torch/
        # huggingface_hub can raise many different exception types for a
        # bad model name, a network failure, or a corrupt cache entry
        # (OSError, HFValidationError, EnvironmentError, ...); all of them
        # mean the same thing to a caller of this module -- the model
        # could not be loaded -- so they are normalized to one clear
        # RerankError rather than leaking an internal exception type.
        raise RerankError(f"failed to load reranker model '{model_name}': {exc}") from exc

    _MODEL_CACHE[model_name] = model
    return model


def rerank(
    query: str,
    candidates: list[tuple[int, str]],
    *,
    model: str = DEFAULT_RERANKER_MODEL,
) -> list[tuple[int, float]]:
    """Rerank `candidates` (a list of `(chunk_id, chunk_text)` pairs,
    e.g. `fuse_rrf`'s fused `(chunk_id, rrf_score)` output with each
    chunk's text attached) against `query` using a local cross-encoder.

    Returns `(chunk_id, rerank_score)` pairs ordered best-first (rerank
    score descending; ties broken by `chunk_id` ascending, purely for
    deterministic output, not a ranking signal -- mirrors `fuse_rrf`'s
    tie-breaking convention in
    `apps/api/src/resolvegrid_api/retrieval.py`). `rerank_score` is the
    cross-encoder's sigmoid-activated relevance probability in `[0, 1]`
    (verified for real against `bge-reranker-base`'s actual output in
    this environment -- see module docstring; `sentence-transformers`
    applies `torch.nn.Sigmoid` by default for a `num_labels=1` model like
    this one, which BGE reranker models are), not a fused/normalized
    score comparable to `fuse_rrf`'s RRF scores.

    `candidates=[]` returns `[]` immediately without loading the model --
    an empty result list is a common real case (e.g. a query with no
    retrieval hits) and must not pay a multi-second model-load cost, or
    fail, just to produce an empty answer.

    Raises `ValueError` if `query` is empty/whitespace-only (a reranker
    has nothing to rank candidates against without a query -- this is a
    caller bug, not a runtime failure, so it is distinguished from
    `RerankError` below).

    Raises `RerankError` if the model fails to load (see `_get_model`) or
    if inference itself fails -- never a silent wrong-shape return.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    if not candidates:
        return []

    reranker_model = _get_model(model)

    chunk_ids = [chunk_id for chunk_id, _ in candidates]
    texts = [text for _, text in candidates]

    try:
        raw_scores = reranker_model.predict(
            [(query, text) for text in texts],
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001 -- see `_get_model`'s comment;
        # a real inference-time failure (e.g. an out-of-memory error on a
        # very large batch) must surface as RerankError, not a raw
        # torch/sentence-transformers exception.
        raise RerankError(f"reranker inference failed: {exc}") from exc

    if len(raw_scores) != len(candidates):
        # Defensive: would only happen if sentence-transformers' contract
        # changed underneath this module. Caught here rather than letting
        # a caller silently zip a wrong-length score list against
        # chunk_ids and get misattributed scores.
        raise RerankError(
            f"reranker returned {len(raw_scores)} score(s) for "
            f"{len(candidates)} candidate(s) -- expected exactly one per "
            f"candidate"
        )

    scored = list(zip(chunk_ids, (float(score) for score in raw_scores)))
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))
