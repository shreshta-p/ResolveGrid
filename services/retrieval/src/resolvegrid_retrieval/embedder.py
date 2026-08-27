"""Embedding generation via Ollama's `/api/embed` endpoint (Phase 7 Task 3).

Endpoint choice, verified rather than assumed (per this task's instructions):
Ollama exposes two embeddings endpoints. `/api/embeddings` (singular) is the
older one -- it takes one `prompt` string and returns one `embedding` list.
`/api/embed` (plural) is the current documented endpoint -- it takes an
`input` field that accepts either a single string or a *list* of strings,
and returns an `embeddings` field containing one vector per input, in the
same order. Both were exercised against a real local Ollama 0.32.15
container (`docker exec resolvegrid-ollama ollama pull nomic-embed-text`,
then real HTTP calls) during this task: `/api/embed` with a 2-element
`input` list returned exactly 2 vectors, confirming real batch support. This
module uses `/api/embed` for that batching, matching this repo's general
preference for the currently-documented API over a legacy one it happens to
also still accept.

Direct-to-Ollama vs. via LiteLLM, decided and documented (per this task's
instructions to check `infra/litellm/config.yaml` before adding a duplicate
path): `infra/litellm/config.yaml` defines no embedding `model_name` today
(only `local-qwen3`/`cloud-primary`/`cloud-fallback` chat models), so there
is no existing embeddings route to reuse or duplicate. `apps/api/llm_gateway.py`
always calls chat completions through the LiteLLM proxy rather than Ollama
directly, for provider abstraction (Anthropic/OpenAI cloud fallback) that
only makes sense for chat. Embeddings in this phase have no cloud-fallback
requirement (`nomic-embed-text` is the only embedding model this phase
uses, per `apps/api/src/resolvegrid_api/models/knowledge.py`), and
`services/retrieval` is a plain library with zero runtime dependencies
today (see `pyproject.toml`) that should not gain a dependency on
`apps/api`'s LiteLLM master-key/env-var conventions just to reach Ollama.
Calling Ollama directly is therefore both simpler and does not foreclose a
future move behind LiteLLM (adding a `model_name: nomic-embed-text` entry
with `litellm_params.model: ollama/nomic-embed-text` later) if a cloud
embedding fallback is ever needed.
"""

import os

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Empirically confirmed (not assumed) against a real Ollama 0.32.15 container
# running nomic-embed-text, 2026-08-27: POST /api/embed with
# {"model": "nomic-embed-text", "input": "hello world"} returned an
# `embeddings` list containing exactly one 768-length float vector. This
# matches EMBEDDING_DIM in
# apps/api/src/resolvegrid_api/models/knowledge.py, which that task's
# author had explicitly flagged as model-card-derived and NOT locally
# verified (nomic-embed-text wasn't pulled in that dev environment at the
# time) -- this constant, and the real call behind it, is that local
# verification. No correction to migration 0010 / EMBEDDING_DIM was needed
# as a result. If a future embedding model swap changes this dimension,
# EMBEDDING_DIM *and* migration 0010's `Vector(768)` column must be updated
# together with this constant -- a pgvector fixed-dimension column cannot
# silently accept a different-sized vector (it raises at insert time).
NOMIC_EMBED_TEXT_DIM = 768

# Ollama's /api/embed accepts an arbitrary-length `input` list in one
# request (confirmed above), but this module still caps how many texts go
# into a single HTTP POST -- an unbounded request body for a large ingestion
# batch is an avoidable failure mode (request size limits, one slow/failed
# call losing an entire large batch's work) that a real ingestion job
# (a later Phase 7 task) will call this with many chunks at once.
DEFAULT_BATCH_SIZE = 64


class EmbeddingError(Exception):
    """Raised for any failure embedding text via Ollama: network errors,
    non-2xx responses (including the model-not-pulled case -- see
    `embed_texts()`'s docstring), or a malformed/wrong-shape response body.
    Callers should only ever need to catch this one exception type, never a
    raw `httpx.HTTPError`/`KeyError`/`ValueError` from this module -- this
    mirrors `apps/api/src/resolvegrid_api/llm_gateway.py`'s
    `LLMGatewayError` convention.
    """


def embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout_seconds: float = 60.0,
) -> list[list[float]]:
    """Embed `texts` via Ollama's `/api/embed` endpoint, returning one
    vector per input text, in the same order.

    `texts=[]` returns `[]` immediately without making any network call.

    Internally batches `texts` into groups of at most `batch_size` and
    issues one `/api/embed` POST per group (see module docstring for why
    batching is capped rather than sent as a single unbounded request).

    Raises `EmbeddingError` if:
      - `model` isn't pulled into the target Ollama instance yet. Verified
        real shape: Ollama returns HTTP 404 with body
        `{"error": "model \\"<name>\\" not found, try pulling it first"}`
        for an unpulled model name -- the raised message names the model
        and suggests `ollama pull <model>` rather than surfacing that raw
        JSON, per this task's "clear error, not a silent wrong-shape
        return" requirement;
      - any other network/HTTP failure occurs;
      - the response is missing `embeddings`, returns a different number
        of vectors than input texts in that batch, or the vectors within
        one batch have inconsistent lengths -- any of these would
        otherwise silently produce wrong-shape output for a caller that
        just trusts the return value's length/dimension.
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(
            _embed_batch(batch, model=model, timeout_seconds=timeout_seconds)
        )
    return vectors


def _embed_batch(
    batch: list[str], *, model: str, timeout_seconds: float
) -> list[list[float]]:
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": batch},
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            f"failed to reach Ollama embeddings endpoint at {OLLAMA_BASE_URL}: {exc}"
        ) from exc

    if response.status_code == 404:
        # Verified real shape (see module docstring):
        # {"error": "model \"X\" not found, try pulling it first"}
        detail = ""
        try:
            detail = response.json().get("error", "")
        except Exception:  # noqa: BLE001 -- best-effort detail only; a
            # malformed error body must not hide the clear message below.
            pass
        raise EmbeddingError(
            f"embedding model '{model}' is not pulled on the Ollama instance "
            f"at {OLLAMA_BASE_URL} -- run `ollama pull {model}` first"
            + (f" (Ollama said: {detail})" if detail else "")
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(f"Ollama embeddings request failed: {exc}") from exc

    try:
        data = response.json()
        embeddings = data["embeddings"]
    except (ValueError, KeyError, TypeError) as exc:
        raise EmbeddingError(
            f"malformed response from Ollama embeddings endpoint: {exc}"
        ) from exc

    if not isinstance(embeddings, list) or len(embeddings) != len(batch):
        raise EmbeddingError(
            f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 'a non-list'} "
            f"embedding(s) for {len(batch)} input text(s) -- expected exactly one per input"
        )

    dims = {len(vector) for vector in embeddings}
    if len(dims) > 1:
        raise EmbeddingError(
            f"Ollama returned inconsistent embedding dimensions within one batch: {sorted(dims)}"
        )

    return embeddings
