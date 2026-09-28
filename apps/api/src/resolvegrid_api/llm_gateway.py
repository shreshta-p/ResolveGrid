import json
import os
import time
from dataclasses import dataclass

import httpx

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-resolvegrid-local-dev")
DEFAULT_MODEL = "local-qwen3"

# Phase 11 Task 4: per-provider LiteLLM virtual keys with real budget caps
# (generated via LiteLLM's real `/key/generate` admin API against the live
# proxy -- see docs/SECURITY.md's "Phase 11 Task 4" section for how and why).
# `None` when unset (e.g. a fresh dev environment that hasn't generated them
# yet) -- `_resolve_api_key` below degrades to the shared master key in that
# case, matching this module's existing "missing config degrades to a safe
# default" precedent (`LITELLM_MASTER_KEY` itself already does this) rather
# than raising. NEVER log/print these values -- same standing constraint as
# `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.
LITELLM_CLOUD_PRIMARY_KEY = os.environ.get("LITELLM_CLOUD_PRIMARY_KEY")
LITELLM_CLOUD_FALLBACK_KEY = os.environ.get("LITELLM_CLOUD_FALLBACK_KEY")

# Maps a LiteLLM model_name (from infra/litellm/config.yaml) to the real
# provider that serves it. Used to derive CompletionResult.provider correctly
# even after a fallback (see complete()'s docstring) -- a small, explicit
# mapping rather than trying to parse it out of the model string, since
# there are only 3 model_names defined today and this stays trivially
# correct as more are added.
_MODEL_GROUP_TO_PROVIDER = {
    "local-qwen3": "ollama",
    "cloud-primary": "anthropic",
    "cloud-fallback": "openai",
}

# Only local-qwen3 (Ollama) understands/tolerates the "think" field -- both
# Anthropic and OpenAI's real APIs reject unrecognized request parameters
# outright ("Extra inputs are not permitted" / "Unrecognized request
# argument"), and LiteLLM's drop_params setting does not strip it for them.
# Discovered via a real forced-fallback call during Phase 5 fresh-state
# verification: sending "think": false unconditionally broke EVERY
# cloud-primary/cloud-fallback call with a 400, regardless of whether the
# target model itself was valid.
_THINK_FALSE_MODELS = {DEFAULT_MODEL}

# Phase 11 Task 4: which real virtual key a given LiteLLM `model_name`
# authenticates as. `local-qwen3` deliberately has NO entry here -- it keeps
# using the shared master key unconditionally (see `_resolve_api_key`'s
# fallback below), since Ollama has no real per-call cost and therefore
# nothing worth a per-provider budget cap. Built as an explicit dict, not a
# formula, for the same reason `_MODEL_GROUP_TO_PROVIDER` above is: there are
# only 2 budget-capped model_names today and this stays trivially correct as
# more are added.
_MODEL_TO_VIRTUAL_KEY = {
    "cloud-primary": LITELLM_CLOUD_PRIMARY_KEY,
    "cloud-fallback": LITELLM_CLOUD_FALLBACK_KEY,
}


def _resolve_api_key(model: str) -> str:
    """Return the real bearer token `complete()` should authenticate to the
    LiteLLM proxy with for a given `model_name` -- the model's own
    budget-capped virtual key when one has been generated and stored in
    `.env` (see module-level docstring comment), otherwise the shared
    `LITELLM_MASTER_KEY` (the pre-Task-4 behavior, still correct for
    `local-qwen3` and for any environment that hasn't generated virtual keys
    yet). This selects which CALLER identity/budget authenticates to the
    LiteLLM proxy -- it has no bearing on which real upstream Anthropic/
    OpenAI credential LiteLLM itself uses server-side to actually serve the
    request (that's `infra/litellm/config.yaml`'s `api_key: os.environ/
    ANTHROPIC_API_KEY`/`OPENAI_API_KEY`, unchanged by this task).

    Code-review fix: checks `is not None`, not truthiness -- an env var
    that's explicitly set to `""` is a real, deliberate override (matching
    `LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", <default>)`'s
    own semantics, where the default only applies when the var is fully
    ABSENT, never when it's present-but-empty) and must be returned as-is,
    not silently swapped for the master key the way a bare truthy check
    would.
    """
    virtual_key = _MODEL_TO_VIRTUAL_KEY.get(model)
    return virtual_key if virtual_key is not None else LITELLM_MASTER_KEY


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    provider: str
    model: str
    # Defaulted (not required positional) so existing callers that construct
    # CompletionResult directly -- e.g. apps/api/tests/test_ticket_summarize.py's
    # mocks, which predate this phase and are out of scope to edit here --
    # keep working unchanged. False/None matches the non-fallback case, which
    # is what those pre-existing callers are simulating anyway.
    fallback_occurred: bool = False
    serving_model_group: str | None = None


class LLMGatewayError(Exception):
    pass


def provider_for_model(model: str) -> str:
    """Public lookup of the real provider that serves a given LiteLLM
    `model_name` (from `infra/litellm/config.yaml`) -- e.g.
    `"cloud-primary"` -> `"anthropic"`. Exposed (not left as the private
    `_MODEL_GROUP_TO_PROVIDER` dict this module already keeps for
    `complete()`'s own post-response provider derivation) specifically so
    `model_call_logging.log_completion`'s error path (Phase 11 Task 4) can
    derive a real provider for an ERROR-shaped `ModelCall` row too --
    before this task, that path hardcoded `provider="ollama"` unconditionally
    (correct only because no non-local model was ever actually requested
    yet; see that module's pre-Task-4 comment, which flagged this exact gap
    by name). Once real routing (this task) can request `cloud-primary`/
    `cloud-fallback`, a failed call to either must not be mislabeled
    "ollama" in its own error row. Falls back to `"unknown"` for a
    `model_name` this mapping doesn't recognize, matching `complete()`'s own
    degrade-rather-than-raise precedent for an unrecognized
    `serving_model_group`/`model` string.
    """
    return _MODEL_GROUP_TO_PROVIDER.get(model, "unknown")


def complete(prompt: str, *, model: str = DEFAULT_MODEL, timeout_seconds: float = 60.0) -> CompletionResult:
    """Call the LiteLLM proxy's OpenAI-compatible chat completions endpoint.

    IMPORTANT: `"think": False` is deliberately included in the request body.
    qwen3:14b is a reasoning model with "thinking" mode ON by default via
    Ollama; empirically, thinking mode costs ~60x the tokens/latency for a
    trivial non-reasoning task like ticket summarization (67s/63 tokens cold,
    vs 0.165s/3 tokens with think=False). This passes through LiteLLM's
    OpenAI-compatible endpoint to Ollama correctly, despite `drop_params:
    true` in the proxy config -- that setting only drops unsupported
    *standard* OpenAI params, not custom passthrough fields like `think`. Do
    not remove this without re-verifying the cost/latency impact.

    `timeout_seconds` defaults to 60s; steady-state calls (think=False) run
    well under a second, but a cold Ollama model load (nothing resident in
    GPU memory) can plausibly still exceed 60s independent of thinking mode
    -- a cold start is a real, not-yet-eliminated way for this to time out.
    Callers doing a first-call-after-idle need to be prepared for that,
    e.g. by raising timeout_seconds or surfacing a "warming up" message
    rather than treating a timeout here as a hard failure.

    `CompletionResult.provider` is derived from `_MODEL_GROUP_TO_PROVIDER`,
    keyed by `serving_model_group` when present (the model that actually
    served the call, which may differ from `model` after a fallback) or
    `model` otherwise (Ollama's local-qwen3 has no fallback config, so it
    never gets a serving_model_group header at all).

    Raises LLMGatewayError uniformly for network failures, non-2xx
    responses, AND a malformed/unparseable response body -- callers should
    only ever need to catch this one exception type, never a raw
    KeyError/IndexError/JSONDecodeError from this function.

    `CompletionResult.fallback_occurred`/`serving_model_group` are read from
    the `x-litellm-attempted-fallbacks`/`x-litellm-model-group` response
    headers, not the JSON body -- LiteLLM only signals a fallback via
    headers (empirically verified against live Anthropic/OpenAI calls). A
    missing/malformed header degrades to fallback_occurred=False rather than
    raising, since losing that signal for one call is preferable to failing
    an otherwise-successful completion over a header-parsing quirk.
    """
    start = time.monotonic()
    request_body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model in _THINK_FALSE_MODELS:
        request_body["think"] = False
    try:
        response = httpx.post(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {_resolve_api_key(model)}"},
            json=request_body,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]["content"]
    except httpx.HTTPError as exc:
        raise LLMGatewayError(str(exc)) from exc
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMGatewayError(f"malformed response from LLM gateway: {exc}") from exc

    latency_ms = int((time.monotonic() - start) * 1000)
    usage = data.get("usage", {})

    # LiteLLM signals fallback via response HEADERS, never the JSON body
    # (empirically verified against live Anthropic/OpenAI calls -- see
    # docs/superpowers/plans/2026-08-25-phase5-cloud-fallback.md's "Task 1
    # status"). A missing/malformed x-litellm-attempted-fallbacks header
    # must degrade to "no fallback" rather than raising LLMGatewayError --
    # losing the fallback signal for one call is a much smaller regression
    # than turning an otherwise-successful completion into a hard failure.
    try:
        fallback_occurred = int(response.headers.get("x-litellm-attempted-fallbacks", "0")) > 0
    except (ValueError, TypeError):
        fallback_occurred = False
    serving_model_group = response.headers.get("x-litellm-model-group")
    provider = _MODEL_GROUP_TO_PROVIDER.get(serving_model_group or model, "unknown")

    return CompletionResult(
        text=choice,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        provider=provider,
        model=model,
        fallback_occurred=fallback_occurred,
        serving_model_group=serving_model_group,
    )
