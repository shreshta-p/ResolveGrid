import json
import os
import time
from dataclasses import dataclass

import httpx

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-resolvegrid-local-dev")
DEFAULT_MODEL = "local-qwen3"

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
            headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
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
