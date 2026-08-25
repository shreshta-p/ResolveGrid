import os
import time
from dataclasses import dataclass

import httpx

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-resolvegrid-local-dev")
DEFAULT_MODEL = "local-qwen3"


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    provider: str
    model: str


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
    """
    start = time.monotonic()
    try:
        response = httpx.post(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "think": False},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMGatewayError(str(exc)) from exc

    latency_ms = int((time.monotonic() - start) * 1000)
    data = response.json()
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return CompletionResult(
        text=choice,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        provider="ollama",
        model=model,
    )
