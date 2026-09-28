from unittest.mock import MagicMock, patch

import httpx
import pytest

from resolvegrid_api import llm_gateway
from resolvegrid_api.llm_gateway import CompletionResult, LLMGatewayError, complete


def _mock_response(status_code=200, json_data=None, headers=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data or {}
    # Default {} correctly simulates a response with no LiteLLM fallback
    # headers present -- what a normal call to local-qwen3 looks like, since
    # only cloud-primary/cloud-fallback model groups ever carry these
    # headers. A plain dict stands in fine for httpx.Headers here since
    # llm_gateway.complete() only ever calls .get() on it.
    response.headers = headers if headers is not None else {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def test_complete_returns_parsed_completion_result():
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "The ticket is about a broken login flow."}}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 52},
        },
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response) as mock_post:
        result = complete("Summarize this ticket.")

    assert mock_post.called
    assert isinstance(result, CompletionResult)
    assert result.text == "The ticket is about a broken login flow."
    assert result.input_tokens == 128
    assert result.output_tokens == 52
    assert result.provider == "ollama"
    assert result.model == "local-qwen3"
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


def test_complete_raises_gateway_error_on_non_2xx_response():
    mock_response = _mock_response(500, {"error": "internal server error"})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        with pytest.raises(LLMGatewayError):
            complete("Summarize this ticket.")


def test_complete_raises_gateway_error_on_timeout():
    with patch(
        "resolvegrid_api.llm_gateway.httpx.post",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        with pytest.raises(LLMGatewayError):
            complete("Summarize this ticket.")


def test_complete_raises_gateway_error_on_malformed_response_body():
    # A 200 response whose body doesn't have the expected shape (e.g. an
    # empty choices list, or a proxy/error page) must still surface as
    # LLMGatewayError -- callers should never need to catch a raw
    # KeyError/IndexError/JSONDecodeError from this function.
    mock_response = _mock_response(200, {"choices": []})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        with pytest.raises(LLMGatewayError):
            complete("Summarize this ticket.")


def test_complete_sends_think_false_in_request_body():
    """The single most important behavior in this module.

    qwen3:14b has "thinking" mode ON by default; without "think": False in
    the request body, every call costs ~60x more tokens/latency for zero
    benefit on a short non-reasoning task like ticket summarization. If this
    test starts failing, someone silently regressed the request body -- do
    not "fix" the test to match, fix the request body instead.
    """
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response) as mock_post:
        complete("Summarize this ticket.")

    assert mock_post.called
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["think"] is False


def test_complete_reports_fallback_when_litellm_headers_indicate_one():
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "Fell back to OpenAI."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        headers={"x-litellm-attempted-fallbacks": "1", "x-litellm-model-group": "cloud-fallback"},
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        result = complete("Summarize this ticket.")

    assert result.fallback_occurred is True
    assert result.serving_model_group == "cloud-fallback"


def test_complete_reports_no_fallback_when_headers_absent_or_zero():
    # No headers set at all -- matches a call to local-qwen3, which has no
    # fallback config and therefore no x-litellm-* headers in its response.
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "No fallback here."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        result = complete("Summarize this ticket.")

    assert result.fallback_occurred is False
    assert result.serving_model_group is None

    # Explicit "0" is equivalent to the header being absent entirely.
    mock_response_zero = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "No fallback here."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        headers={"x-litellm-attempted-fallbacks": "0", "x-litellm-model-group": "cloud-primary"},
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response_zero):
        result_zero = complete("Summarize this ticket.")

    assert result_zero.fallback_occurred is False
    assert result_zero.serving_model_group == "cloud-primary"


def test_complete_defaults_fallback_occurred_false_on_malformed_header():
    # A malformed x-litellm-attempted-fallbacks header must not turn an
    # otherwise-successful completion into a hard LLMGatewayError -- it
    # should just degrade to fallback_occurred=False.
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
        headers={"x-litellm-attempted-fallbacks": "not-a-number"},
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        result = complete("Summarize this ticket.")

    assert result.fallback_occurred is False


def test_complete_omits_think_for_cloud_models():
    """Only local-qwen3 (Ollama) understands "think" -- Anthropic/OpenAI's
    real APIs reject unrecognized request params outright. Discovered via a
    real forced-fallback call during Phase 5 fresh-state verification: this
    field being sent unconditionally broke EVERY cloud-primary/cloud-fallback
    call with a 400, regardless of whether the target model was valid. If
    this test starts failing, someone regressed the conditional -- do not
    "fix" it to always send think=False again.
    """
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
        headers={"x-litellm-model-group": "cloud-primary"},
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response) as mock_post:
        result = complete("Summarize this ticket.", model="cloud-primary")

    assert mock_post.called
    _, kwargs = mock_post.call_args
    assert "think" not in kwargs["json"]
    assert result.provider == "anthropic"


def test_complete_derives_provider_from_serving_model_group_after_fallback():
    # model="cloud-primary" was requested, but cloud-fallback actually served
    # it -- provider must reflect the real (fallback) provider, not the
    # originally-requested one.
    mock_response = _mock_response(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
        headers={"x-litellm-attempted-fallbacks": "1", "x-litellm-model-group": "cloud-fallback"},
    )

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=mock_response):
        result = complete("Summarize this ticket.", model="cloud-primary")

    assert result.provider == "openai"


# --- Phase 11 Task 4: per-provider virtual key resolution -------------------
#
# Real, CI-safe (mocked httpx.post, no network/credentials) unit tests of
# `_resolve_api_key`/its wiring into `complete()`'s Authorization header --
# NOT the real, cloud-reaching budget-enforcement proof (that's a separate,
# local, evidence-cited verification step; see docs/SECURITY.md's "Phase 11
# Task 4" section). These only prove which virtual key `complete()` DECIDES
# to send, using fake, obviously-not-real key strings -- never touching the
# actual `.env`-held virtual keys.


def _mock_ok_response():
    return _mock_response(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )


def test_complete_authenticates_cloud_primary_calls_with_its_own_virtual_key(monkeypatch):
    monkeypatch.setattr(llm_gateway, "LITELLM_CLOUD_PRIMARY_KEY", "sk-fake-cloud-primary-virtual-key")
    monkeypatch.setattr(llm_gateway, "_MODEL_TO_VIRTUAL_KEY", {"cloud-primary": "sk-fake-cloud-primary-virtual-key"})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=_mock_ok_response()) as mock_post:
        complete("hello", model="cloud-primary")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-fake-cloud-primary-virtual-key"


def test_complete_authenticates_cloud_fallback_calls_with_its_own_virtual_key(monkeypatch):
    monkeypatch.setattr(llm_gateway, "_MODEL_TO_VIRTUAL_KEY", {"cloud-fallback": "sk-fake-cloud-fallback-virtual-key"})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=_mock_ok_response()) as mock_post:
        complete("hello", model="cloud-fallback")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-fake-cloud-fallback-virtual-key"


def test_complete_uses_shared_master_key_for_local_qwen3():
    # local-qwen3 has no entry in _MODEL_TO_VIRTUAL_KEY at all -- it must
    # keep using the shared master key unconditionally (Ollama has no real
    # per-call cost, so no per-provider budget cap is warranted for it).
    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=_mock_ok_response()) as mock_post:
        complete("hello", model="local-qwen3")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {llm_gateway.LITELLM_MASTER_KEY}"


def test_complete_falls_back_to_master_key_when_virtual_key_env_var_unset(monkeypatch):
    # A cloud model_name with no virtual key generated/configured yet (e.g. a
    # fresh dev environment) must degrade to the shared master key, not raise
    # or send a literal "Bearer None".
    monkeypatch.setattr(llm_gateway, "_MODEL_TO_VIRTUAL_KEY", {"cloud-primary": None})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=_mock_ok_response()) as mock_post:
        complete("hello", model="cloud-primary")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {llm_gateway.LITELLM_MASTER_KEY}"


def test_complete_respects_an_explicitly_empty_virtual_key_rather_than_swapping_to_master(monkeypatch):
    # Code-review fix: `_resolve_api_key` checks `is not None`, not
    # truthiness -- an env var explicitly set to "" is a deliberate
    # override (mirrors LITELLM_MASTER_KEY's own os.environ.get(key,
    # default) semantics, where the default only kicks in when the var is
    # fully ABSENT) and must be sent as-is, not silently swapped for the
    # master key the way a bare `if virtual_key` truthy check would.
    monkeypatch.setattr(llm_gateway, "_MODEL_TO_VIRTUAL_KEY", {"cloud-primary": ""})

    with patch("resolvegrid_api.llm_gateway.httpx.post", return_value=_mock_ok_response()) as mock_post:
        complete("hello", model="cloud-primary")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer "


def test_every_non_default_model_group_has_a_virtual_key_mapping_entry():
    """Important, code-review-flagged drift guard: `_MODEL_TO_VIRTUAL_KEY`
    and `_MODEL_GROUP_TO_PROVIDER` are legitimately different mappings
    (different key sets/purposes -- provider identity vs. caller-budget
    identity) and must not be merged into one. But nothing else enforces
    they stay in sync: if a future `model_name` is added to
    `_MODEL_GROUP_TO_PROVIDER`/`infra/litellm/config.yaml` and its virtual
    -key entry is simply forgotten here, `_resolve_api_key` silently
    degrades that model to the shared master key -- quietly defeating this
    whole task's per-provider budget-cap purpose, with no error/warning at
    runtime. This test is the guard: every `model_name` in
    `_MODEL_GROUP_TO_PROVIDER` OTHER than `DEFAULT_MODEL` (which
    deliberately has no entry -- Ollama has no real per-call cost, so no
    budget cap is warranted for it) must have at least a (possibly
    `None`-valued) key in `_MODEL_TO_VIRTUAL_KEY`. A future model addition
    that forgets this fails CI here instead of silently degrading budget
    enforcement in production.
    """
    real_model_names = set(llm_gateway._MODEL_GROUP_TO_PROVIDER) - {llm_gateway.DEFAULT_MODEL}
    missing = real_model_names - set(llm_gateway._MODEL_TO_VIRTUAL_KEY)
    assert not missing, (
        f"model_name(s) {missing} exist in _MODEL_GROUP_TO_PROVIDER but have no "
        "entry (not even a None placeholder) in _MODEL_TO_VIRTUAL_KEY -- a real "
        "call to one of these will silently authenticate with the shared master "
        "key instead of a per-provider budget-capped virtual key. Add an entry "
        "(a real env-var-backed key, or explicit None if this model_name is "
        "intentionally never budget-capped) to _MODEL_TO_VIRTUAL_KEY."
    )
