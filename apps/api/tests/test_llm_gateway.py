from unittest.mock import MagicMock, patch

import httpx
import pytest

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
