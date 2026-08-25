from unittest.mock import MagicMock, patch

import httpx
import pytest

from resolvegrid_api.llm_gateway import CompletionResult, LLMGatewayError, complete


def _mock_response(status_code=200, json_data=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data or {}
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
