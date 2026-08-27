"""Unit tests for `resolvegrid_retrieval.embedder.embed_texts`.

Mocking convention matches `apps/api/tests/test_llm_gateway.py` (`patch`
on the module's `httpx.post`, `MagicMock(spec=httpx.Response)`) -- this
module's real Ollama round-trip is exercised separately, against a real
Ollama container, by `apps/api/tests/test_knowledge_store.py` (Phase 7
Task 3's "no mocking" integration test), not here.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from resolvegrid_retrieval.embedder import EmbeddingError, embed_texts


def _mock_response(status_code=200, json_data=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data if json_data is not None else {}
    if status_code >= 400 and status_code != 404:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def test_empty_input_returns_empty_list_without_a_network_call():
    with patch("resolvegrid_retrieval.embedder.httpx.post") as mock_post:
        result = embed_texts([])
    assert result == []
    mock_post.assert_not_called()


def test_embed_texts_returns_one_vector_per_input_in_order():
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_response = _mock_response(200, {"embeddings": vectors})

    with patch(
        "resolvegrid_retrieval.embedder.httpx.post", return_value=mock_response
    ) as mock_post:
        result = embed_texts(["first text", "second text"])

    assert result == vectors
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["input"] == ["first text", "second text"]
    assert sent_body["model"] == "nomic-embed-text"


def test_embed_texts_batches_requests_at_batch_size_boundary():
    # 5 texts, batch_size=2 -> 3 requests of sizes 2, 2, 1.
    call_count = 0

    def _fake_post(url, json, timeout):
        nonlocal call_count
        call_count += 1
        n = len(json["input"])
        return _mock_response(200, {"embeddings": [[float(call_count)] * 3] * n})

    with patch("resolvegrid_retrieval.embedder.httpx.post", side_effect=_fake_post):
        result = embed_texts(["a", "b", "c", "d", "e"], batch_size=2)

    assert call_count == 3
    assert len(result) == 5


def test_model_not_found_raises_clear_embedding_error_not_raw_404():
    mock_response = _mock_response(
        404, {"error": 'model "nomic-embed-text" not found, try pulling it first'}
    )

    with patch("resolvegrid_retrieval.embedder.httpx.post", return_value=mock_response):
        with pytest.raises(EmbeddingError, match="not pulled"):
            embed_texts(["hello"])


def test_network_error_raises_embedding_error():
    with patch(
        "resolvegrid_retrieval.embedder.httpx.post",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        with pytest.raises(EmbeddingError):
            embed_texts(["hello"])


def test_mismatched_vector_count_raises_embedding_error():
    # 2 inputs, only 1 vector back -- must not silently return wrong-shape output.
    mock_response = _mock_response(200, {"embeddings": [[0.1, 0.2]]})

    with patch("resolvegrid_retrieval.embedder.httpx.post", return_value=mock_response):
        with pytest.raises(EmbeddingError, match="expected exactly one per input"):
            embed_texts(["first", "second"])


def test_inconsistent_dimensions_within_a_batch_raises_embedding_error():
    mock_response = _mock_response(
        200, {"embeddings": [[0.1, 0.2, 0.3], [0.1, 0.2]]}
    )

    with patch("resolvegrid_retrieval.embedder.httpx.post", return_value=mock_response):
        with pytest.raises(EmbeddingError, match="inconsistent embedding dimensions"):
            embed_texts(["first", "second"])


def test_missing_embeddings_key_raises_embedding_error():
    mock_response = _mock_response(200, {"unexpected": "shape"})

    with patch("resolvegrid_retrieval.embedder.httpx.post", return_value=mock_response):
        with pytest.raises(EmbeddingError, match="malformed response"):
            embed_texts(["hello"])
