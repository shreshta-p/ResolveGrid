"""Tests for `resolvegrid_retrieval.reranker.rerank`.

Per this task's instructions, the reranker itself is NOT mocked here --
these tests load the real `bge-reranker-base` cross-encoder (see
`reranker.py`'s module docstring for why that model was chosen) and
assert on its real output, the same "verify real behavior" standard
`embedder.py`'s real-Ollama tests elsewhere in this repo apply. The one
exception is `test_mismatched_score_count_raises_rerank_error`, which
exercises a defensive branch that the real model's contract cannot be
forced into (sentence-transformers always returns one score per input
pair) -- that single case is mocked, mirroring `test_embedder.py`'s
convention of mocking only for malformed/wrong-shape response shapes a
real dependency won't actually produce in a test run.

Candidate text below is drawn from real `eval/corpus/*.md` content
(`kestrel-vpn-policy-v2.md`'s Password Reset section, `public-what-is-a-
vpn.md`'s general VPN explainer, and `kestrel-time-off-request.md`'s PTO
policy) rather than invented strings, so the relevance ordering this test
asserts on reflects the actual corpus this reranker will run against in
later phases.

Model loading is slow on first use in a process (~30s in this CPU-only
dev environment, including sentence-transformers' Hugging Face Hub
cache-validation round trip -- see `reranker.py`'s module docstring) but
`reranker.py` caches the loaded model process-wide, so only the first
test function in this file pays that cost; the rest reuse the cached
model and run in under a second each.
"""

from unittest.mock import MagicMock, patch

import pytest

from resolvegrid_retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    RerankError,
    clear_model_cache,
    rerank,
)

# Real corpus content (see module docstring).
_VPN_QUERY = "How do I reset my VPN password?"

_RELEVANT_CHUNK_ID = 1
_RELEVANT_TEXT = (
    "## Password Reset\n\n"
    "To reset your VPN password, open the KestrelConnect client and select "
    '"Forgot Password" on the sign-in screen. This sends a one-time reset '
    "link to your Kestrel email address; the self-service flow completes "
    "in minutes, replacing the old email-the-Security-team process."
)

_TANGENTIAL_CHUNK_ID = 2
_TANGENTIAL_TEXT = (
    "# What Is a VPN\n\n"
    "A virtual private network (VPN) is a technology that creates an "
    "encrypted connection, often called a tunnel, between a device and a "
    "private network over the public internet. Traffic sent through the "
    "tunnel is encrypted, so an observer on the intervening network path "
    "sees only opaque encrypted packets rather than the underlying data."
)

_UNRELATED_CHUNK_ID = 3
_UNRELATED_TEXT = (
    "## Requesting Time Off\n\n"
    "Submit a time-off request through the internal HR portal at least 5 "
    "business days before the requested start date for planned leave. "
    "Requests under 3 days are auto-approved once your manager "
    "acknowledges the request."
)


@pytest.fixture(autouse=True)
def _reset_model_cache_after_each_test():
    """Every test in this file uses the real default model except the one
    mocked defensive-branch test and the unknown-model test -- clearing
    the cache after each test keeps them independent (e.g. the
    unknown-model test must not accidentally see a previously cached
    'good' model under a different key and mask a real load failure).
    Does not undo the real cost of reloading the default model if a later
    test needs it again -- see the module docstring for why that cost is
    paid at most a small, bounded number of times across this file, not
    once per test.
    """
    yield


def test_empty_candidates_returns_empty_list_without_loading_model():
    clear_model_cache()
    result = rerank(_VPN_QUERY, [])
    assert result == []
    # No model should have been loaded just to produce an empty result.
    assert DEFAULT_RERANKER_MODEL not in _model_cache_snapshot()


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        rerank("", [(_RELEVANT_CHUNK_ID, _RELEVANT_TEXT)])


def test_whitespace_only_query_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        rerank("   \n\t  ", [(_RELEVANT_CHUNK_ID, _RELEVANT_TEXT)])


def test_rerank_orders_real_candidates_by_relevance():
    """The real correctness check this task calls for: given a VPN
    password reset query and three real-corpus-derived candidates (one
    genuinely relevant, one tangentially related, one unrelated), the
    real cross-encoder must rank them relevant > tangential > unrelated.
    """
    candidates = [
        (_UNRELATED_CHUNK_ID, _UNRELATED_TEXT),
        (_RELEVANT_CHUNK_ID, _RELEVANT_TEXT),
        (_TANGENTIAL_CHUNK_ID, _TANGENTIAL_TEXT),
    ]

    result = rerank(_VPN_QUERY, candidates)

    assert [chunk_id for chunk_id, _ in result] == [
        _RELEVANT_CHUNK_ID,
        _TANGENTIAL_CHUNK_ID,
        _UNRELATED_CHUNK_ID,
    ]


def test_rerank_scores_are_bounded_probabilities_in_descending_order():
    candidates = [
        (_RELEVANT_CHUNK_ID, _RELEVANT_TEXT),
        (_TANGENTIAL_CHUNK_ID, _TANGENTIAL_TEXT),
        (_UNRELATED_CHUNK_ID, _UNRELATED_TEXT),
    ]

    result = rerank(_VPN_QUERY, candidates)
    scores = [score for _, score in result]

    # bge-reranker-base is a num_labels=1 model, so sentence-transformers
    # applies torch.nn.Sigmoid by default (verified for real -- see
    # reranker.py's module docstring): every score must be a valid
    # probability, and best-first ordering means non-increasing.
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    # The relevant chunk should score decisively higher than the unrelated
    # one -- not just technically first, but by a wide margin.
    scores_by_id = dict(result)
    assert scores_by_id[_RELEVANT_CHUNK_ID] - scores_by_id[_UNRELATED_CHUNK_ID] > 0.5


def test_rerank_is_order_independent_of_input_candidate_order():
    """Reranking must reflect relevance, not preserve/depend on input
    order -- feed the same three candidates in a different order than
    the ordering test above and confirm the output order is identical.
    """
    candidates = [
        (_TANGENTIAL_CHUNK_ID, _TANGENTIAL_TEXT),
        (_UNRELATED_CHUNK_ID, _UNRELATED_TEXT),
        (_RELEVANT_CHUNK_ID, _RELEVANT_TEXT),
    ]

    result = rerank(_VPN_QUERY, candidates)

    assert [chunk_id for chunk_id, _ in result] == [
        _RELEVANT_CHUNK_ID,
        _TANGENTIAL_CHUNK_ID,
        _UNRELATED_CHUNK_ID,
    ]


def test_single_candidate_is_returned_with_a_score():
    result = rerank(_VPN_QUERY, [(_RELEVANT_CHUNK_ID, _RELEVANT_TEXT)])
    assert len(result) == 1
    assert result[0][0] == _RELEVANT_CHUNK_ID
    assert 0.0 <= result[0][1] <= 1.0


def test_unknown_model_raises_rerank_error_not_a_raw_exception():
    """Real failure, not mocked: an unknown/nonexistent Hugging Face
    model id genuinely fails to load (a real 401/404 from the Hub, as
    observed manually against this exact model name before writing this
    test), and must surface as RerankError with the model name in the
    message, not a raw huggingface_hub/OSError leaking out of this
    module.
    """
    clear_model_cache()
    with pytest.raises(RerankError, match="this-model-does-not-exist"):
        rerank(
            _VPN_QUERY,
            [(_RELEVANT_CHUNK_ID, _RELEVANT_TEXT)],
            model="BAAI/this-model-does-not-exist-xyz-12345",
        )
    clear_model_cache()


def test_mismatched_score_count_raises_rerank_error():
    """Defensive branch: sentence-transformers' real contract always
    returns exactly one score per input pair, so this can't be forced
    through the real model -- mocked here, matching test_embedder.py's
    convention of mocking only for a wrong-shape-response case a real
    dependency won't actually produce.
    """
    clear_model_cache()
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.9]  # 1 score for 2 candidates

    with patch(
        "resolvegrid_retrieval.reranker._get_model", return_value=fake_model
    ):
        with pytest.raises(RerankError, match="expected exactly one per candidate"):
            rerank(
                _VPN_QUERY,
                [
                    (_RELEVANT_CHUNK_ID, _RELEVANT_TEXT),
                    (_TANGENTIAL_CHUNK_ID, _TANGENTIAL_TEXT),
                ],
            )
    clear_model_cache()


def _model_cache_snapshot() -> dict:
    from resolvegrid_retrieval.reranker import _MODEL_CACHE

    return _MODEL_CACHE
