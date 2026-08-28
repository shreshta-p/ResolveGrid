"""Tests for `resolvegrid_retrieval.context_budget`.

Per this task's instructions, exact per-entry token costs are computed
via the module's own `estimate_token_count` (the same estimator being
tested against, mirroring `test_dedup.py`'s precedent of asserting real
measured values rather than guessed-at round numbers) so `max_tokens`
budgets in these tests are derived from real entry sizes, not fragile
hand-counted approximations.
"""

from resolvegrid_retrieval.context_budget import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    ContextBlock,
    assemble_context,
)
from resolvegrid_retrieval.tokenizer import estimate_token_count


def _entry_text(chunk_id: int, title: str, text: str) -> str:
    """Reproduces `context_budget._format_entry`'s exact format, kept
    independent here (rather than importing the private helper) so these
    tests would actually catch a drift between the module's real output
    format and what this test file expects.
    """
    return f'[chunk:{chunk_id}] (from "{title}"):\n{text}'


def _entry_tokens(chunk_id: int, title: str, text: str) -> int:
    return estimate_token_count(_entry_text(chunk_id, title, text))


# Real corpus content, matching the style of test_dedup.py's fixtures
# (eval/corpus/kestrel-*.md) so these aren't synthetic placeholder
# strings picked purely for convenience.
_PTO_CHUNK_ID = 10
_PTO_TITLE = "Kestrel Time-Off Request Policy"
_PTO_TEXT = (
    "Full-time employees accrue PTO at a fixed rate per pay period based on "
    "tenure, up to an annual cap. Unused PTO up to 40 hours carries over "
    "into the next calendar year."
)

_VPN_CHUNK_ID = 21
_VPN_TITLE = "Kestrel VPN Policy v2"
_VPN_TEXT = (
    "VPN passwords must be at least 12 characters and are rotated every 90 "
    "days. Multi-factor authentication is mandatory for every VPN session, "
    "with no opt-out, for all employees and contractors."
)

_RESET_CHUNK_ID = 30
_RESET_TITLE = "Public: What Is a VPN"
_RESET_TEXT = (
    "To reset your VPN password, open the KestrelConnect client and select "
    '"Forgot Password" on the sign-in screen.'
)


def test_assemble_context_empty_input_returns_empty_block():
    result = assemble_context([])

    assert result == ContextBlock(text="", chunk_ids=[], dropped_chunk_ids=[])


def test_assemble_context_all_candidates_fit_comfortably_under_budget():
    """No truncation/dropping needed -- and the assembled text must match
    exactly what the graph's current `_build_context_block` produces
    (`"\\n\\n".join` of `[chunk:<id>] (from "<title>"):\\n<text>` entries),
    so a later wiring task (Task 7) is a clean drop-in swap.
    """
    chunks = [
        (_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT),
        (_VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT),
        (_RESET_CHUNK_ID, _RESET_TITLE, _RESET_TEXT),
    ]

    result = assemble_context(chunks, max_tokens=DEFAULT_CONTEXT_TOKEN_BUDGET)

    expected_text = "\n\n".join(
        _entry_text(chunk_id, title, text) for chunk_id, title, text in chunks
    )
    assert result.text == expected_text
    assert result.chunk_ids == [_PTO_CHUNK_ID, _VPN_CHUNK_ID, _RESET_CHUNK_ID]
    assert result.dropped_chunk_ids == []


def test_assemble_context_excludes_lower_priority_chunks_once_budget_is_exceeded():
    """Budget set to exactly cover the first two (higher-rerank-priority)
    chunks' real measured entry cost, with nothing left for the third
    (lowest-priority) chunk -- the first two must survive in order, the
    third must be dropped, not silently truncated.
    """
    chunks = [
        (_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT),
        (_VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT),
        (_RESET_CHUNK_ID, _RESET_TITLE, _RESET_TEXT),
    ]
    exact_budget_for_first_two = _entry_tokens(
        _PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT
    ) + _entry_tokens(_VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT)

    result = assemble_context(chunks, max_tokens=exact_budget_for_first_two)

    assert result.chunk_ids == [_PTO_CHUNK_ID, _VPN_CHUNK_ID]
    assert result.dropped_chunk_ids == [_RESET_CHUNK_ID]
    expected_text = "\n\n".join(
        [
            _entry_text(_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT),
            _entry_text(_VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT),
        ]
    )
    assert result.text == expected_text


def test_assemble_context_one_token_short_drops_only_the_last_chunk():
    """One token less than the exact-fit budget: the third chunk's entry
    no longer fits (even by one token) and must be dropped -- confirms
    the fit check is a real `<=` comparison against remaining budget, not
    an off-by-one-tolerant approximation.
    """
    chunks = [
        (_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT),
        (_VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT),
    ]
    exact_fit = _entry_tokens(_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT) + _entry_tokens(
        _VPN_CHUNK_ID, _VPN_TITLE, _VPN_TEXT
    )

    result = assemble_context(chunks, max_tokens=exact_fit - 1)

    assert result.chunk_ids == [_PTO_CHUNK_ID]
    assert result.dropped_chunk_ids == [_VPN_CHUNK_ID]


def test_assemble_context_skips_oversized_chunk_but_continues_to_smaller_ones():
    """The single-oversized-chunk edge case (Phase 7's chunker documents
    that one un-splittable paragraph can become its own oversized chunk).
    The higher-priority (first) chunk here is deliberately larger than
    the *entire* budget on its own -- per this module's skip-not-truncate
    decision, it must be dropped whole (never partially included), and
    assembly must continue past it to the smaller, lower-priority chunk
    that does fit, rather than stopping the whole pass at the first miss.
    """
    oversized_chunk_id = 99
    oversized_title = "Kestrel Employee Handbook"
    # 400 space-separated content words -> exactly 400 tokens under this
    # repo's word-regex estimator (no punctuation to add extra tokens),
    # deliberately larger than any reasonable small test budget below.
    oversized_text = " ".join(f"policy{i}" for i in range(400))

    chunks = [
        (oversized_chunk_id, oversized_title, oversized_text),
        (_RESET_CHUNK_ID, _RESET_TITLE, _RESET_TEXT),
    ]
    small_budget = _entry_tokens(_RESET_CHUNK_ID, _RESET_TITLE, _RESET_TEXT)
    assert (
        _entry_tokens(oversized_chunk_id, oversized_title, oversized_text)
        > small_budget
    ), "test setup: the oversized chunk must not fit in the small budget"

    result = assemble_context(chunks, max_tokens=small_budget)

    assert result.chunk_ids == [_RESET_CHUNK_ID]
    assert result.dropped_chunk_ids == [oversized_chunk_id]
    # Skip, not truncate: none of the oversized chunk's content appears
    # in the assembled text, not even a partial slice of it.
    assert "policy0" not in result.text
    assert str(oversized_chunk_id) not in result.text
    assert result.text == _entry_text(_RESET_CHUNK_ID, _RESET_TITLE, _RESET_TEXT)


def test_assemble_context_zero_budget_drops_everything():
    chunks = [(_PTO_CHUNK_ID, _PTO_TITLE, _PTO_TEXT)]

    result = assemble_context(chunks, max_tokens=0)

    assert result == ContextBlock(
        text="", chunk_ids=[], dropped_chunk_ids=[_PTO_CHUNK_ID]
    )
