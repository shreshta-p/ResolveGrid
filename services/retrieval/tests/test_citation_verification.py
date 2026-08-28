"""Tests for `resolvegrid_retrieval.citation_verification`.

Citation format matches the exact `[chunk:<id>]` token
`_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE`/`_build_context_block`
(`services/agent-orchestration/.../graph.py`) instruct the model to
produce -- see `citation_verification.py`'s module docstring for the
confirmed exact text quoted from that template.
"""

from resolvegrid_retrieval.citation_verification import (
    Citation,
    VerificationResult,
    verify_citations,
)


def test_all_citations_valid():
    answer = (
        "VPN passwords must be rotated every 90 days [chunk:1], and MFA is "
        "mandatory for every session [chunk:2]."
    )
    result = verify_citations(answer, valid_chunk_ids={1, 2, 3})

    assert result.all_verified is True
    assert result.verified_chunk_ids == [1, 2]
    assert result.fabricated_chunk_ids == []
    assert [c.chunk_id for c in result.citations] == [1, 2]
    assert all(c.verified for c in result.citations)


def test_mix_of_valid_and_fabricated_citations():
    # The model hallucinates [chunk:9999] when only {1, 2, 3} were ever in
    # this context -- the exact scenario this module exists to catch.
    answer = (
        "Per the VPN policy [chunk:1], passwords rotate every 90 days. "
        "Also, all remote laptops ship with a free coffee maker [chunk:9999]."
    )
    result = verify_citations(answer, valid_chunk_ids={1, 2, 3})

    assert result.all_verified is False
    assert result.verified_chunk_ids == [1]
    assert result.fabricated_chunk_ids == [9999]
    assert result.citations == [
        Citation(
            chunk_id=1,
            verified=True,
            raw_text="[chunk:1]",
            start=answer.index("[chunk:1]"),
            end=answer.index("[chunk:1]") + len("[chunk:1]"),
        ),
        Citation(
            chunk_id=9999,
            verified=False,
            raw_text="[chunk:9999]",
            start=answer.index("[chunk:9999]"),
            end=answer.index("[chunk:9999]") + len("[chunk:9999]"),
        ),
    ]


def test_zero_citations_is_valid_not_an_error():
    # A general-knowledge answer with nothing to cite -- valid and
    # unremarkable, not a failure. There is nothing to have gotten wrong.
    answer = "In general, 2 + 2 equals 4."
    result = verify_citations(answer, valid_chunk_ids={1, 2, 3})

    assert result == VerificationResult(
        citations=[], all_verified=True, verified_chunk_ids=[], fabricated_chunk_ids=[]
    )


def test_empty_answer_text_is_valid():
    result = verify_citations("", valid_chunk_ids={1, 2, 3})

    assert result == VerificationResult(
        citations=[], all_verified=True, verified_chunk_ids=[], fabricated_chunk_ids=[]
    )


def test_citation_to_a_real_chunk_id_not_in_this_context_is_unverified():
    # chunk_id 42 is a real, persisted chunk somewhere in the corpus (it's
    # a plausible, "real-looking" id, not an absurd sentinel like 9999) --
    # but it simply was never part of THIS context (valid_chunk_ids). What
    # matters is membership in valid_chunk_ids, not whether the id exists
    # anywhere in the corpus at all. This is the easy-to-get-wrong
    # distinction the task calls out explicitly.
    answer = "The VPN password policy requires rotation every 90 days [chunk:42]."
    result = verify_citations(answer, valid_chunk_ids={1, 2, 3})

    assert result.all_verified is False
    assert result.fabricated_chunk_ids == [42]
    assert result.verified_chunk_ids == []


def test_malformed_citation_syntax_does_not_crash_and_is_not_counted():
    # [chunk:abc] (non-numeric id), [chunk 5] (missing colon), and
    # [Chunk:6] (wrong case) are not the exact instructed format -- they
    # must not raise, and must not be counted as either verified or
    # fabricated since they were never recognized as citations at all.
    answer = (
        "See [chunk:abc] for details, also [chunk 5] and [Chunk:6], but "
        "the real one is [chunk:1]."
    )
    result = verify_citations(answer, valid_chunk_ids={1})

    assert result.all_verified is True
    assert result.verified_chunk_ids == [1]
    assert result.fabricated_chunk_ids == []
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == 1


def test_duplicate_citation_to_same_id_produces_one_occurrence_per_span_but_one_id_entry():
    answer = "Rotate every 90 days [chunk:1]. Repeated for emphasis [chunk:1]."
    result = verify_citations(answer, valid_chunk_ids={1})

    assert len(result.citations) == 2
    assert result.citations[0].start != result.citations[1].start
    assert result.verified_chunk_ids == [1]
    assert result.all_verified is True


def test_valid_chunk_ids_empty_set_flags_every_citation_as_fabricated():
    answer = "Per policy [chunk:1]."
    result = verify_citations(answer, valid_chunk_ids=set())

    assert result.all_verified is False
    assert result.fabricated_chunk_ids == [1]
