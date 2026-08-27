"""Unit tests for `resolvegrid_retrieval.chunker.chunk_markdown`.

Fixture: `tests/fixtures/multi_section.md` has 4 top-level headings --
Overview, Getting Started, Appendix (each a single short paragraph, meant
to stay as one unsplit chunk) and Deep Dive (14 near-identical "lorem
ipsum" paragraphs, each tagged with a unique `DEEPDIVE-P<n>` marker,
meant to force sub-splitting). Each section's paragraphs carry a unique
marker string so cross-section content mixing is mechanically detectable
rather than eyeballed.
"""

from pathlib import Path

import pytest

from resolvegrid_retrieval.chunker import ChunkRecord, chunk_markdown
from resolvegrid_retrieval.tokenizer import estimate_token_count

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_SECTION_MARKERS = {
    "Overview": "OVERVIEW-MARKER",
    "Getting Started": "GETTING-STARTED-MARKER",
    "Deep Dive": "DEEPDIVE-P",  # prefix shared by all 14 Deep Dive paragraphs
    "Appendix": "APPENDIX-MARKER",
}


def _load_multi_section() -> str:
    return (FIXTURES_DIR / "multi_section.md").read_text(encoding="utf-8")


def _markers_present(text: str) -> set[str]:
    return {name for name, marker in _SECTION_MARKERS.items() if marker in text}


# --- tokenizer ------------------------------------------------------------


def test_estimate_token_count_is_computed_not_hardcoded():
    # "Hello, world." -> "Hello", ",", "world", "." == 4 tokens under the
    # word-or-punctuation regex tokenizer. This pins down the exact
    # counting method (not just "returns something nonzero").
    assert estimate_token_count("Hello, world.") == 4


def test_estimate_token_count_empty_and_whitespace_is_zero():
    assert estimate_token_count("") == 0
    assert estimate_token_count("   \n\t  ") == 0


def test_estimate_token_count_scales_with_text_length():
    short = "one two three"
    long = " ".join(["word"] * 100)
    assert estimate_token_count(long) > estimate_token_count(short)


# --- section boundaries / no cross-section mixing --------------------------


def test_no_chunk_mixes_content_from_two_different_sections():
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) > 1  # sanity: fixture actually produced multiple chunks

    for chunk in chunks:
        present = _markers_present(chunk.text)
        assert len(present) == 1, (
            f"chunk ordinal={chunk.ordinal} mixes markers from sections {present}: "
            f"{chunk.text!r}"
        )


def test_chunks_are_ordered_and_ordinals_are_sequential():
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# --- short sections are not split or padded --------------------------------


@pytest.mark.parametrize("heading", ["Overview", "Getting Started", "Appendix"])
def test_short_section_is_a_single_unsplit_chunk(heading: str):
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    marker = _SECTION_MARKERS[heading]
    matching = [c for c in chunks if marker in c.text]

    assert len(matching) == 1, (
        f"expected exactly one chunk for short section {heading!r}, got {len(matching)}"
    )
    # Not force-padded: the chunk's token count reflects its actual (short)
    # content, not some inflated/padded value, and is well under the
    # ~300-500 target band this fixture's short sections don't need to hit.
    assert 0 < matching[0].token_count < 100
    assert matching[0].text.startswith(f"# {heading}" if heading == "Overview" else f"## {heading}")


# --- long sections are sub-split, within a reasonable token band ----------


def test_long_section_is_split_into_multiple_chunks_within_target_band():
    text = _load_multi_section()
    chunks = chunk_markdown(
        text,
        parser_version="p1",
        chunking_version="c1",
        target_min_tokens=300,
        target_max_tokens=500,
    )

    deepdive_chunks = [c for c in chunks if "DEEPDIVE-P" in c.text]
    assert len(deepdive_chunks) > 1, "Deep Dive section should require sub-splitting"

    # Every sub-chunk must carry the section's heading, so a reader of any
    # individual chunk still knows which section it came from.
    for c in deepdive_chunks:
        assert c.text.startswith("## Deep Dive")

    # Each sub-chunk should be within a reasonable band around the
    # 300-500 target: not empty/trivial, and not wildly over target_max
    # (a little overshoot from the prepended heading + overlap paragraph
    # is expected and fine; a chunk several times target_max would not be).
    for c in deepdive_chunks:
        assert 0 < c.token_count <= 500 * 1.5


def test_long_section_chunks_cover_all_paragraphs_without_gaps():
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    deepdive_chunks = [c for c in chunks if "DEEPDIVE-P" in c.text]
    combined = "\n".join(c.text for c in deepdive_chunks)
    for n in range(1, 15):
        assert f"DEEPDIVE-P{n} " in combined, f"paragraph DEEPDIVE-P{n} missing from output"


# --- overlap between adjacent sub-chunks in the same section ---------------


def test_overlap_is_present_between_adjacent_subchunks():
    """Verify the documented overlap strategy concretely: each sub-chunk
    after the first must begin (right after its heading line) with the
    exact last paragraph of the *previous* sub-chunk.
    """
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    deepdive_chunks = [c for c in chunks if "DEEPDIVE-P" in c.text]
    assert len(deepdive_chunks) >= 2, "need at least 2 sub-chunks to prove overlap"

    for prev_chunk, next_chunk in zip(deepdive_chunks, deepdive_chunks[1:]):
        prev_paragraphs = prev_chunk.text.split("\n\n")
        next_paragraphs = next_chunk.text.split("\n\n")

        prev_last_paragraph = prev_paragraphs[-1]
        # next_paragraphs[0] is the heading line ("## Deep Dive"); the
        # overlap paragraph is the one right after it.
        assert next_paragraphs[0] == "## Deep Dive"
        next_overlap_paragraph = next_paragraphs[1]

        assert next_overlap_paragraph == prev_last_paragraph, (
            "expected the previous chunk's last paragraph to be repeated "
            "verbatim as the overlap paragraph at the start of the next chunk"
        )


def test_overlap_does_not_compound_across_many_subchunks():
    """The overlap paragraph pulled into chunk i+1 must come from chunk i's
    own (non-overlap) group content, not from chunk i's already-overlapped
    text -- otherwise overlap would grow with every sub-chunk. Verify the
    overlap paragraph in each chunk appears exactly once in that chunk's
    text (not duplicated), which would not hold if overlap compounded.
    """
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    deepdive_chunks = [c for c in chunks if "DEEPDIVE-P" in c.text]
    for chunk in deepdive_chunks[1:]:
        paragraphs = chunk.text.split("\n\n")
        overlap_paragraph = paragraphs[1]
        assert paragraphs.count(overlap_paragraph) == 1


# --- parser_version / chunking_version validation --------------------------


def test_empty_parser_version_or_chunking_version_raises():
    text = _load_multi_section()
    with pytest.raises(ValueError):
        chunk_markdown(text, parser_version="", chunking_version="c1")
    with pytest.raises(ValueError):
        chunk_markdown(text, parser_version="p1", chunking_version="")


# --- fenced code blocks are atomic (not scanned for headings) -------------


def test_fenced_code_block_is_not_treated_as_heading_and_not_split():
    text = (
        "# Config\n\n"
        "CONFIG-MARKER Here is a sample config block:\n\n"
        "```yaml\n"
        "# this is a comment, not a heading\n"
        "key: value\n"
        "other_key: other_value\n"
        "```\n\n"
        "CONFIG-MARKER-2 Text after the code block.\n"
    )
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) == 1
    assert "```yaml" in chunks[0].text
    assert "# this is a comment, not a heading" in chunks[0].text
    # The whole fenced block must survive intact as one contiguous unit.
    assert (
        "```yaml\n"
        "# this is a comment, not a heading\n"
        "key: value\n"
        "other_key: other_value\n"
        "```"
    ) in chunks[0].text


# --- edge cases: no headings, preamble, adjacent headings, nesting --------


def test_document_with_no_headings_becomes_one_section():
    text = "PREAMBLE-MARKER first paragraph.\n\nPREAMBLE-MARKER second paragraph."
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) == 1
    assert "PREAMBLE-MARKER first paragraph." in chunks[0].text
    assert "PREAMBLE-MARKER second paragraph." in chunks[0].text


def test_preamble_before_first_heading_is_its_own_section():
    text = (
        "PREAMBLE-MARKER intro text before any heading.\n\n"
        "# First Heading\n\n"
        "HEADING-MARKER content under the heading."
    )
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) == 2
    preamble_chunks = [c for c in chunks if "PREAMBLE-MARKER" in c.text]
    heading_chunks = [c for c in chunks if "HEADING-MARKER" in c.text]
    assert len(preamble_chunks) == 1
    assert len(heading_chunks) == 1
    # The preamble chunk has no heading line prepended (there was none).
    assert not preamble_chunks[0].text.startswith("#")
    # No mixing between the two.
    assert "HEADING-MARKER" not in preamble_chunks[0].text
    assert "PREAMBLE-MARKER" not in heading_chunks[0].text


def test_heading_immediately_followed_by_another_heading_produces_bare_chunk():
    text = "# First\n\n# Second\n\nSECOND-MARKER only content is here."
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) == 2
    bare_chunks = [c for c in chunks if c.text.strip() == "# First"]
    assert len(bare_chunks) == 1, "expected one bare-heading chunk with no paragraphs"
    second_chunks = [c for c in chunks if "SECOND-MARKER" in c.text]
    assert len(second_chunks) == 1
    assert second_chunks[0].text.startswith("# Second")


def test_nested_heading_levels_do_not_mix_with_parent_or_sibling():
    text = (
        "# Parent\n\n"
        "PARENT-MARKER parent-level content.\n\n"
        "## Child\n\n"
        "CHILD-MARKER child-level content.\n\n"
        "## Sibling\n\n"
        "SIBLING-MARKER sibling-level content."
    )
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert len(chunks) == 3
    for chunk in chunks:
        markers_present = sum(
            marker in chunk.text
            for marker in ("PARENT-MARKER", "CHILD-MARKER", "SIBLING-MARKER")
        )
        assert markers_present == 1, f"chunk mixes markers: {chunk.text!r}"


def test_single_oversized_paragraph_is_never_split_mid_paragraph():
    # One paragraph alone exceeds target_max_tokens (500) -- must still
    # survive intact wherever it appears (its own group, and/or copied
    # verbatim as the next chunk's overlap paragraph) rather than being
    # split mid-paragraph.
    huge_paragraph = "OVERSIZED-MARKER " + " ".join(["word"] * 600)
    text = f"# Big Section\n\n{huge_paragraph}\n\nSMALL-MARKER a short trailing paragraph."
    chunks = chunk_markdown(
        text, parser_version="p1", chunking_version="c1", target_max_tokens=500
    )

    oversized_chunks = [c for c in chunks if "OVERSIZED-MARKER" in c.text]
    assert len(oversized_chunks) >= 1
    for c in oversized_chunks:
        # The full huge paragraph, verbatim and complete, must be present --
        # if it had been split mid-paragraph, this exact substring wouldn't
        # match (a truncated copy would be missing trailing "word"s).
        assert huge_paragraph in c.text
        assert c.token_count > 500

    # The trailing short paragraph must appear in exactly one chunk: the
    # final one (it is never itself duplicated by the overlap mechanism,
    # since overlap only carries the *previous* group's last paragraph
    # forward, never a future paragraph backward).
    small_chunks = [c for c in chunks if "SMALL-MARKER" in c.text]
    assert len(small_chunks) == 1
    assert small_chunks[0] is chunks[-1]


# --- return shape maps directly onto Chunk model fields --------------------


def test_chunk_record_has_ordinal_text_token_count_fields():
    text = _load_multi_section()
    chunks = chunk_markdown(text, parser_version="p1", chunking_version="c1")

    assert all(isinstance(c, ChunkRecord) for c in chunks)
    for c in chunks:
        assert isinstance(c.ordinal, int)
        assert isinstance(c.text, str)
        assert isinstance(c.token_count, int)
        assert c.token_count == estimate_token_count(c.text)
