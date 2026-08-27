"""Structure-aware (heading-preserving) Markdown chunker.

Design summary (Phase 7 Task 2 -- see
`docs/superpowers/plans/2026-08-27-phase7-knowledge-retrieval.md`):

1.  Parse the raw Markdown into an ordered list of *blocks*: ATX headings
    (`#` .. `######`), and "paragraphs" (any other maximal run of
    non-blank lines). Fenced code blocks (``` or ~~~) are parsed as a
    single atomic paragraph block each -- their contents are never
    scanned for heading syntax and are never split internally.

    This is a deliberately small hand-rolled parser (not
    `markdown-it-py`/the stdlib `markdown` package): the only structural
    facts this chunker needs are "where do headings fall" and "where are
    the paragraph boundaries," which a ~100-line regex-driven line
    scanner gets right without pulling in a full CommonMark AST. See
    `pyproject.toml` for the fuller dependency-footprint tradeoff note.

2.  Group blocks into *sections*: a section is a heading (or `None` for
    any preamble content before the first heading) plus every paragraph
    block that follows it, up to the next heading of any level. Sections
    are **never merged** with each other, in either direction -- a short
    section is never padded out by borrowing a neighboring section's
    content, and a long section is never truncated into a neighbor. This
    is what makes "no chunk mixes content from two different sections"
    structurally true rather than something checked after the fact.

3.  Per section, decide chunk count:
      - If the section's full text (heading line + its paragraphs,
        joined by blank lines) is within `target_max_tokens` (default
        500), or the section has no paragraphs at all (a bare heading),
        it becomes exactly **one** chunk. Short sections are emitted as
        one small chunk and are never force-padded up to
        `target_min_tokens` -- `target_min_tokens` is a *target* for the
        upper sub-splitting path below, not a floor enforced on every
        chunk.
      - Otherwise the section's paragraphs are greedily grouped: keep
        appending paragraphs to the current group until the next
        paragraph would push the group over `target_max_tokens`, then
        start a new group. A single paragraph that alone exceeds
        `target_max_tokens` is still emitted as its own (oversized)
        chunk -- paragraph boundaries are the only split points this
        chunker uses; it never splits inside a paragraph. Because of
        this, and because a section's final leftover group can be
        smaller than `target_min_tokens`, "~300-500 tokens" is a target
        band, not a guaranteed range for every chunk.

4.  Overlap (only between sub-chunks *within the same section*, never
    across a section boundary): every sub-chunk after the first is
    built as `[heading line if any] + [the last `overlap_paragraphs`
    paragraph(s) of the *previous* group, taken from that group's
    original paragraphs, not from any overlap already prepended to it]
    + [this group's own paragraphs]`. Concretely, with the default
    `overlap_paragraphs=1`, each sub-chunk after the first repeats the
    single last paragraph of the previous sub-chunk's own content
    verbatim at its start (after the heading line). Pulling the overlap
    from the *original* group (rather than the previous chunk's already
    overlap-augmented text) keeps the overlap a fixed size instead of
    compounding across many sub-chunks. This is a deliberately simple,
    exactly-verifiable overlap strategy: a later reader (or test) can
    split any two adjacent same-section chunks on blank lines and find
    the earlier chunk's last paragraph as a literal prefix (after the
    heading line, if present) of the later chunk's paragraph list.

Ordinals are assigned sequentially over the whole document in section
order, then sub-chunk order within a section -- exactly the ordering a
later ingestion task should persist to `Chunk.ordinal`
(`apps/api/src/resolvegrid_api/models/knowledge.py`).

Known scope limits (fail-soft, not crashes): only ATX headings (`# Title`)
starting at column 0 are recognized -- an indented ATX heading or a
setext-style heading (`Title` underlined with `===`/`---`) is treated as
ordinary paragraph text instead of a heading. Real external vendor docs
ingested in a later task may use either style; if so their headings would
silently merge into surrounding paragraph content rather than split
correctly. Worth a follow-up if that's observed in practice.
"""

import re
from dataclasses import dataclass

from resolvegrid_retrieval.tokenizer import estimate_token_count

DEFAULT_TARGET_MIN_TOKENS = 300
DEFAULT_TARGET_MAX_TOKENS = 500
DEFAULT_OVERLAP_PARAGRAPHS = 1

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_PREFIXES = ("```", "~~~")


@dataclass(frozen=True)
class ChunkRecord:
    """One retrieval chunk, shaped to map 1:1 onto `Chunk` model fields
    (`ordinal`, `text`, `token_count`) -- `document_version_id` is wired
    in by a later ingestion task, not here.
    """

    ordinal: int
    text: str
    token_count: int


@dataclass
class _Section:
    # (level, title, raw_heading_line) or None for content before any heading.
    heading: tuple[int, str, str] | None
    paragraphs: list[str]


def _parse_blocks(text: str) -> list[tuple[str, object]]:
    """Parse `text` into an ordered list of `("heading", (level, title, line))`
    and `("para", block_text)` tuples. See module docstring for the fenced
    code block and blank-line-separation rules.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, object]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            block_text = "\n".join(current).strip("\n")
            if block_text.strip():
                blocks.append(("para", block_text))
            current.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith(_FENCE_PREFIXES):
            # Fenced code block: flush whatever came before, then consume
            # the whole fence (open + body + close, if a close exists)
            # as one atomic paragraph block, never scanned for headings.
            flush()
            marker = stripped[:3]
            code_lines = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(marker):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                code_lines.append(lines[i])  # closing fence
                i += 1
            blocks.append(("para", "\n".join(code_lines)))
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            blocks.append(("heading", (level, title, line.strip())))
            i += 1
            continue

        if stripped == "":
            flush()
            i += 1
            continue

        current.append(line)
        i += 1

    flush()
    return blocks


def _group_sections(blocks: list[tuple[str, object]]) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(heading=None, paragraphs=[])

    for block_type, payload in blocks:
        if block_type == "heading":
            if current.heading is not None or current.paragraphs:
                sections.append(current)
            current = _Section(heading=payload, paragraphs=[])  # type: ignore[arg-type]
        else:
            current.paragraphs.append(payload)  # type: ignore[arg-type]

    if current.heading is not None or current.paragraphs:
        sections.append(current)

    return sections


def _split_section_into_groups(
    paragraphs: list[str], target_max_tokens: int
) -> list[list[str]]:
    """Greedily group `paragraphs` so each group's paragraph-only token
    total stays at or under `target_max_tokens`, splitting only at
    paragraph boundaries. A single paragraph larger than
    `target_max_tokens` becomes its own (oversized) group rather than
    being split mid-paragraph.
    """
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_token_count(para)
        if current_group and current_tokens + para_tokens > target_max_tokens:
            groups.append(current_group)
            current_group = []
            current_tokens = 0
        current_group.append(para)
        current_tokens += para_tokens

    if current_group:
        groups.append(current_group)

    return groups


def chunk_markdown(
    text: str,
    parser_version: str,
    chunking_version: str,
    *,
    target_min_tokens: int = DEFAULT_TARGET_MIN_TOKENS,
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[ChunkRecord]:
    """Chunk raw Markdown `text` into an ordered list of `ChunkRecord`.

    `parser_version` and `chunking_version` are required, non-empty
    strings identifying the parsing/chunking logic that produced this
    result. They are validated here but deliberately not embedded in the
    returned `ChunkRecord`s: that provenance belongs on the
    `DocumentVersion` row a later ingestion task creates
    (`DocumentVersion.parser_version`/`chunking_version` in
    `apps/api/src/resolvegrid_api/models/knowledge.py`), not duplicated
    onto every `Chunk`. Accepting them here (rather than adding them
    later) keeps this function's signature stable if a future
    `chunking_version` bump ever needs to branch on the version string
    internally.

    See the module docstring for the full section-grouping, sub-splitting,
    and overlap strategy.
    """
    if not parser_version:
        raise ValueError("parser_version must be a non-empty string")
    if not chunking_version:
        raise ValueError("chunking_version must be a non-empty string")
    if target_max_tokens <= 0:
        raise ValueError("target_max_tokens must be positive")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs must not be negative")

    sections = _group_sections(_parse_blocks(text))

    chunks: list[ChunkRecord] = []
    ordinal = 0

    for section in sections:
        heading_line = section.heading[2] if section.heading else None
        full_parts = ([heading_line] if heading_line else []) + section.paragraphs
        full_text = "\n\n".join(full_parts)

        if not full_text.strip():
            continue

        full_tokens = estimate_token_count(full_text)

        if not section.paragraphs or full_tokens <= target_max_tokens:
            chunks.append(ChunkRecord(ordinal=ordinal, text=full_text, token_count=full_tokens))
            ordinal += 1
            continue

        groups = _split_section_into_groups(section.paragraphs, target_max_tokens)

        for i, group in enumerate(groups):
            parts: list[str] = []
            if heading_line:
                parts.append(heading_line)
            if i > 0 and overlap_paragraphs > 0:
                previous_group = groups[i - 1]
                parts.extend(previous_group[-overlap_paragraphs:])
            parts.extend(group)

            chunk_text = "\n\n".join(parts)
            chunks.append(
                ChunkRecord(
                    ordinal=ordinal,
                    text=chunk_text,
                    token_count=estimate_token_count(chunk_text),
                )
            )
            ordinal += 1

    return chunks
