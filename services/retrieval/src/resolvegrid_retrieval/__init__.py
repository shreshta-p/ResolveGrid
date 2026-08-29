"""Structure-aware Markdown parsing/chunking for ResolveGrid's Phase 7
knowledge ingestion pipeline.

Public entry point: `chunk_markdown(text, parser_version, chunking_version)`,
which returns an ordered list of `ChunkRecord` (`ordinal`, `text`,
`token_count`) shaped to map directly onto the `Chunk` model's fields
(`apps/api/src/resolvegrid_api/models/knowledge.py`). This package has no
dependency on `resolvegrid_api`, SQLAlchemy, or a database -- persistence
is wired in by a later ingestion task.
"""

from resolvegrid_retrieval.chunker import ChunkRecord, chunk_markdown
from resolvegrid_retrieval.status_adjustment import apply_status_adjustment
from resolvegrid_retrieval.tokenizer import estimate_token_count

__all__ = [
    "ChunkRecord",
    "apply_status_adjustment",
    "chunk_markdown",
    "estimate_token_count",
]
