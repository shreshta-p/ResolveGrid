"""Persistence glue: writes chunker output + embedding vectors into the
`Chunk`/`Embedding` tables (Phase 7 Task 3).

Dependency-direction note (mirrors
`services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py`'s
module docstring, which explains the same problem for `complete_fn`):
`services/retrieval` is a pure workspace *library* -- `chunker.py` has zero
runtime dependencies and `embedder.py` only talks to Ollama over HTTP; it
does not import SQLAlchemy, `resolvegrid_api`, or anything DB-shaped (see
its `pyproject.toml`). The `Chunk`/`Embedding` SQLAlchemy models and the
`Session` they're persisted through both live in this *app*
(`resolvegrid_api.models.knowledge`, `resolvegrid_api.db`), so the glue
that turns a `resolvegrid_retrieval.chunker.ChunkRecord` + an embedding
vector into real rows lives here too, not in `services/retrieval`. This
keeps the workspace dependency arrow pointing one direction: `apps/api`
depends on `resolvegrid-retrieval` (added to this app's `pyproject.toml`
for this task), never the reverse -- exactly the same shape as
`apps/api` -> `resolvegrid-agent-orchestration` already established in
Phase 6.
"""

from sqlalchemy.orm import Session

from resolvegrid_api.models.knowledge import Chunk, Embedding
from resolvegrid_retrieval.chunker import ChunkRecord


def store_chunks_with_embeddings(
    session: Session,
    *,
    document_version_id: int,
    chunk_records: list[ChunkRecord],
    vectors: list[list[float]],
    embedding_model: str,
    embedding_version: str,
) -> list[tuple[Chunk, Embedding]]:
    """Write one `Chunk` row + one linked `Embedding` row per
    `(chunk_record, vector)` pair, all pointing at `document_version_id`.

    `chunk_records` and `vectors` must be the same length and
    index-aligned -- i.e. `vectors[i]` is expected to be
    `resolvegrid_retrieval.embedder.embed_texts([c.text for c in chunk_records])[i]`.
    This function deliberately does not call the embedder itself, so a
    caller controls batching/retries for the (network-calling) embed step
    independently of this (pure-DB) persistence step. Raises `ValueError`
    if the lengths don't match, before writing anything.

    Does not commit -- callers control the transaction boundary, matching
    this codebase's `db_session` (rollback-wrapped)/`raw_db_session`
    (real-commit) fixture split in `apps/api/tests/conftest.py`.

    Returns the created `(Chunk, Embedding)` pairs in input order, with
    both rows' primary keys populated (this flushes internally twice: once
    to obtain `Chunk.id` values before building the FK-dependent
    `Embedding` rows, once more so the `Embedding` rows' own ids are
    populated for the caller).
    """
    if len(chunk_records) != len(vectors):
        raise ValueError(
            f"chunk_records ({len(chunk_records)}) and vectors ({len(vectors)}) "
            "must be the same length and index-aligned"
        )

    chunks = [
        Chunk(
            document_version_id=document_version_id,
            ordinal=record.ordinal,
            text=record.text,
            token_count=record.token_count,
        )
        for record in chunk_records
    ]
    session.add_all(chunks)
    session.flush()  # populate chunk.id before building Embedding rows below

    embeddings = [
        Embedding(
            chunk_id=chunk.id,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            vector=vector,
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    session.add_all(embeddings)
    session.flush()

    return list(zip(chunks, embeddings))
