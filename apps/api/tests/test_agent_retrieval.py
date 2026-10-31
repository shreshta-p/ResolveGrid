"""Phase 14 Task 2 code-review follow-up: tests for `resolvegrid_api.
agent_retrieval`'s two new real config toggles (`RESOLVEGRID_RERANK_
ENABLED`, `RESOLVEGRID_RETRIEVAL_CACHE_ENABLED`), added for the Task 2
baseline load-test condition but previously shipped with no automated
test coverage at all.

Both toggles are module-level constants read ONCE from `os.environ` at
import time (see `agent_retrieval.py`'s own module docstring: "a baseline
run is started as a *separate* `uvicorn` process with these set to
'false', never flipped mid-process"). Two things are tested for each,
per that real design:

1. `test_*_env_var_parsing_*`: the real `os.environ.get(...).lower() not
   in (...)` expression itself, for the exact env var NAME this module
   reads, via `importlib.reload` -- proves the parsing (default "true" /
   explicit "0"/"false"/"no") is wired to the right variable, not just a
   helper tested in isolation.
2. `test_retrieve_for_agent_*`: with the already-computed module boolean
   monkeypatched to each value (exactly what that boolean models -- a
   fixed value for this process's lifetime), calls the REAL `retrieve_
   for_agent` function (not a unit around the toggle alone) and proves
   the toggle actually changes its real, observable behavior: whether
   `rerank()` is actually invoked, and whether the retrieval pipeline
   actually recomputes on a second identical call instead of serving a
   cached result.

Real Postgres (via `raw_db_session`, needed because `retrieve_for_agent`
opens its OWN session via `session_factory()` against the real pooled
engine -- a separate connection from any rolled-back `db_session`
transaction, so fixture rows must be real, committed rows) and real
Redis throughout, matching `test_retrieval_cache.py`'s established
"nothing mocked away" philosophy for this cache layer. `vector_search` is
short-circuited via a mocked `EmbeddingError` (this test is about the
toggles, not about exercising a real Ollama embedding call) so only
`lexical_search` needs to be patched to return a controlled candidate
list pointing at a real seeded chunk.
"""

import importlib

import pytest
import redis
from sqlalchemy import delete

from resolvegrid_api import agent_retrieval
from resolvegrid_api.cache import REDIS_URL
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion
from resolvegrid_retrieval.embedder import EmbeddingError

_UNRESTRICTED_SCOPE = {"unrestricted": True, "allowed_tags": []}


def _clear_retrieval_cache_namespace(client: redis.Redis) -> None:
    for key in client.scan_iter(match="cache:retrieval:*"):
        client.delete(key)


@pytest.fixture(autouse=True)
def _clean_retrieval_cache():
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()
    _clear_retrieval_cache_namespace(client)
    yield
    _clear_retrieval_cache_namespace(client)


@pytest.fixture
def one_real_public_chunk(raw_db_session):
    """One real, committed `Chunk` (public -- no `access_scope_tags`), so
    `retrieve_for_agent`'s real DB round trip (resolving chunk text/
    document title/status for the fused candidate ids) has a real row to
    find. Public/unrestricted scope keeps this test about the toggles,
    not about authz filtering.
    """
    document = Document(source_type="synthetic_private", title="Toggle Test Doc", checksum="toggle-test-checksum")
    raw_db_session.add(document)
    raw_db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        parser_version="test-parser-v1",
        chunking_version="test-chunk-v1",
        content_hash="toggle-test-content-hash",
    )
    raw_db_session.add(version)
    raw_db_session.flush()
    chunk = Chunk(
        document_version_id=version.id,
        ordinal=0,
        text="Toggle test chunk content about VPN access.",
        token_count=7,
    )
    raw_db_session.add(chunk)
    raw_db_session.flush()
    raw_db_session.commit()
    chunk_id = chunk.id
    try:
        yield chunk_id
    finally:
        raw_db_session.execute(delete(Chunk).where(Chunk.id == chunk_id))
        raw_db_session.execute(delete(DocumentVersion).where(DocumentVersion.id == version.id))
        raw_db_session.execute(delete(Document).where(Document.id == document.id))
        raw_db_session.commit()


def _patch_pipeline_inputs(monkeypatch, chunk_id: int, rerank_spy_calls: list):
    """Short-circuits the query embedding (so `vector_search` is never
    reached -- this test has no need to exercise a real Ollama embedding
    call) and makes `lexical_search` return exactly the one real seeded
    chunk above, deterministically. `rerank` is replaced with a spy that
    records whether it was called and returns a trivially valid reranked
    list, so `_RERANK_ENABLED=True` tests can prove it WAS called without
    depending on the real (optional) reranker dependency being installed.
    """

    def _raise_embedding_error(*_args, **_kwargs):
        raise EmbeddingError("no embedding service for this test")

    monkeypatch.setattr(agent_retrieval, "get_or_compute_embeddings", _raise_embedding_error)
    monkeypatch.setattr(
        agent_retrieval, "lexical_search", lambda *_args, **_kwargs: [(chunk_id, 1)]
    )

    def _spy_rerank(query_text, candidates):
        rerank_spy_calls.append((query_text, candidates))
        # Real `rerank()` contract (see `_rerank_or_degrade`): a list of
        # `(chunk_id, score)` 2-tuples -- the caller re-attaches text
        # itself. A trivially valid descending pseudo-score per candidate.
        return [(cid, float(len(candidates) - i)) for i, (cid, _text) in enumerate(candidates)]

    monkeypatch.setattr(agent_retrieval, "rerank", _spy_rerank)


# ---------------------------------------------------------------------------
# Env-var parsing (both toggles, both modules read the same convention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var_name, module_attr",
    [
        ("RESOLVEGRID_RERANK_ENABLED", "_RERANK_ENABLED"),
        ("RESOLVEGRID_RETRIEVAL_CACHE_ENABLED", "_RETRIEVAL_CACHE_ENABLED"),
    ],
)
@pytest.mark.parametrize("false_value", ["0", "false", "no", "FALSE", "No"])
def test_toggle_env_var_parses_false_like_values_to_disabled(
    monkeypatch, env_var_name, module_attr, false_value
):
    monkeypatch.setenv(env_var_name, false_value)
    try:
        importlib.reload(agent_retrieval)
        assert getattr(agent_retrieval, module_attr) is False
    finally:
        monkeypatch.delenv(env_var_name, raising=False)
        importlib.reload(agent_retrieval)  # restore the real default for later tests


@pytest.mark.parametrize(
    "env_var_name, module_attr",
    [
        ("RESOLVEGRID_RERANK_ENABLED", "_RERANK_ENABLED"),
        ("RESOLVEGRID_RETRIEVAL_CACHE_ENABLED", "_RETRIEVAL_CACHE_ENABLED"),
    ],
)
def test_toggle_env_var_default_unset_is_enabled(monkeypatch, env_var_name, module_attr):
    """The real default (env var genuinely unset, e.g. a normal dev server
    or the test suite itself) must preserve the exact prior behavior:
    enabled.
    """
    monkeypatch.delenv(env_var_name, raising=False)
    importlib.reload(agent_retrieval)
    assert getattr(agent_retrieval, module_attr) is True


# ---------------------------------------------------------------------------
# RESOLVEGRID_RERANK_ENABLED -- real behavioral proof via retrieve_for_agent
# ---------------------------------------------------------------------------


def test_rerank_default_enabled_calls_real_rerank_function(
    monkeypatch, one_real_public_chunk
):
    rerank_spy_calls: list = []
    _patch_pipeline_inputs(monkeypatch, one_real_public_chunk, rerank_spy_calls)
    monkeypatch.setattr(agent_retrieval, "_RERANK_ENABLED", True)

    result = agent_retrieval.retrieve_for_agent("what is a VPN?", _UNRESTRICTED_SCOPE)

    assert len(rerank_spy_calls) == 1
    assert result["chunks"] and result["chunks"][0]["chunk_id"] == one_real_public_chunk


def test_rerank_disabled_toggle_skips_real_rerank_and_degrades_to_fused_order(
    monkeypatch, one_real_public_chunk
):
    """`RESOLVEGRID_RERANK_ENABLED=false`'s real contract (`_rerank_or_
    degrade`'s docstring): `rerank()` must never be called at all, and the
    pipeline must still produce a real result (the fused order, not an
    empty/broken one) -- proving the toggle takes the disabled CODE PATH,
    not just that some boolean somewhere flipped.
    """
    rerank_spy_calls: list = []
    _patch_pipeline_inputs(monkeypatch, one_real_public_chunk, rerank_spy_calls)
    monkeypatch.setattr(agent_retrieval, "_RERANK_ENABLED", False)

    result = agent_retrieval.retrieve_for_agent("what is a VPN?", _UNRESTRICTED_SCOPE)

    assert rerank_spy_calls == []  # the real rerank() was never reached
    assert result["chunks"] and result["chunks"][0]["chunk_id"] == one_real_public_chunk


# ---------------------------------------------------------------------------
# RESOLVEGRID_RETRIEVAL_CACHE_ENABLED -- real behavioral proof
# ---------------------------------------------------------------------------


def test_retrieval_cache_default_enabled_second_identical_call_is_a_real_cache_hit(
    monkeypatch, one_real_public_chunk
):
    rerank_spy_calls: list = []
    _patch_pipeline_inputs(monkeypatch, one_real_public_chunk, rerank_spy_calls)
    monkeypatch.setattr(agent_retrieval, "_RETRIEVAL_CACHE_ENABLED", True)

    agent_retrieval.retrieve_for_agent("cache toggle default-enabled query", _UNRESTRICTED_SCOPE)
    agent_retrieval.retrieve_for_agent("cache toggle default-enabled query", _UNRESTRICTED_SCOPE)

    # lexical_search (the real pipeline's own work) ran only on the FIRST
    # call -- the second was a real cache hit, never recomputed.
    assert len(rerank_spy_calls) == 1


def test_retrieval_cache_disabled_toggle_recomputes_on_every_call(
    monkeypatch, one_real_public_chunk
):
    """`RESOLVEGRID_RETRIEVAL_CACHE_ENABLED=false`'s real contract
    (`retrieve_for_agent`'s own docstring): bypasses `get_or_compute_
    retrieval` entirely, so every call genuinely re-runs the full
    pipeline -- proven here by the real rerank spy firing TWICE for two
    identical calls, never served from Redis.
    """
    rerank_spy_calls: list = []
    _patch_pipeline_inputs(monkeypatch, one_real_public_chunk, rerank_spy_calls)
    monkeypatch.setattr(agent_retrieval, "_RETRIEVAL_CACHE_ENABLED", False)

    agent_retrieval.retrieve_for_agent("cache toggle disabled query", _UNRESTRICTED_SCOPE)
    agent_retrieval.retrieve_for_agent("cache toggle disabled query", _UNRESTRICTED_SCOPE)

    assert len(rerank_spy_calls) == 2  # the real pipeline ran twice, no cache hit
