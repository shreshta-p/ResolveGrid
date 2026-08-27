# Phase 7 — Knowledge ingestion + baseline retrieval

Entry dependency: Phase 6 (Verified, `a8d38b7`). Master plan reference: `plan.md` §14 Phase 7 (stages 4-5-6-7 of the 13-stage agent workflow).

## Objective

Give the chat agent a real knowledge base instead of pure model recall: structure-aware ingestion of a bounded synthetic+attributed-public corpus, hybrid (vector + lexical) retrieval with authorization-aware filtering applied *before* the query runs, and a deterministic sufficiency check — wired into the existing LangGraph graph between `classify_intent` and `compose_response`.

## Grounding research (verify before coding, don't trust memory)

- `pgvector/pgvector:pg16` image is already the Postgres container (`infra/docker-compose.yml`) — extension just needs `CREATE EXTENSION IF NOT EXISTS vector` in a migration, not a new container.
- Redis is already in compose but nothing consumes it yet — this phase is the first real async job (ingestion). Confirm current `arq` (or equivalent) isn't already a dependency anywhere before assuming greenfield.
- Embeddings: `nomic-embed-text` via Ollama's `/api/embed` (or LiteLLM's embeddings passthrough — check which `infra/litellm/config.yaml` already exposes, if anything, before adding a duplicate path). Confirm actual installed pgvector Python binding (`pgvector-sqlalchemy` or raw `Vector` type) via `uv run python -c "import pgvector"` rather than assuming a package name.
- SQLAlchemy model conventions: read `apps/api/src/resolvegrid_api/models/agent.py` and `base.py` for the established mapped-column/FK style before adding `Document`/`Chunk`/etc.
- `services/agent-orchestration/src/resolvegrid_agent_orchestration/graph.py` is the current graph (`classify_intent → compose_response → finalize`) — the retrieve step slots in as a new node/subgraph between the first two, using the same `CompleteFn`-injection pattern already established (don't have the retrieval node import `resolvegrid_api` directly).

## Tasks

1. **Schema + migration** — `Document`, `DocumentVersion`, `Chunk`, `Embedding`, `IngestionRun` models (fields per plan.md §2 domain model) in a new `apps/api/src/resolvegrid_api/models/knowledge.py`; migration `0010` enabling the `vector` extension, creating the tables, and an HNSW index on `Embedding.vector`. Verify migration applies cleanly on a fresh DB and `\d chunk`/`\d embedding` show expected columns/index.

2. **`services/retrieval` package scaffold + parser/chunker** — new uv workspace member. Structure-aware (heading-preserving, not fixed-width) Markdown chunker targeting ~300-500 tokens with overlap, tagging `parserVersion`/`chunkingVersion`. Unit tests on hand-written fixture docs (heading boundaries respected, overlap correct, token counts within target band).

3. **Embeddings + pgvector storage** — embed chunks via `nomic-embed-text` (Ollama), store in `Embedding` with `embeddingModel`/`embeddingVersion`. Verify a stored vector round-trips (cosine similarity between two known-similar synthetic sentences ranks above two known-dissimilar ones).

4. **Lexical search + RRF fusion** — Postgres `tsvector`/`ts_rank_cd` search function, RRF fusion combining vector-rank and lexical-rank lists with a documented fusion constant. Unit test the fusion math directly against a hand-constructed pair of ranked lists (known expected output).

5. **Authz-aware metadata filtering + sufficiency check** — `build_authz_filters` (calls `packages/authz`, turns principal into concrete allowed-scope predicate) as a *required* parameter baked into the SQL query itself, not applied post-hoc; `assess_sufficiency` node (deterministic: min top-k score, required-field coverage). Adversarial test: a principal without a document's access scope must get zero matching chunks back, enforced at the query level (verify by inspecting the actual SQL/query plan, not just the returned list).

6. **Ingestion pipeline (Arq job) + seed corpus** — Redis-backed Arq worker (first real async job in this repo — confirm Arq isn't already partially wired before scaffolding). Seed a bounded corpus: a handful of synthetic Kestrel-internal policies (deliberately including at least one stale/superseded doc) + a small number of attributed public vendor docs (URL, publisher, retrieved date, checksum, license note). `IngestionRun` records parser/chunk/embed versions + stats. Verify idempotent re-run (same source content, same checksum → no duplicate chunks).

7. **Wire retrieval into the agent graph** — new `retrieve` subgraph node(s) in `services/agent-orchestration`, injected the same way `complete_fn` is (no direct `resolvegrid_api` import), invoked between `classify_intent` and `compose_response`. `/chat` endpoint and web page updated: citations rendered, and the old "no ticket or company-specific knowledge base yet" caption removed/updated now that a real KB exists.

8. **Tests** — parser/chunker unit tests (task 2 covers these but confirm coverage), ingestion idempotency test, the adversarial authz-filter leakage test (task 5), RRF fusion unit test (task 4). All wired into the `agent-orchestration`/new `retrieval` CI jobs.

9. **Evaluation baseline** — small retrieval-focused golden subset (queries with hand-labeled relevant chunk IDs) measuring recall/precision/MRR/nDCG against the ingested corpus; record baseline numbers in `docs/EXPERIMENT_REGISTRY.md` with exact versions (parser/chunk/embed/corpus).

10. **Docs + fresh-state verification** — `docs/RAG_INGESTION.md`, public-source attribution log, `docs/PROGRESS.md` Phase 7 row, `scripts/smoke_test.sh` extended if ingestion needs a bootstrap step. Full fresh-state teardown/rebuild verification (matching Phases 4-6's precedent), real browser walkthrough of a cited chat answer, CI green, push.

## Exit criteria (from plan.md §14)

Zero-leakage adversarial retrieval test passes across N trials; baseline recall/precision/MRR recorded with exact versions in `EXPERIMENT_REGISTRY.md`; UI citations verifiably map to real ingested chunks.
