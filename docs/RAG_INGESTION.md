# RAG Ingestion & Retrieval

As-built documentation of Phase 7's knowledge ingestion and hybrid retrieval
pipeline. This describes the real, shipped implementation (commits through
`12c88cb`) — several details evolved during implementation away from the
plan doc's original proposal (`docs/superpowers/plans/2026-08-27-phase7-knowledge-retrieval.md`);
this document follows the code, not the plan. See `docs/DECISION_LOG.md`'s
2026-08-27 entries for the reasoning behind the choices called out below,
and `docs/EXPERIMENT_REGISTRY.md`'s `phase7_retrieval_v1` entry for the
recorded baseline numbers.

## Schema

Five tables, added in migration `0010` (extension + tables + HNSW index)
and `0011` (lexical search column), in
`apps/api/src/resolvegrid_api/models/knowledge.py`:

- **`Document`** — one source document (`source_type`: `"public"` |
  `"synthetic_private"`). Carries attribution fields (`url`, `publisher`,
  `retrieved_at`, `license`), lifecycle fields (`status`: `"active"` |
  `"superseded"` | `"stale"` | `"restricted"`, a self-referential
  `supersedes_document_id`), a `checksum`, and `access_scope_tags`
  (Postgres `ARRAY(String)`, not this codebase's usual `_json`-suffixed
  opaque-blob convention — a real array so authz filtering can use
  `&&`/`ANY` overlap operators directly in SQL).
- **`DocumentVersion`** — one parsed/chunked snapshot of a `Document`
  (`parser_version`, `chunking_version`, `content_hash`). `Chunk` rows
  point at the `DocumentVersion` that produced them, not directly at the
  `Document`, so re-ingesting updated content doesn't discard the chunk
  history of the version that produced it.
- **`Chunk`** — one retrieval unit (`ordinal`, `text`, `token_count`), plus
  a nullable self-referential `parent_chunk_id` reserved for future
  parent-child retrieval (not populated by anything in this phase) and a
  `search_vector` column (see "Lexical search" below).
- **`Embedding`** — one vector per `Chunk` (`embedding_model`,
  `embedding_version`, `vector: Vector(768)` via `pgvector.sqlalchemy`).
- **`IngestionRun`** — one ingestion batch's bookkeeping (`parser_version`/
  `chunking_version`/`embedding_version`, `documents_processed`,
  `chunks_created`, `status`: `"running"` | `"completed"` | `"error"`,
  `error_message`). No FK to `Document`/`DocumentVersion` — a run-level
  summary row, matching `ModelCall`'s precedent elsewhere in this codebase.

`EMBEDDING_DIM = 768` matches `nomic-embed-text`'s untruncated default
output dimensionality, empirically confirmed (not just taken from the
model card) against a real local Ollama container — see
`services/retrieval/src/resolvegrid_retrieval/embedder.py`'s
`NOMIC_EMBED_TEXT_DIM` docstring.

## Chunking strategy

`services/retrieval/src/resolvegrid_retrieval/chunker.py`'s
`chunk_markdown()` — a small hand-rolled, heading-preserving Markdown
chunker (not a full CommonMark parser):

1. Parses raw Markdown into ATX headings (`#`..`######`, column 0 only —
   setext-style and indented headings are NOT recognized and fall through
   as ordinary paragraph text, a known, documented scope limit) and
   paragraph blocks; fenced code blocks are consumed atomically and never
   scanned for headings.
2. Groups blocks into **sections**: a heading plus everything up to the
   next heading of any level. Sections are never merged or split across
   each other — no chunk mixes content from two different sections.
3. Per section: if the whole section (heading + paragraphs) is at or
   under `target_max_tokens` (default 500), or the section has no
   paragraphs at all, it becomes exactly one chunk. Otherwise paragraphs
   are greedily grouped up to `target_max_tokens` per group (a single
   oversized paragraph still becomes its own chunk rather than being
   split mid-paragraph). `target_min_tokens` (default 300) is a target
   for this grouping, not a floor enforced on every chunk — short
   sections are never padded out, and a section's final leftover group
   can be smaller than the target.
4. **Overlap** (only between sub-chunks within the same section, never
   across a section boundary): each sub-chunk after the first repeats the
   previous group's last `overlap_paragraphs` paragraph(s) (default 1)
   verbatim at its start, pulled from that group's *original* paragraphs
   (not an already-overlap-augmented copy), so overlap stays a fixed size
   rather than compounding across many sub-chunks.

`Chunk.ordinal` is assigned sequentially over the whole document (section
order, then sub-chunk order within a section) — this is also what the
evaluation harness and golden set key relevance labels on (see below),
since it's stable across re-ingestion of unchanged corpus content, unlike
the autoincrement `Chunk.id`.

Token counting is `resolvegrid_retrieval.tokenizer.estimate_token_count` —
an estimate, not a real model-specific tokenizer call.

## Embedding

`services/retrieval/src/resolvegrid_retrieval/embedder.py`'s
`embed_texts()` calls Ollama's `/api/embed` endpoint (the current
documented batch endpoint — `input` accepts a list, `embeddings` returns
one vector per input in order) **directly, bypassing LiteLLM** — unlike
every chat completion in this codebase, which always goes through the
LiteLLM proxy. Per `docs/DECISION_LOG.md`'s 2026-08-27 entry: `infra/litellm/config.yaml`
defines no embedding route today, `nomic-embed-text` has no cloud-fallback
requirement in this phase, and `services/retrieval` is a zero-runtime-dependency
library that shouldn't gain a dependency on `apps/api`'s LiteLLM
conventions just to reach Ollama. Revisit trigger noted there: a future
cloud embedding provider, or a need for `ModelCall`-style centralized
cost/telemetry on embedding calls.

Requests are batched (`DEFAULT_BATCH_SIZE = 64` texts per POST) to avoid
an unbounded request body on a large ingestion run. `EmbeddingError` is
raised for an unpulled model (Ollama's real 404 shape, confirmed
directly), any other HTTP/network failure, or a malformed/wrong-shape
response (missing `embeddings`, wrong count, or inconsistent vector
dimensions within one batch).

## Hybrid retrieval

`apps/api/src/resolvegrid_api/retrieval.py` — lives in `apps/api` (owns
all DB access), not `services/retrieval` (a pure library with zero
SQLAlchemy dependency).

**Vector search** (`vector_search`): pgvector cosine distance (`<=>`
operator) over `Embedding.vector`, nearest-first, joined through `Chunk` →
`DocumentVersion` → `Document` so the authz filter (below) can apply in
the same query. The Python float list is rendered to pgvector's
`"[0.1,0.2,...]"` string literal form and cast via `(:param)::vector` —
confirmed directly against the live DB that a bound Python list alone is
not accepted for a raw `text()` query (pgvector's adapter registration
only covers `Vector`-typed ORM columns).

**Lexical search** (`lexical_search`): Postgres full-text search over
`Chunk.search_vector` — a **stored, generated** `tsvector` column
(`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`, migration
`0011`), GIN-indexed, rather than computing `to_tsvector` inline per
query. Chosen over the inline form because it keeps `ts_rank_cd` queries
index-backed instead of a sequential recompute per row — the migration's
docstring frames this as "not a throwaway prototype" even though the
corpus is currently small enough that the inline form would also have
been tolerable. Queries are parsed with `websearch_to_tsquery('english',
...)` (tolerates natural free-text input, unlike `to_tsquery`'s operator
syntax). Only chunks that actually match AND are authorized are returned
— there is no zero-rank fallback, so this can return fewer than `limit`
results, including zero.

**RRF fusion** (`fuse_rrf`): standard Reciprocal Rank Fusion (Cormack,
Clarke & Buettcher 2009), `score(chunk) = Σ 1/(k + rank_in_list)` summed
over whichever of the two ranked lists contain the chunk (absent from a
list contributes 0, not a penalty). **`k = 60`** — the constant from the
original RRF paper and the common default in real IR systems (Elasticsearch's
`rank.rrf`, OpenSearch hybrid search); chosen because no corpus-specific
tuning signal existed yet when Task 4 shipped (the evaluation baseline,
Task 9, hadn't run), so a well-documented default was preferred over an
arbitrary guess. Ties broken by `chunk_id` ascending for deterministic
output ordering only, not a ranking signal.

## Authorization-aware filtering

`apps/api/src/resolvegrid_api/retrieval_authz.py`'s `build_authz_filter()`
turns a resolved `Principal` into an `AuthzFilter(unrestricted: bool,
allowed_tags: frozenset[str])`, via the *same* `authorize()` entry point
`routers/tickets.py` already calls (`packages/authz`'s policy module) —
not a parallel authorization mechanism. A new `"knowledge.retrieve"`
action was added to that policy's self-scoped actions for this purpose:

- Global admin → `unrestricted=True` (every chunk visible, tag check
  skipped entirely).
- Department-scoped grant (analyst/approver) → `allowed_tags` is the
  granted departments, normalized via `normalize_department_tag()`
  (lowercase, spaces → underscores, e.g. `"Platform Engineering"` →
  `"platform_engineering"`).
- No matching grant (self-scope) → falls back to the employee's own home
  department, same normalization. An employee with no department, or who
  doesn't resolve, gets an **empty** `allowed_tags` (fail closed — sees
  only public/unscoped documents, never everything).

This is baked directly into `vector_search`/`lexical_search`'s SQL `WHERE`
clause — `authz_filter` is a **required** keyword parameter on both, not
applied post-hoc to a Python list — so an unauthorized chunk never
materializes as a row in the result set at all:

```sql
WHERE (:unrestricted
       OR document.access_scope_tags = ARRAY[]::varchar[]
       OR document.access_scope_tags && (:allowed_tags)::varchar[])
```

Per `docs/DECISION_LOG.md`'s 2026-08-27 entry: a `Document` with an
**empty** `access_scope_tags` is treated as public/unscoped, visible to
every principal — including one whose own `allowed_tags` resolves to
empty (fail-closed for restricted content, but not fail-closed against
genuinely public content). This is verified directly, not just asserted
by code review: a principal with an empty allowed-tags set still sees
untagged/public documents while a tagged document stays excluded. The
real risk this convention creates is upstream in ingestion, not in the
filter itself — see "Untagged-`synthetic_private` safety check" below.

## Deterministic sufficiency check

`assess_sufficiency()` in `retrieval.py` — no model call, per the plan
doc's "deterministic checks first" framing. Two checks, both required for
`sufficient=True`:

1. **Result-count coverage**: `len(fused_results) >= min_results` (default
   `DEFAULT_MIN_RESULTS = 1`).
2. **Min top-k score**: the best fused score `>= min_score_threshold`,
   where `DEFAULT_MIN_SCORE_THRESHOLD = 1.0 / (DEFAULT_RRF_K + 1)` —
   i.e., the score a chunk would get from ranking **#1 in exactly one** of
   the two signals. This is deliberately pinned to a concrete point on
   the RRF scale (RRF scores aren't a normalized 0–1 similarity) rather
   than an arbitrary small number: a true top-1 hit in either signal
   always clears it; a result that only ever placed moderately (rank 5+)
   in both signals scores below it.

The plan doc's other named deterministic check ("required-field
coverage") is **not implemented** at this layer — `fuse_rrf`'s output is
bare `(chunk_id, score)` with no attached metadata to check field-presence
on. Documented as a known gap in `retrieval.py`'s own docstring, not
silently dropped.

## Ingestion pipeline

`apps/api/src/resolvegrid_api/ingestion.py`'s `ingest_document()` is the
single place that wires `chunk_markdown` → `embed_texts` →
`store_chunks_with_embeddings` together, plus `Document`/`DocumentVersion`
bookkeeping.

**Idempotency**: `Document.title` is the natural key (no separate
slug/external-id column). On each call:

1. Look up an existing `Document` by title.
2. If none exists, create a new `Document` + `DocumentVersion`
   (`content_hash = sha256(raw_markdown)`), then chunk/embed/store.
3. If a `Document` with that title exists, **re-validate this call's
   `source_type`/`access_scope_tags` against what's already persisted** —
   a mismatch raises `IngestionError` rather than silently reusing (or
   silently overwriting) the existing classification. Then check for an
   existing `DocumentVersion` with the same `content_hash`: if found, this
   is an idempotent no-op (returns the existing version without
   re-chunking/re-embedding/re-storing).
4. If the title exists but `content_hash` differs, a new
   `DocumentVersion` is created under the existing `Document` and
   re-chunked/re-embedded — deliberate re-ingestion of updated content.
   `Document`-level metadata is **not** updated on this path (only first
   ingestion sets it) — a documented scope limit.

**The title-collision fix** (step 3's re-validation) was added after
review found a real bypass: the initial implementation only validated the
untagged-`synthetic_private` check (below) against the *current call's*
own arguments, which meant re-ingesting a genuinely different,
differently-classified document under an *already-existing* title would
silently keep the OLD document's classification, discarding the new
call's. Fixed with a real regression test. See `docs/DECISION_LOG.md`'s
2026-08-27 entry for the *remaining*, deliberately-deferred gap: no DB-level
unique constraint on `document.title`, so a genuine *concurrent* race
between two first-ingestion calls for the same brand-new title is not
addressed (not exploitable today — ingestion is single-threaded/sequential
in this phase's scope).

**Untagged-`synthetic_private` safety check**: because an empty
`access_scope_tags` means "public" (see above), `ingest_document` refuses
—before any database write — to ingest a `source_type="synthetic_private"`
document with empty `access_scope_tags` (would silently leak a private
document to everyone), and the symmetric case: a `source_type="public"`
document must carry *no* `access_scope_tags` (a non-empty set would
incorrectly restrict something meant to be public). Both raise
`IngestionError`.

### Arq job

`apps/api/src/resolvegrid_api/ingestion_worker.py` — the first real async
job in this repo (Redis was already running via `infra/docker-compose.yml`
but unconsumed before this phase). One core function,
`run_seed_corpus_ingestion(session)`, called from two entry points so the
`IngestionRun` bookkeeping lives in exactly one place:

- **Direct/sync**: `uv run --package resolvegrid-api python -m resolvegrid_api.ingestion_worker`
  (`main()`) — what this task's fresh-state verification used, and what
  `apps/api/tests/test_ingestion_worker.py` calls for most of its coverage.
- **Real Arq job**: `ingest_seed_corpus_task(ctx)`, registered in
  `WorkerSettings.functions`. A real worker process:
  `uv run --package resolvegrid-api arq resolvegrid_api.ingestion_worker.WorkerSettings`.
  It opens its own SQLAlchemy session (Arq jobs run in a separate process
  with no access to FastAPI's request-scoped `get_db`), and offloads the
  blocking DB/HTTP work via `asyncio.to_thread`. One test
  (`test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis`)
  proves this is a genuine wire-format Arq job — not just a same-process
  call — by enqueuing onto the real `resolvegrid-redis` queue and running
  a real `arq.worker.Worker` in burst mode.

Pinned versions for this phase's runs (`ingestion_worker.py` module
constants): `PARSER_VERSION = "markdown-v1"`, `CHUNKING_VERSION =
"heading-aware-v1"`, `EMBEDDING_MODEL = "nomic-embed-text"`,
`EMBEDDING_VERSION = "v1"`. Bumping any of these is a deliberate
re-ingestion-triggering event.

`run_seed_corpus_ingestion` processes `resolvegrid_api.seed_corpus.SEED_CORPUS`
in list order (required, since `supersedes_title` references must resolve
against an already-ingested title), records one `IngestionRun` row per
batch (`status="running"` → `"completed"` with `documents_processed`/
`chunks_created`, or `status="error"` with `error_message` and partial
counts, re-raised after recording), and does not commit itself — callers
control the transaction boundary.

## Wired into the agent graph

`services/agent-orchestration/.../graph.py`'s graph is now
`classify_intent → retrieve → compose_response → finalize`. The
`retrieve` node calls an injected `RetrieveFn` (same dependency-injection
pattern as `CompleteFn` — this package never imports `resolvegrid_api`
directly) and attaches `retrieved_chunks`/`retrieval_sufficient` to state;
any retrieval failure degrades softly to "no chunks, not sufficient"
rather than crashing the graph. `compose_response` branches its prompt on
`retrieval_sufficient AND chunks non-empty`: sufficient → a citation-ready
context block (`[chunk:<id>] (from "<title>"): <text>`) with instructions
to cite inline; otherwise → the original Phase-6 general-knowledge-only
prompt.

**Deliberate scope limit** (documented in `graph.py`'s own module
docstring): this is *not* the plan doc's full "sufficient → answer with
citations, insufficient → abstain/clarify" routing. There is no hard
abstention branch — an insufficient retrieval result still gets a
best-effort general-knowledge answer, since `assess_sufficiency`'s bar
answers "is there company-specific evidence worth citing," not "is this
question answerable at all."

`apps/api/src/resolvegrid_api/routers/chat.py`'s `/chat` endpoint resolves
`build_authz_filter(principal, session)` per-request and passes it into
the graph as a plain `retrieval_scope` dict. The response surfaces
`citations` (chunk id + document title) and a `caption` **only** when
`retrieval_sufficient AND retrieved_chunks` — mirroring `graph.py`'s own
branch condition exactly, so the UI never claims a grounded answer the
model wasn't actually given context for. The old Phase-6 "no ticket or
company-specific knowledge base yet" caption is replaced by
`"General-knowledge answer — no matching company knowledge-base article
was found for this question."` for the ungrounded case, since a real KB
now exists.

## Evaluation

`apps/api/src/resolvegrid_api/eval_retrieval.py` — see
`docs/EXPERIMENT_REGISTRY.md`'s `phase7_retrieval_v1` entry for the
recorded baseline (recall@5=1.0000, precision@5=0.2000, MRR=0.9524,
nDCG@5=0.9643 over 14 answerable golden cases, zero authz leakage, zero
distractor wins). Golden cases key relevance by `(Document.title,
Chunk.ordinal)`, not `Chunk.id` (unstable across re-ingestion), resolved
at eval-run time against whatever corpus is actually loaded.

`eval_retrieval.main()` ingests-then-rolls-back inside one session (zero
DB residue for a manual run against a shared dev DB) — Task 10's
fresh-state verification instead ingested via `ingestion_worker.main()`
(which commits) and then ran `run_eval()` directly against that persisted
session, to confirm the baseline reproduces against the corpus left
ingested in the dev DB (see "Current state of the dev database" below).

## Public-source attribution log

Per `apps/api/src/resolvegrid_api/seed_corpus.py`, exactly as defined
there. **These are honestly synthetic placeholder attributions** — short,
original explainer paragraphs about real, generically-known public
technical concepts, written for this corpus, **not scraped from any real
external page**. `url` uses `example.com` (the domain IANA reserves for
documentation/placeholder use, RFC 2606) specifically so it never
impersonates a real publisher's real domain; `publisher` and `license`
both say plainly, in their own text, that this is a synthetic placeholder,
not a real citation.

| Title | `url` | `publisher` | `retrieved_at` | `license` |
|---|---|---|---|---|
| What Is a VPN (Public Reference) | `https://example.com/knowledge/what-is-a-vpn` | "ResolveGrid Knowledge Base (synthetic placeholder publisher — these excerpts are original text written for this corpus, not scraped from a real external site; see seed_corpus.py's module docstring)" | 2026-08-27 | "Author-written summary of a publicly-known general concept; no external copyrighted source — treat as freely reproducible within this synthetic corpus." |
| What Is Multi-Factor Authentication (Public Reference) | `https://example.com/knowledge/what-is-mfa` | (same placeholder publisher text as above) | 2026-08-27 | (same placeholder license text as above) |
| What Is a Service-Level Agreement (Public Reference) | `https://example.com/knowledge/what-is-an-sla` | (same placeholder publisher text as above) | 2026-08-27 | (same placeholder license text as above) |

All three are `source_type="public"` with empty `access_scope_tags`
(visible to every principal, per the empty-tags-means-public convention
above). The remaining 5 corpus documents are `source_type="synthetic_private"`
Kestrel-internal policies, each tagged to one real seeded department —
including the deliberately-superseded VPN policy pair (`Kestrel VPN
Access Policy (v1, deprecated)`, `status="superseded"`, and its
replacement `(v2)`, `status="active"`, `supersedes_title` pointing back at
v1) used to test that current content outranks stale content in
retrieval. See `apps/api/src/resolvegrid_api/seed_corpus.py` for the full
manifest and `eval/corpus/*.md` for the raw source text.

## Current state of the dev database

As of this task's fresh-state verification (2026-08-27/28), the local dev
database has the seed corpus ingested (8 documents, 36 chunks, 36
embeddings) and 75 employees seeded (`--seed 42`) — left in place
deliberately, not restored to empty. See `docs/PROGRESS.md`'s Phase 7 row
for the reasoning.

**Constraint discovered while closing out this task** (see
`docs/DECISION_LOG.md`'s 2026-08-28 entry): the full `apps/api` test
suite must only ever be run against a database with zero Phase-7
knowledge content — `test_retrieval.py`'s fixed-content search-ranking
tests, and at least one test in `test_chat_api.py`
(`test_chat_success_writes_agent_run_and_four_success_spans`, which
assumes a non-seed-corpus-department requester finds nothing to cite),
assert exact result/citation lists that genuinely fail once real
seed-corpus chunks are present — not an isolated case scoped to one
file. Seeding employees and ingesting the corpus must always be the
last, untested steps of a verification sequence, never followed by
another full-suite run. `test_ingestion_worker.py`'s and
`test_chat_api.py`'s own corpus-*ingesting* tests are safe to run
against an already-populated DB (their cleanup now tracks rows by
id-watermark, not by title, so they correctly delete nothing when the
corpus was already present, verified directly) — the remaining risk is
specifically tests elsewhere that assert exact/empty result sets without
accounting for real corpus content.
