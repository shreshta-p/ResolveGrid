# RAG Ingestion & Retrieval

As-built documentation of Phase 7's knowledge ingestion/hybrid retrieval
pipeline and Phase 8's reranking, dedup, context-budgeting, and citation-
verification additions on top of it. This describes the real, shipped
implementation (commits through `66372fa`) — several details evolved
during implementation away from each phase's plan doc's original proposal
(`docs/superpowers/plans/2026-08-27-phase7-knowledge-retrieval.md`,
`docs/superpowers/plans/2026-08-28-phase8-reranking-citations.md`); this
document follows the code, not the plans. See `docs/DECISION_LOG.md`'s
2026-08-27 and 2026-08-28/29 entries for the reasoning behind the choices
called out below, and `docs/EXPERIMENT_REGISTRY.md`'s `phase7_retrieval_v1`,
`phase8_reranking_v1`, and `phase8_reranking_v2` entries for the recorded
before/after numbers.

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

## Reranking (Phase 8 Task 1 / Task 7)

`services/retrieval/src/resolvegrid_retrieval/reranker.py`'s `rerank()` —
a local cross-encoder reranking pass applied to `fuse_rrf`'s full fused
candidate list (not pre-truncated to a small top-N before reranking).

**Model choice**: `BAAI/bge-reranker-base` (`DEFAULT_RERANKER_MODEL`), via
`sentence-transformers`' `CrossEncoder`. This is a real, measured decision,
not the master plan's original first choice — the plan named
`bge-reranker-v2-m3` with `bge-reranker-base` as a fallback "if the
footprint is too heavy." Both were measured for real, CPU-only, in this
environment (no CUDA assumed; confirmed `torch.cuda.is_available() ==
False` on the resolved `torch==2.13.0+cpu` wheel):

| Model | Params | Download | First load | Inference, 12 candidates, warm |
|---|---|---|---|---|
| `bge-reranker-v2-m3` | 568M | 2.2GB | ~134s | ~4.08s total (~340ms/candidate) |
| `bge-reranker-base` | 278M | 1.1GB | ~66s | ~0.94s total (~78ms/candidate) |

`v2-m3`'s ~4s warm latency for a realistic 10-12-candidate batch doesn't
fit a `/chat` request's budget when reranking is one step among retrieval
+ LLM generation, not the whole budget; `bge-reranker-base` at under 1s
is practical. `v2-m3`'s multilingual strength (its main advantage per
BAAI's published benchmarks) buys nothing on this English-only internal
IT-ops corpus. `model` is a keyword argument on `rerank()` so a future
task can override it if GPU inference becomes available.

**Library**: `sentence-transformers` over BAAI's own `FlagEmbedding` —
both require `torch`+`transformers`, but `FlagEmbedding` additionally
pulls in training-oriented dependencies (`datasets`, `accelerate`) this
module has no use for (inference-only).

**Dependency isolation**: `sentence-transformers` (and the
`torch`/`transformers`/`scipy`/`numpy` stack it pulls in — measured at
~200MB across 28 packages) is declared as an **optional extra**
(`resolvegrid-retrieval[reranker]`) on `services/retrieval`'s
`pyproject.toml`, not a core dependency of that package or of
`apps/api`. This repo's `uv` workspace resolves one shared venv across
all `tool.uv.workspace` members, so `uv sync --all-packages --extra
reranker` (run once, at the workspace root) makes `resolvegrid_retrieval.
reranker` importable from `apps/api`'s own code with **no**
`apps/api/pyproject.toml` change at all. See `docs/DECISION_LOG.md` for
the CI leak this created and how it was found and fixed (a plain `uv
sync --all-packages`, with no `--extra reranker`, silently installs
`sentence-transformers` into the shared venv from CI's dev-dependency
group unless that group is kept clean — see the `pyproject.toml`
comment above `[dependency-groups] dev` for the exact mechanism).

**Output**: `rerank()` returns `(chunk_id, rerank_score)` pairs,
best-first, where `rerank_score` is the cross-encoder's sigmoid-activated
relevance probability in `[0, 1]` — not a fused/normalized score
comparable to `fuse_rrf`'s RRF scores. Model loading is cached
process-wide (`_MODEL_CACHE`) so the ~1-2s weight-load-from-disk cost is
only paid once per process. A `RerankError` (missing dependency, model
load failure, inference failure) degrades to the unreranked fused order
rather than zeroing out perfectly good retrieval results — see
`apps/api/src/resolvegrid_api/agent_retrieval.py`'s `_rerank_or_degrade`.

## Near-duplicate dedup (Phase 8 Task 2)

`services/retrieval/src/resolvegrid_retrieval/dedup.py`'s `dedup()` —
applied to the reranked (and, in the live pipeline, status-adjusted —
see below) candidate list before context assembly.

**Method**: word-level (unigram) Jaccard similarity over each candidate's
normalized, stopword-filtered token set — chosen over embedding cosine
similarity (rejected: would require either a DB dependency this
zero-SQL-dependency package deliberately doesn't have, or a fresh Ollama
round-trip per `dedup()` call) and over bigram/trigram shingles (measured
and rejected: too sensitive to word order/phrasing — a genuine
same-fact-reworded pair scored under 0.07, indistinguishable from
unrelated content).

**Threshold**: `DEFAULT_DEDUP_THRESHOLD = 0.30`, validated against real
corpus content, not picked arbitrarily:

| Case | Measured Jaccard similarity |
|---|---|
| Genuine near-duplicate (real PTO-accrual section vs. a reworded FAQ restatement of the same policy) | 0.370 |
| Genuine near-miss (VPN v1 vs. v2 MFA/password sections — same vocabulary, opposite facts) | 0.229 |
| Unrelated control (VPN password reset vs. PTO request) | 0.031 |

0.30 sits strictly between the near-miss ceiling (0.229) and the
near-duplicate floor (0.370), deliberately closer to the near-miss side:
an incorrectly *kept* duplicate only costs a little context-budget space,
while an incorrectly *collapsed* near-miss can silently delete a real,
distinct, possibly safety-relevant fact (e.g. the current mandatory-MFA
chunk collapsing into the deprecated optional-MFA chunk). Known limitation:
a heavily synonym-substituted paraphrase of the same fact measured at
0.172 — below the near-miss ceiling — so this heuristic is documented to
under-detect (not over-collapse) an aggressively reworded duplicate; that
is the accepted failure direction, not a gap glossed over.

The higher-rerank-score (or, in the live pipeline, higher-status-adjusted-
score) survivor wins any collapsed cluster.

## Context budgeting (Phase 8 Task 3)

`services/retrieval/src/resolvegrid_retrieval/context_budget.py`'s
`assemble_context()` replaces the old unbounded `"\n\n".join(...)` over
every retrieved chunk with a real, measured token-budget cap.

**Budget derivation** — measured directly against this deployment's real
running `qwen3:14b`, not taken from the model's marketing/architecture
max (`context length: 40960` per `ollama show`) or from Ollama's
self-contradictory public docs (its FAQ says 4096, its Modelfile
reference says 2048): a real completion call was sent to the live Ollama
container, then `docker exec resolvegrid-ollama ollama ps` was checked
while the model was resident — its `CONTEXT` column reported **4096**,
the actual context window this deployment runs `qwen3:14b` with (no
`num_ctx` is set anywhere in the request path, so Ollama's runtime
default for this tag on this host governs). Budget built up from that
real ceiling:

| Reservation | Tokens |
|---|---|
| Fixed prompt boilerplate (measured 146, rounded up for headroom) | 200 |
| User's question | 250 |
| Room for the model's generated answer (citations included) | 800 |
| **Reserved subtotal** | **1250** |
| Raw headroom for retrieved context (4096 − 1250) | 2846 |

`DEFAULT_CONTEXT_TOKEN_BUDGET = 2000` — comfortably under the 2846-token
raw headroom (leaving ~800 tokens / ~28% margin against
`estimate_token_count`'s documented ±10-20% undercount relative to a real
BPE tokenizer), while still fitting a realistic 4-6 chunks of context per
answer at this corpus's ~300-500-token chunk size.

**Skip, not truncate, an oversized chunk.** This is a deliberate
citation-integrity decision, not a style preference: `verify_citations`
(below) checks "was chunk id X present in the context the model was
given," not "was *all* of chunk id X's text present." Truncating a
chunk's text to fit the remaining budget would let a citation to that
chunk look fully valid while the model may have only seen a fragment —
possibly cut off mid-sentence or mid-qualifier (e.g. losing a trailing
"...with no opt-out"). A confidently-cited partial fact is a strictly
worse failure mode in an IT-ops/policy context than the model honestly
saying the context doesn't fully answer the question (which the prompt
template already instructs it to do). `assemble_context` continues past
a chunk that doesn't fit and keeps trying subsequent, lower-priority
candidates, so one oversized chunk only costs itself, not the rest of
the budget pass. `ContextBlock` exposes `text`, `chunk_ids` (best-first,
exactly what made it into `text` — what citation verification needs),
and `dropped_chunk_ids` (budget-excluded, for observability).

## Citation verification (Phase 8 Task 4 / Task 7)

`services/retrieval/src/resolvegrid_retrieval/citation_verification.py`'s
`verify_citations()` — a deterministic, no-model-call check: every literal
`[chunk:<id>]` token in the model's answer is parsed (`_CITATION_PATTERN
= r"\[chunk:(\d+)\]"`, matching only the exact form the prompt instructs
the model to produce; a differently-cased or malformed variant is simply
not recognized as a citation, neither verified nor fabricated) and
classified as verified (its id is in the caller-supplied
`valid_chunk_ids`) or fabricated (it is not).

The critical semantic: `valid_chunk_ids` means "shown to the model for
*this* context" (`ContextBlock.chunk_ids` — after rerank, status-adjust,
dedup, and budgeting), never "exists anywhere in the corpus." A citation
to a real, persisted chunk that was simply reranked/deduped/budgeted out
of *this* answer's context is exactly as fabricated, for this answer, as
a citation to an id that doesn't exist at all.

**Graph-level consequence: strip, don't abstain.** `verify_citations_node`
(`services/agent-orchestration/.../graph.py`, runs after `compose_response`,
before `finalize`) removes every fabricated `[chunk:<id>]` marker from
`output_text` in place (exact character-span surgery via
`Citation.start`/`end`), leaving the surrounding prose and any genuinely
verified citations untouched. The whole answer is never discarded or
replaced with a refusal on a fabricated citation — one untrustworthy
claimed source doesn't make the rest of a possibly-correct,
possibly-mostly-general-knowledge answer untrustworthy. A harder
consequence (abstain/escalate the whole answer on any fabrication) was
considered and rejected as disproportionate for this phase; flagged as a
real, documented gap for a future task with a stricter trust bar (e.g.
`risk_level == "high"`), not silently skipped. `state["citations_verified"]`
records whether verification found zero fabrications (vacuously `True`
for a zero-citation answer); `/chat` surfaces only `verified_chunk_ids`
as real citations to the UI, never the pre-verification list.

## `Document.status`-aware distractor fix (Phase 8 Task 7 Part B)

`services/retrieval/src/resolvegrid_retrieval/status_adjustment.py`'s
`apply_status_adjustment()` fixes a real regression Task 6 found (see
`docs/EXPERIMENT_REGISTRY.md`'s `phase8_reranking_v1` entry): reranking
alone flipped one of the three VPN v1/v2 supersession-distractor golden
cases, ranking the deprecated `Kestrel VPN Access Policy (v1, deprecated)`
chunk above the current `(v2)` chunk for a "what client software should I
install" query, by a thin, non-robust cross-encoder margin (0.989 vs.
0.980).

**Mechanism**: a full-range score penalty, not a soft multiplicative
discount. `DEFAULT_SUPERSEDED_PENALTY = 1.0` is subtracted from every
candidate whose `Document.status == "superseded"`. Since `rerank()`'s
output is a `[0, 1]` sigmoid probability, this is a strict guarantee —
whenever both a superseded and non-superseded candidate are present, the
non-superseded one always sorts first, regardless of how thin or wide the
original rerank margin is. A smaller/multiplicative penalty tuned only to
survive the one measured 0.009-point gap would have been a fragile,
case-specific fix; the full-range penalty is correct for any margin the
cross-encoder might produce.

**Deprioritization, not filtering**: a superseded chunk is never removed
from the candidate list, only pushed below every non-superseded one. It
can still survive dedup and reach the final budgeted context if no
non-superseded chunk covers the same point (e.g. a user genuinely asking
"what was the OLD policy") — matching this phase's established
under-hiding-over-deleting philosophy from `dedup.py`.

**Architectural placement**: `Document.status` lives in `apps/api`'s
Postgres schema; `services/retrieval` has no DB dependency, so this
module takes `superseded_chunk_ids` as a plain `set[int]`/`frozenset[int]`
the caller (`apps/api/src/resolvegrid_api/agent_retrieval.py`) resolves
from data it already queried, rather than this module querying the DB
itself. Applied after `rerank()`, before `dedup()` (dedup's own defensive
resort by `(-score, chunk_id)` must see the already-penalized score, or
it would undo the reordering).

**Re-measured result** (`docs/EXPERIMENT_REGISTRY.md`'s `phase8_reranking_v2`
entry): all 3 distractor cases now report `distractor_beats_best_relevant
=False` with a wide (6-8 rank) positive margin, not just a narrow correct
order. The previously-flipped case moved from margin −1 (flipped) to
margin **+8** (fixed). The eval harness's regression-guard assertion was
tightened from "at most 1 of 3 flips" (Task 6's observed ceiling) to
zero-tolerance — a future change reintroducing even one flip now fails CI.

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
`classify_intent → retrieve → compose_response → verify_citations →
finalize` — the `verify_citations` node (Phase 8 Task 7 Part C) is new,
running deterministically after `compose_response` and before `finalize`.

The `retrieve` node calls an injected `RetrieveFn` (same
dependency-injection pattern as `CompleteFn` — this package never imports
`resolvegrid_api` directly). Its real implementation,
`apps/api/src/resolvegrid_api/agent_retrieval.py`'s `retrieve_for_agent`,
now runs the full pipeline before the graph ever sees a result: **fuse →
rerank → status-adjust → dedup → token-budget** (fuse_rrf's full
candidate list, not pre-truncated; see the sections above for each
stage). The result attaches `retrieved_chunks` (exactly the chunks whose
formatted entry survived into the final budgeted context —
`ContextBlock.chunk_ids`, not the larger pre-budget candidate pool),
`retrieval_sufficient`, and `context_block` (the pre-assembled,
already-budgeted text) to state; any retrieval failure (including a
`RerankError` from a broken/missing optional reranker dependency)
degrades softly to "no chunks, not sufficient" rather than crashing the
graph — reranking specifically degrades to the unreranked fused order,
never to an empty result, so a missing `[reranker]` extra can't zero out
otherwise-good retrieval.

`compose_response` branches its prompt on `retrieval_sufficient AND
chunks non-empty AND context_block non-empty`: sufficient → substitutes
`context_block` directly into `_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE`'s
`<retrieved_context>...</retrieved_context>`-delimited slot (see
"Prompt-injection fix" in the Security section below); otherwise → the
original Phase-6 general-knowledge-only prompt.

`verify_citations` (new) then runs `resolvegrid_retrieval.
citation_verification.verify_citations` against `output_text` and
`retrieved_chunks`, and strips any fabricated `[chunk:<id>]` marker from
`output_text` in place — see "Citation verification" above for the full
strip-not-abstain reasoning. `state["citations_verified"]`/
`verified_chunk_ids`/`fabricated_chunk_ids` are threaded through for
`/chat` to build its response from.

**Deliberate scope limit** (documented in `graph.py`'s own module
docstring): this is *not* the plan doc's full "sufficient → answer with
citations, insufficient → abstain/clarify" routing. There is no hard
abstention branch — an insufficient retrieval result still gets a
best-effort general-knowledge answer, since `assess_sufficiency`'s bar
answers "is there company-specific evidence worth citing," not "is this
question answerable at all." Likewise, a fabricated citation strips that
one citation rather than forcing abstention on the whole answer (see
above).

`apps/api/src/resolvegrid_api/routers/chat.py`'s `/chat` endpoint resolves
`build_authz_filter(principal, session)` per-request and passes it into
the graph as a plain `retrieval_scope` dict. The response surfaces
`citations` (chunk id + document title, filtered to `verified_chunk_ids`
only — never a fabricated or budget-excluded chunk) and a `caption`
**only** when `retrieval_sufficient AND retrieved_chunks` — mirroring
`graph.py`'s own branch condition exactly, so the UI never claims a
grounded answer the model wasn't actually given context for, or a
citation the deterministic verifier rejected. The old Phase-6 "no ticket
or company-specific knowledge base yet" caption is replaced by
`"General-knowledge answer — no matching company knowledge-base article
was found for this question."` for the ungrounded case, since a real KB
now exists.

## Security: prompt-injection finding and fix (Phase 8 Task 4 / Task 7)

See `docs/SECURITY.md`'s "Phase 8" section for the full write-up: Phase 7's
retrieval pipeline handed the model raw, undelimited retrieved chunk text,
and a real live-model probe against `qwen3:14b` confirmed the model would
follow an embedded "SYSTEM OVERRIDE"-style instruction hidden inside a
retrieved chunk (0/2 real runs resisted it). Task 7 fixed this by wrapping
retrieved context in explicit `<retrieved_context>`/`</retrieved_context>`
delimiters with an untrusted-data framing, and re-verified against the
real live model on two different injection framings: 3/3 runs resisted
each, byte-identical output across all runs, a full reversal of the
pre-fix result.

## Evaluation

`apps/api/src/resolvegrid_api/eval_retrieval.py` — see
`docs/EXPERIMENT_REGISTRY.md`'s `phase7_retrieval_v1` (Phase 7 baseline:
recall@5=1.0000, precision@5=0.2000, MRR=0.9524, nDCG@5=0.9643, zero
authz leakage, zero distractor wins), `phase8_reranking_v1` (Task 6:
reranking alone — MRR/nDCG improve to 0.9643/0.9736, but introduces a
1-of-3 distractor flip, reported honestly as a mixed result), and
`phase8_reranking_v2` (Task 7: `apply_status_adjustment` fixes the flip —
all 3 distractor cases back to correct with a wide 6-8 rank margin, the
eval harness's regression assertion tightened to zero-tolerance) entries
for the full before/after numbers. `run_eval` now accepts an optional
`reranker_model` argument (`None` reproduces Phase 7's baseline path
byte-for-byte; set, it runs rerank → status-adjust → dedup in the same
order as the live `agent_retrieval.retrieve_for_agent` pipeline, so a
flip in the harness means the live graph has the same bug). Golden cases
key relevance by `(Document.title, Chunk.ordinal)`, not `Chunk.id`
(unstable across re-ingestion), resolved at eval-run time against
whatever corpus is actually loaded.

`eval_retrieval.main()` ingests-then-rolls-back inside one session (zero
DB residue for a manual run against a shared dev DB) — fresh-state
verification runs instead ingest via `ingestion_worker.main()` (which
commits) and then run `run_eval()` directly against that persisted
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

As of Phase 8 Task 8's fresh-state verification (2026-08-29, a full
`docker compose down -v` followed by re-ingestion from scratch), the local
dev database has the seed corpus ingested (8 documents, 36 chunks, 36
embeddings) and 75 employees seeded (`--seed 42`, 78 total including 3
pre-existing test fixtures) — left in place deliberately, not restored to
empty, matching Phase 7 Task 10's precedent (this phase's implementers
are likely to build on a populated knowledge base and directory, not an
empty one; re-ingesting/re-seeding is a one-command operation either way).
The two `AgentRun`/8 `Span` rows produced by this task's own browser
walkthrough were deleted afterward as incidental chat-session artifacts,
not meaningful state — mirroring Phase 7's identical cleanup call. See
`docs/PROGRESS.md`'s Phase 8 row for the full verification evidence.

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
