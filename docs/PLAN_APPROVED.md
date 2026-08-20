# ResolveGrid — Implementation Plan (Draft for Approval)

## Context

`C:\Dev\ResolveGrid\plan.md` is a master specification the user wrote for a greenfield, production-shaped internal IT service-management + AI-ops platform — meant to be a genuinely realistic enterprise product (not a shallow resume demo), with role-aware surfaces, a measurable agentic RAG workflow, durable HITL approvals, real observability/cost accounting, and evaluation-driven claims about performance and quality improvements. The workspace (`C:\Dev\ResolveGrid`) currently contains only that spec file — no code, no git repo, true greenfield.

Before finalizing an implementation plan, four research threads were run: (1) read-only inspection of the sibling project `C:\Dev\Project WIN` for reusable architectural lessons and pitfalls to avoid, (2) confirmation of the current workspace state, (3) primary-source research on LangGraph/Google ADK/LiteLLM/A2A/Ollama model selection for the target RTX 4080 12GB machine/pgvector indexing, and (4) primary-source research on self-hostable LLM observability tooling (Langfuse/Phoenix/OTel/Prometheus/Grafana) and enterprise IT-support/AI-observability UI patterns. That research surfaced one material conflict the user needed to resolve directly: Google ADK's own documentation shows it wants to own orchestration itself (not defer to LangGraph as originally hoped), and A2A is explicitly not a sub-agent/tool-call protocol — it's for independently-trusted service boundaries this local platform doesn't have. The user resolved this by choosing to include ADK for exactly one bounded, in-process specialist agent, with no A2A in v1, and to configure Anthropic as the primary cloud model provider with OpenAI as fallback behind LiteLLM.

This document is the resulting plan: it proposes a product/company identity, resolves the stack's responsibility boundaries against the research findings, defines the domain model, agent workflow, RAG design, evaluation strategy, telemetry/cost schema, security posture, frontend IA, documentation system, and a dependency-ordered set of implementation phases — each with explicit entry dependencies and objective exit criteria, and no time estimates, per the spec's own constraints. No code, scaffolding, or dependencies have been created; this plan is presented for approval only.

---

## 1. Executive Summary, Assumptions, Non-Goals, and Proposed Identity

### Proposed identity

**Fictional employer (the org the platform serves):** **Kestrel Softworks, Inc.** — a ~1,000-employee enterprise software company. "Kestrel" reads as precise/technical (a fast, sharp-sighted falcon) without colliding with an obvious real ITSM/AI vendor, and gives departments, locations, and email domains (`@kestrelsoftworks.com`) a plausible, non-generic anchor.

**Product name:** **ResolveGrid** (kept — it was a candidate, not mandatory, but it earns its place). "Resolve" maps directly to incident/ticket resolution; "Grid" evokes both the org chart (reporting grid) and the dependency/service grid the agent reasons over. It's short, has no cutesy AI-branding smell (no "Sentinel/Aura/Nexus"), and reads as something a platform-engineering team would actually name an internal tool. Framing: ResolveGrid is built and operated by Kestrel's internal Platform Engineering / IT organization for Kestrel's own employees — this framing is what justifies the RBAC, department, and approval-authority modeling being taken seriously rather than generic.

### Assumptions
- Single-developer-paced, iterative delivery on the stated local machine (RTX 4080 12GB / 32GB RAM, Windows host, Docker Desktop/WSL2 for Linux containers).
- All data is synthetic; no real PII, no real production traffic. Every claim about scale/latency/cost is a target to be measured, not asserted.
- Approval of this plan authorizes scaffolding and Phase 1 only in spirit — actual "start coding" still requires the user to exit plan mode.
- Optional AWS mapping is documentation-only unless separately approved; no cloud resources are created as part of this plan.

### Non-goals (v1)
- HR, Finance, or any department beyond internal IT support.
- A2A protocol (evaluated, deferred — see §4).
- External policy engine (OPA/Cedar) — deferred, in-process authorization module used instead (see §3, §10).
- Nx/Turborepo-class build-graph tooling — deferred as premature (see §3).
- Kafka/NATS-class event bus — deferred; Redis-backed job queue covers the concrete async needs identified (see §3).
- Self-RAG/CRAG-style corrective retrieval loops by default (see §7) — deferred behind measurable triggers.
- Multi-tenant support, mobile app, real cloud deployment.

---

## 2. Users, Workflows, Domain Model, and Authorization

### Users
Four primary role-aware surfaces (Employee, IT Analyst, Approver, AI/Platform Admin) plus a narrow Super-Admin (identity/auth management only, not a large UI surface), exactly as specified in plan.md. Roles are **not** mutually exclusive fields on a user — modeled as `RoleAssignment` rows so a person can be, e.g., both an Analyst and an Approver for a specific department, matching how real staged approval eligibility works.

### Domain model (schema-shape, not DDL)

**Org / identity**
- `Employee` — id, displayName, email, title, employmentStatus, hireDate, terminationDate?, timezone, `locationId` FK, `departmentId` FK, `teamId` FK, **`managerId` self-referential FK** (reusing Project WIN's pattern) plus a cycle-guard equivalent to `wouldCreateCircle` enforced in the authz/org service before any manager reassignment.
- `Department` — id, name, `parentDepartmentId` self-ref (hierarchy — the gap Project WIN left as flat strings), costCenter.
- `Team` — id, name, `departmentId` FK, `leadEmployeeId` FK.
- `Location` — id, name, region, timezone.
- `RoleAssignment` — employeeId, role, scope (global/department/team), grantedBy, grantedAt, expiresAt? — the basis for approval-eligibility resolution per stage.
- `AccessGroup`/`Entitlement` and `EmployeeEntitlement` (grant/revoke history, sourceTicketId link).
- `Device` — assetTag, employeeId? (nullable for pooled devices), type, os, status.

**Ticketing**
- `Ticket` — requesterId, assigneeId?, queueId, type (incident/service_request), category, priority, status, slaDueAt, timestamps, source.
- `TicketMessage` — ticketId, authorType (employee/analyst/agent/system), authorId?, body, createdAt, `agentRunId?` (links a message to the trace that produced it).
- `TicketStateTransition` — explicit history row per transition (fromStatus, toStatus, actorType, actorId?, reason) — durable audit beyond "current status only," enforced by a **single state-machine module** with TS `as const`-style exhaustive unions on the frontend/contracts side and mirrored Python enums, reusing Project WIN's lightweight pattern but centralizing it (Project WIN scattered guard clauses).
- `Queue`, `ServiceCatalogItem` (requiresApproval flag → `ApprovalPolicy`).

**Approvals (HITL)**
- `ApprovalRequest` — ticketId?, agentRunId, actionType, actionParamsJson, boundEvidenceRefs, riskContext, status, expiresAt, and a **snapshot hash** covering action+params+actor+evidence+expiry (the binding plan.md requires).
- `ApprovalDecision` — approvalRequestId, approverId, decision, comment, decidedAt, decision-time evidence snapshot.
- `ApprovalPolicy` — staged config (peer-review stage → higher-authority stage), eligibility resolved per stage from `RoleAssignment`, modeled after the JSM-style staged pattern noted in research.

**Knowledge (RAG)**
- `Document` (sourceType: public/synthetic_private, url?, publisher?, retrievedAt?, effectiveDate?, checksum, license?, status incl. superseded/stale/restricted, `supersedesDocumentId` self-ref, accessScope tags).
- `DocumentVersion` (parserVersion, contentHash) — independent of source-document changes.
- `Chunk` (documentVersionId, ordinal, text, tokenCount, `parentChunkId?` for parent-child retrieval, metadataJson for effective dates/access tags).
- `Embedding` (chunkId, embeddingModel, embeddingVersion, vector) — separate from `Chunk` so re-embedding doesn't destroy lineage.
- `IngestionRun` (parserVersion, chunkingVersion, embeddingVersion, stats).

**Tool audit / mutations**
- `ToolCall` — agentRunId, toolName, toolVersion, inputParamsJson, outputJson?, status (success/error/timeout/dry_run), errorTaxonomyCode?, idempotencyKey?, approvalRequestId?.
- `AuditLog` — actorType, actorId?, action, entityType, entityId, **beforeJson/afterJson diff columns**, metadataJson, **`recordHash`/`previousRecordHash` hash chain** for tamper-evidence — directly closes the two gaps flagged in Project WIN's audit schema (no diff, no immutability guarantee).

**Traces / cost (ResolveGrid-owned, not Langfuse's schema)**
- `AgentRun`, `Span` (per-stage, one of the 13 workflow stages), `ModelCall` (provider, model, purpose, promptVersion, tokens, latency, retries, cacheStatus, routingReason, fallbackReason?, errorCode?, providerCostUsd?, estimatedCostUsd, `pricingVersionId`), `PricingVersion` (versioned so historical cost never changes when current prices change), `EvalRun` (full version tuple).

### Authorization
Single principle: **agents and services inherit the requesting principal's permissions, enforced by a centralized in-process authorization module (`packages/authz`), never by UI hiding, and never re-implemented per service** — see §3 Decision C for the alternatives considered.

---

## 3. Recommended Architecture and Alternatives Rejected

Per plan.md's instruction to challenge the stack and prefer "clear bounded modules over premature microservices," three decision points are presented at the level that actually matters — not a full three-way system rebuild.

### Decision A — Process/service topology
- **Option 1 (rejected — too rigid for v1):** Every bounded module (web, api, agent-orchestration, retrieval, evaluation, adapters) as an independently deployed microservice.
- **Option 2 (rejected — under-powered):** Single process, everything synchronous, nothing async.
- **Option 3 (recommended):** A **modular monolith** — one FastAPI deployable that *imports* `agent-orchestration`, `retrieval`, `authz`, `telemetry`, and `operational-adapters` as Python libraries with clean module boundaries, **plus** a small number of Redis/Arq-backed worker processes for the concrete async needs actually identified: document ingestion/embedding, batch evaluation runs, load-replay execution, and notification fan-out. The core ticket→agent-turn request/response stays synchronous, streamed via SSE, and durability comes from LangGraph's `AsyncPostgresSaver` checkpointing — not from a queue.
  - **Concrete future trigger to split further:** sustained CPU/memory contention between API request-handling and agent-graph execution, or a proven need to scale agent workers independently of the HTTP tier. At that point the agent-orchestration library becomes a worker process consuming a queue — still not network-boundary microservice/A2A sprawl.

### Decision B — Monorepo tooling
- **Option 1 (rejected — overkill):** Nx or Turborepo. Both shine with many packages, remote caching, and multi-team scale; this repo has ~8 bounded modules and one contributor pace. The build-graph/caching machinery isn't earning its configuration cost yet.
- **Option 2 (recommended):** **pnpm workspaces** for the TypeScript side (`apps/web`, generated contract types) + **uv workspace** for the Python side (`apps/api`, `services/*`, `packages/authz`, `packages/telemetry`) + a root **justfile/Makefile** for cross-cutting commands (`just dev`, `just seed`, `just test`). uv is chosen over Poetry for its native workspace support and speed, matching pnpm's workspace model conceptually.
  - **Concrete future trigger to add Nx/Turborepo:** measured CI time dominated by redundant cross-package rebuilds, or package count growing past what a justfile can readably orchestrate.

### Decision C — Depth of the centralized authorization module
- **Option 1 (rejected — the flagged anti-pattern):** Thin per-service authz helpers duplicated into every service method, as in Project WIN. Explicitly rejected because it's the exact gap the user asked to fix.
- **Option 2 (rejected for v1 — premature):** External policy engine (OPA/Rego or AWS Cedar) evaluated out-of-process. More powerful (hot-reloadable policy, non-engineer policy authors, formal language) but adds a network hop and a new DSL to an already technology-dense stack, with no current requirement for runtime policy updates without redeploy.
  - **Concrete future trigger:** policy complexity or authorship needs outgrow in-process rules (e.g., compliance/legal wants to edit policies without a deploy).
- **Option 3 (recommended):** A single `packages/authz` module exposing one policy-evaluation entry point (`authorize(principal, action, resource, context) -> Decision`) that **every** FastAPI route dependency, **every** LangGraph tool/retrieval node, and **every** operational-adapter call goes through before doing anything. Decisions are logged to `AuditLog`. This directly satisfies plan.md's "RBAC and resource authorization must be enforced by backend and tool execution, not merely hidden in the UI."

These three decisions are justified against plan.md's own criteria: bounded modules over microservices, and "add queues/events only where async execution, retries, or isolation require them" (ingestion/eval/load-replay/notifications are the only proven cases).

---

## 4. Responsibility of Every Technology and A2A Criteria

| Technology | Single defensible responsibility |
|---|---|
| Next.js/TypeScript | All four role-aware frontend surfaces; no business logic beyond presentation/optimistic UI. |
| FastAPI | HTTP boundary: auth, request validation, invoking the LangGraph graph, invoking operational adapters directly for non-agentic CRUD. |
| LangGraph | Application workflow orchestration, state, checkpoints (`AsyncPostgresSaver`), retries, routing, resumability, HITL `interrupt()`/`Command(resume=...)`. **The only top-level orchestrator.** |
| Google ADK | Exactly one bounded specialist — a diagnostic/triage agent doing structured multi-step reasoning over ticket symptoms — invoked **in-process** as a LangGraph node/subgraph, never over a network boundary, never owning orchestration. |
| LiteLLM (self-hosted Proxy) | Model gateway: provider abstraction, virtual-key budgets, fallback routing, centralized normalized usage/spend logging. Requires Postgres (already in stack) for budget state. |
| Ollama | Zero-marginal-cost local inference: Qwen3 14B (Q4_K_M) for chat/reasoning, `nomic-embed-text` v1.5 for embeddings. |
| bge-reranker-v2-m3 | Local reranking process via `sentence-transformers`/`FlagEmbedding` (no native Ollama support confirmed) — a small standalone local service, not shoehorned into Ollama. |
| PostgreSQL/pgvector | System of record for all domain entities + vector store (HNSW index) + lexical search (tsvector/ts_rank_cd), fused via RRF. |
| Redis | Job queue backing (Arq) for ingestion/eval/load-replay/notifications, plus the caching layer (§7). |
| OpenTelemetry Collector | Single ingestion/normalization layer for every instrumented stage, using GenAI semantic conventions. |
| Langfuse (self-hosted) | LLM-specific trace/eval/cost UI and system of record for prompt-level debugging — **not** the sole source of truth (see §9). |
| Prometheus + Grafana | Infra/system metrics only (latency percentiles, error rate, queue depth, saturation) — explicitly not LLM trace storage, to avoid duplicating Langfuse. |
| Anthropic | Primary cloud model provider, behind LiteLLM. |
| OpenAI | Fallback cloud provider, behind LiteLLM. |
| A2A | **Skipped for v1** — documented evaluated-and-deferred. |

### A2A criteria (deferred, documented)
A2A is warranted only when a specialist agent must run as an **independently deployed and independently scaled service with its own trust boundary** — e.g., if the ADK diagnostic specialist later needs its own deployment lifecycle, its own team ownership, or must be reachable by a system outside ResolveGrid's process. None of that is true today; the specialist is a bounded in-process subgraph. This is the concrete future trigger to revisit.

---

## 5. Monorepo Layout, Local Topology, and Optional Cloud Mapping

### Monorepo layout (proposed top-level paths under `C:\Dev\ResolveGrid`)

```
apps/
  web/                      # Next.js/TypeScript — all 4 role surfaces
  api/                      # FastAPI — HTTP boundary, auth, route → service/graph invocation
services/
  agent-orchestration/      # LangGraph graphs/nodes/subgraphs, ADK specialist subgraph, checkpointer config
  retrieval/                # Parsers, chunkers, embedders, hybrid retrieval, reranker client, ingestion jobs
  evaluation/               # Golden dataset runner, deterministic graders, judge calibration, Arq eval workers
  operational-adapters/     # Ticketing, identity/account, directory, assets/entitlements, service health,
                             #   knowledge, approvals, notifications, audit — vendor-neutral adapter interfaces
packages/
  contracts/                # Shared typed contracts: tool schemas, event schemas, eval-case schema,
                             #   OpenAPI/JSON Schema → generated TS types + Pydantic models
  authz/                    # Centralized authorization/policy module (single call-through point)
  telemetry/                # OTel instrumentation helpers, cost-schema client, shared structured logging
infra/
  docker-compose.yml        # Postgres+pgvector, Redis, Ollama, LiteLLM proxy, OTel Collector, Langfuse stack, MinIO
  migrations/                # Alembic
  otel-collector/           # Collector config
  grafana/ prometheus/      # Dashboard provisioning, scrape config
  aws/                      # Optional, documentation/IaC-mapping only, gated behind separate approval
docs/
  ARCHITECTURE.md  DATA_MODEL.md  WORKFLOWS_TOOLS.md  RAG_INGESTION.md  EVALUATIONS.md
  TELEMETRY_COST.md  SECURITY.md  API_CONTRACTS.md  DESIGN_SYSTEM.md  EXPERIMENT_REGISTRY.md
  RUNBOOKS.md  DECISION_LOG.md  PROGRESS.md   # the single ledger
  adr/                      # individual ADRs
eval/
  golden/                   # versioned golden-dataset fixtures
  adversarial/              # adversarial case fixtures
  workloads/                # replayable load-test manifests
  results/                  # raw run outputs (large artifacts kept out of git where needed; manifests tracked)
scripts/                    # bootstrap, seed, load-test invocation
CLAUDE.md                   # lean index/pointer only
README.md                   # human-facing entry point (setup, stack, structure)
docker-compose.yml -> infra/docker-compose.yml (symlink or root convenience)
justfile
pyproject.toml              # uv workspace root
package.json                # pnpm workspace root
.env.example
```

### Local topology
Docker Compose brings up: Postgres (pgvector extension), Redis, Ollama, self-hosted LiteLLM proxy (using the same Postgres for budget/spend state), OTel Collector, Langfuse's own compose stack (Postgres+ClickHouse+Redis+MinIO), Prometheus + Grafana. **Reuse decision:** the MinIO instance Langfuse's stack already brings in is reused as ResolveGrid's own object store for document originals and eval-artifact storage, rather than standing up a second S3-compatible service — a direct application of "smallest non-duplicative combination."

### Optional AWS production mapping (documentation only, not built)
ECS/Fargate for `apps/api`, `apps/web`, and Arq workers; RDS for PostgreSQL (Aurora PostgreSQL supports pgvector) as the system of record; ElastiCache for Redis; S3 replacing local MinIO; Secrets Manager for credentials; managed OTel ingestion or self-hosted Collector on EC2/Fargate with Langfuse either self-hosted on EC2 or its managed cloud offering. This mapping stays 1:1 with the compose services so no local-only architecture decisions have to change to move to it. **Gated: not implemented unless separately approved.**

---

## 6. Agent Workflows, Tools, Approvals, and Error Handling

### The 13 stages mapped onto LangGraph

| # | Stage (plan.md) | LangGraph mapping |
|---|---|---|
| 1 | Authentication and requesting principal | **Not a graph node** — resolved by FastAPI auth dependency before graph invocation; `Principal` (identity, role assignments, department, entitlements) passed into initial graph state as an immutable input the rest of the graph trusts. |
| 2 | Intent, risk, and data-scope classification | Node `classify_intent` — cheap local model (Qwen3 14B via Ollama), structured output (intent category, riskLevel, dataScopeTags). |
| 3 | Retrieval/tool decision | Conditional-edge router `decide_path`, reading classification output; routes to retrieval subgraph, direct-tool subgraph, the ADK specialist subgraph, or a clarification node. |
| 4 | Query rewrite/decomposition when justified | Node `rewrite_query` inside the retrieval subgraph, invoked **only** when the classifier flags ambiguity/multi-part — skipped otherwise to keep the baseline cheap. |
| 5 | Authorization-aware metadata filters | Node `build_authz_filters` — calls `packages/authz` to turn the principal into a concrete filter predicate (allowed accessScope tags, excluded restricted-document IDs) **before** any retrieval query runs. |
| 6 | Candidate retrieval, fusion, reranking, dedup | Subgraph `retrieve`: `vector_search` (pgvector HNSW, filtered) → `lexical_search` (Postgres FTS) → `fuse_rrf` → `rerank` (bge-reranker-v2-m3 local process) → `dedup`. |
| 7 | Evidence-sufficiency decision | Node `assess_sufficiency` — deterministic checks first (min top-k score, required-field coverage), model-assisted only when ambiguous. This is the hook point for a future corrective loop, but v1 just routes to abstain/clarify. |
| 8 | Tool selection and schema validation | Nodes `select_tool` + `validate_tool_schema` against typed contracts in `packages/contracts`. |
| 9 | Read-only execution or persistent approval interrupt | Read-only/low-risk tools → `execute_readonly_tool` directly. Mutating/high-risk → `request_approval`, which calls LangGraph's `interrupt()`, checkpointed via `AsyncPostgresSaver`. The `ApprovalRequest` row is written idempotently (upsert on a stable key) so it's safe if this node re-executes on resume. |
| 10 | Authorized mutation after approval | Node `execute_mutation`, placed **strictly after** the interrupt/resume boundary. Re-validates the approval snapshot hash (action+params+actor+evidence+expiry) before calling the tool — closing the LangGraph re-execution caveat and blocking tampered/stale replays. |
| 11 | Structured response and citation verification | `compose_response` (structured output with citation spans) → `verify_citations` (deterministic: every citation must map to a chunk actually present in context). |
| 12 | Safe answer, abstention, or escalation | Terminal router `finalize` — answer / abstain-with-reason / escalate-to-human, driven by upstream sufficiency/verification/risk flags. |
| 13 | Telemetry, evaluation hooks, feedback | Cross-cutting: every node wrapped with OTel span emission via `packages/telemetry`; terminal `emit_feedback_hooks` node writes `AgentRun`/`Span`/`ModelCall` and, for sampled runs, enqueues an async eval job rather than blocking the response. |

### Where the ADK specialist plugs in
Invoked from a single LangGraph node, `run_diagnostic_specialist`, as a plain in-process async call — never HTTP/A2A. `decide_path` routes here for tickets classified as multi-symptom diagnostic/triage cases (e.g., "VPN keeps disconnecting"). The specialist returns a structured `DiagnosticResult` (probable-cause categories, recommended next tool/questions, confidence) back into shared graph state, and the graph continues normally through stages 7–13. From LangGraph's perspective this is just a node with internal complexity — orchestration ownership never splits.

### Authorization-aware retrieval filtering (mechanism)
The filter predicate built at stage 5 is a **required, non-optional parameter** to `vector_search`/`lexical_search`, enforced inside the SQL query itself (not filtered post-hoc in application code), so a bug downstream cannot leak unauthorized chunks into context. The same filter's hash is part of the semantic-cache key (ties directly to cache-as-security-boundary in §10).

### Error handling / error taxonomy
Defined categories, each with a safe user-facing message and a policy: `ValidationError`, `AuthorizationError`, `NotFoundError`, `ToolTimeoutError`, `ToolExecutionError`, `ProviderError` (triggers LiteLLM fallback), `ApprovalExpiredError`, `ApprovalTamperError`, `ApprovalDuplicateReplayError`, `RetrievalEmptyError`. Each maps to whether it's retried, escalated, or results in abstention — abstention/escalation is explicitly treated as a **success** outcome per plan.md, not a failure.

---

## 7. Baseline RAG, Progressive Enhancements, Ingestion, and Caching

### Baseline (the simplest strong measurable baseline)
- Structure-aware, heading-preserving parsing/chunking (not naive fixed-width), target ~300–500 tokens with overlap; `parserVersion`/`chunkingVersion` tracked.
- Embeddings: `nomic-embed-text` v1.5 via Ollama (8192-token context, ~274MB — fits full chunks comfortably), stored in pgvector with an **HNSW** index (no training step, builds fine on empty tables).
- Lexical: Postgres `tsvector`/`ts_rank_cd` full-text search.
- Fusion: Reciprocal Rank Fusion combining vector + lexical ranked lists, per pgvector's own README guidance, with the fusion constant documented.
- Authorization/metadata filtering applied **before** the query, not after.
- Reranking: `bge-reranker-v2-m3` (or `bge-reranker-base` if latency requires) via a local `sentence-transformers`/`FlagEmbedding` process, applied to the fused top-N.
- Dedup of near-duplicate/adjacent chunks.
- Context budgeting: token-budget-aware assembly prioritized by rerank score, with citation mapping (chunk ID → inline marker).
- Full version tracking: Document/DocumentVersion/Chunk/Embedding.

### Progressive enhancements (each gated on a measurable trigger — not built by default)
- **HNSW vs IVFFlat comparison** — trigger: measured ANN latency/recall regression as corpus scale grows.
- **Chunk-size sweep** — trigger: recall/precision plateau or citation-verification failures traced to chunk boundaries.
- **Parent-child retrieval** — trigger: evals show reranked child chunks lack needed surrounding context.
- **Query expansion** — trigger: measured recall gap on paraphrased/long-tail golden-set queries.
- **Neighboring-chunk inclusion** — trigger: faithfulness failures from truncated context at chunk edges.
- **Reranker choice experiments** — trigger: reranking nDCG uplift below target or latency budget exceeded.
- **Freshness/conflicting-document handling** (effective dates, superseded flags surfaced in-prompt) — trigger: adversarial stale/conflicting-policy cases failing.
- **Access-control metadata refinement** — trigger: authorization adversarial cases (cross-department leakage) failing.
- **Self-RAG/CRAG-style corrective loops — explicitly deferred.** Concrete triggers to revisit: (a) sustained low retrieval-confidence below a defined threshold over a meaningful sample, (b) measured conflicting-source rate above threshold, (c) citation-verification failure rate above threshold, (d) high-risk requests where the single-pass sufficiency check is measurably insufficient. Any loop added later must be **bounded** (max iterations) and must escalate/abstain rather than loop indefinitely.

### Ingestion
Two clearly labeled sources per plan.md: public attributable knowledge (URL, publisher, retrieval date, version/effective date, checksum, license note, parser/chunk/embedding versions) and synthetic private company knowledge (internal policies/runbooks/catalog with intentionally stale, conflicting, restricted, and incomplete cases seeded deliberately). Ingestion runs as an Arq job (the first genuine async need — long-running, batchy, naturally isolated from the request/response path).

### Caching
Tiers: exact-response, semantic-response, retrieval, embedding, reranker, safe-read-only-tool, and provider prompt-prefix (via LiteLLM/Anthropic-OpenAI native caching). **Cache key composition always includes:** authorization-scope hash, workflow/prompt/model version, knowledge version (document/embedding version), locale. Scope levels: global (e.g., public-doc embeddings), role-scoped, department-scoped, user-specific, non-cacheable (anything touching individual ticket/PII specifics). Redis-backed, TTL + explicit event-based invalidation (e.g., document version bump invalidates dependent entries), every hit/miss logged with tier and scope to telemetry. **Cache authorization is treated as a security boundary** — cross-user leakage is a mandatory adversarial test class, not an afterthought (§10, Phase 11).

---

## 8. Enterprise Data and Golden-Dataset Strategy

~1,000 coherent synthetic employees generated with deterministic seeds and scenario manifests (reporting lines, departments, teams, locations, timezones, employment status, IT roles, approval authority, devices, entitlements, access groups, lifecycle dates). Connected ticket histories cover common and long-tail issues, duplicates, multi-turn cases, escalations, SLA breaches, approvals, outages, missing information, security-sensitive requests, tool failures, and reopened tickets. No real PII.

### Golden dataset (`EvalCase`)
Fields: identity/permissions fixture ref, request/context, intent, answerability, relevant-evidence refs, expected citations, tool/argument constraints, expected state transition, approval-policy expectation, forbidden actions, expected structured result, rubric, risk/difficulty, provenance, human-review status. **A model may draft candidate cases, but nothing becomes "gold" without passing a human-review gate** (`humanReviewStatus`) — the same model never both generates and self-grades its own answers.

### Grading order
Deterministic graders run first and are authoritative wherever possible: schema validity, authorization/approval-compliance exact checks, retrieval recall/precision/MRR/nDCG against labeled relevant evidence, citation correctness (chunk-membership check), state-transition correctness, latency/tokens/cost (measured directly). Model-judge graders are reserved for genuinely fuzzy dimensions — groundedness/faithfulness, free-text correctness, abstention-appropriateness — and are **calibrated** against a human-reviewed sample with reported inter-rater agreement, never assumed reliable.

### Adversarial cases (concrete instantiations)
Injected-document (prompt injection embedded in a knowledge chunk), cross-user data request, conflicting/stale policy pair, unsupported question (expect abstention), fabricated ticket/asset ID (expect graceful not-found, not hallucination), malformed/timed-out tool call, simulated provider outage (expect fallback), duplicate/expired approval replay, empty retrieval result.

### Versioning
Every `EvalRun` records dataset version, prompt version, workflow version, retriever version, reranker version, embeddings version, generation model version, judge version, tool-schema version, and git commit — enabling exact reproduction and fair before/after comparison.

---

## 9. Evaluations, Telemetry, Cost Accounting, and Admin-Console Requirements

### Telemetry/cost schema (ResolveGrid-owned)
`ModelCall` carries every field plan.md's "Model routing and cost accounting" section requires: trace/span ID, protected request/user identity (stored as a pseudonymized/hashed principal reference, not raw PII, in the telemetry table itself), provider, model, purpose, prompt/workflow version, input/output/cached tokens, latency, retries, cache status, routing/fallback reason, error, timestamp, provider cost, and locally estimated cost via a **versioned `PricingVersion`** table so historical costs never change when current prices change. Ollama calls show `$0.00` provider cost while retaining full latency/token data.

### Mapping to the observability stack
The **OTel Collector** is the single ingestion/normalization layer for every instrumented stage (API, auth, workflow node, model call, retrieval, reranking, cache, tool, approval, eval, response), using GenAI semantic conventions for model-call attributes. It fans out to: **Langfuse** (via OTLP, for LLM-specific trace/eval/cost debugging UI — mature built-in cost math, no other tool matches it) and **Prometheus** (infra metrics — latency percentiles, error rate, queue depth, saturation — feeding **Grafana**, deliberately *not* LLM trace storage, to avoid duplicating Langfuse). Critically, the app **writes directly** to ResolveGrid's own `AgentRun`/`Span`/`ModelCall`/`PricingVersion`/`EvalRun` tables at call time — the admin console queries this owned schema first; Langfuse is a secondary deep-debugging surface, never the sole source of truth. **Phoenix is skipped** (near-total overlap with Langfuse). **Risk flagged, not blocking:** Langfuse's ClickHouse acquisition (Jan 2026) — both parties committed to MIT/self-hosting staying first-class; monitor, don't block on it.

### Redaction
Raw PII (ticket body detail, employee names) is redacted/truncated before OTLP export to Langfuse where not needed for debugging; full detail lives only in the access-controlled primary Postgres store, with its own retention/deletion policy.

### Admin console requirement
Genuine telemetry only — no static mock charts. This is a hard exit criterion checked explicitly in Phase 14.

---

## 10. Security and Threat Model

- **Cache authorization as a security boundary.** Every cache key includes the authz-scope hash; adversarial eval cases specifically test cross-user/cross-department cache leakage as a required test class (Phase 11).
- **Prompt-injection boundaries.** Retrieved document content and tool outputs are always treated as untrusted data, never instructions — structurally separated from system/developer instructions in prompt construction (distinct roles/clearly delimited untrusted blocks). Tool-selection decisions cannot be driven by content found only inside retrieved chunks without passing through the normal classify → authorize → validate pipeline. Directly tested via the injected-document adversarial case (Phase 8).
- **Tool allowlists / least privilege.** Typed tool contracts live in `packages/contracts`; each declares a minimum required role/entitlement. Allowlist filtering happens **before** the model even sees available tool definitions — not just checked after the model "chooses" one.
- **Approval binding.** `ApprovalRequest` snapshot hash covers action type + params + actor + evidence refs + risk context + expiry; re-verified at `execute_mutation` time. Expired/tampered/duplicate-replayed approvals are rejected with explicit error-taxonomy codes, all logged to the hash-chained `AuditLog`.
- **Standard hardening, scoped concretely:** per-principal + per-IP rate limiting at the FastAPI middleware layer; secrets via `.env` locally / AWS Secrets Manager in the optional cloud mapping; safe error envelopes (no stack traces/internal identifiers surfaced to non-admin roles); container scanning (e.g., Trivy) as a CI step; documented `pg_dump`-based backup/restore runbook.
- **Agents inherit the requesting principal's permissions** — enforced structurally by threading `Principal` immutably through graph state and requiring every node touching data or tools to call `packages/authz` (§3 Decision C, §6).

---

## 11. Frontend IA, Design Research, and Recommended Visual Direction

### Skill availability check (honest report)
Installed and usable: `design-taste-frontend`, `frontend-design`, `impeccable`, `emil-design-eng` (Emil Kowalski's philosophy on UI polish, component design, animation decisions, and invisible-detail craft — a strong fit for the microinteraction/streaming/approval-state feel plan.md calls for). All four will be invoked through their real mechanism during Phase 14.

### Information architecture (role-aware, per plan.md)
- **Employee:** ask cited questions, create/track requests, provide missing info, see only authorized personal data.
- **IT Analyst:** flat single-queue triage (not tabbed multi-channel — analysts triage one queue of tickets/agent-runs), a **three-pane workspace** (queue list → record detail → correlated context/trace panel) rather than a 1:1-ported legacy-ITSM form layout, agent-assisted diagnosis, tool invocation, escalate/resolve, feedback.
- **Approver:** risk/evidence/affected-resources/parameters/rollback view, staged role-gated approval (peer-review → higher-authority, per-stage eligibility) modeled after JSM-style change management but not cloned.
- **AI/Platform Admin:** trace-as-timed-waterfall/tree with **inline per-node token/cost** (not a separate disconnected cost page); purpose-built single-question dashboards (Latency / Cost / Usage) instead of one kitchen-sink page.
- **Super-Admin:** narrow identity/auth management surface only — deliberately not expanded into a large UI area.

### Anti-patterns explicitly avoided
Gradient-hero "AI Score" vanity KPI cards, chat-bubble-styled trace viewers that hide timing/structure, generic AI gradients, gratuitous glass, decorative dashboards, huge empty cards, repetitive KPI grids, excessive rounding, delayed interactions, fake terminal aesthetics, inaccessible motion.

### Reused patterns from Project WIN (adapted, not copied)
Radix/shadcn table primitives, kebab-menu row actions, inline filter-row toggles, skeleton loading — reused, but **with virtualization and bulk multi-select added** for enterprise ticket volume (Project WIN lacked both). A global motion-config honoring `prefers-reduced-motion` with centralized animation tokens is reused directly.

Two or three concrete visual directions will be presented and one recommended **during Phase 14 itself** (not now, since design is a later-phase concern) using the `frontend-design`/`impeccable`/`design-taste-frontend` skills through their real mechanism.

---

## 12. Testing, Load Testing, Metric Definitions, and Before/After Experiments

### Test pyramid
Unit (state machines, deterministic graders, cache-key composition, cycle-guard), integration (DB + authz + tool contracts), **contract tests** (`packages/contracts` schemas validated on both the TS and Python sides via generated types — CI fails on drift), E2E (Playwright across all four role surfaces), security tests (adversarial eval suite, cross-user cache/retrieval leakage, injection resistance), evaluation tests (golden-dataset run as a CI gate on any PR touching prompts/graph/retrieval/reranker), load tests (the replay harness).

### Metric definitions
p50/p95/p99 latency (wall-clock per `AgentRun`, and per-stage via `Span`), TTFT (time to first streamed token from `compose_response`), stage latency (per-`Span` duration), queue time (Arq job wait time), error rate (error-status `ToolCall`/`ModelCall` ÷ total), throughput (completed `AgentRun`s/sec sustained), cache hit rate (per tier), tokens (input/output/cached per `ModelCall`, aggregated per run), cost/request and cost/successful-task (sum `estimatedCostUsd`, the latter restricted to task-completion=true cases), quality by route/model (`EvalRun` metrics grouped by provider/model).

### Before/after experiment protocol
Fixed dataset version + fixed seed + only the variable under test changes + paired comparison; raw results stored under `eval/results/<experiment-id>/`; comparison report includes hypothesis, independent variable, controlled variables, sample size, statistical treatment, result, limitations, and adopt/reject/inconclusive decision. Applies to reranking, caching, routing, chunk sizing, and every other tuning experiment.

### Load testing
Replayable workload generator simulating 1,000+ daily requests with realistic arrivals/bursts/concurrency/repeated shared questions — not a constant average rate. Local load, paid-provider tests, offline eval throughput, and interactive performance are measured **separately**. No claim of online-generation batching improvement unless LiteLLM/provider actually supports it for the tested path.

### Manual-lookup benchmark
Reproducible protocol comparing manual vs. AI-assisted task completion: defined sampling, start/stop rules, correctness threshold, median/percentile time, failure treatment, sample size, documented bias and limitations — reported honestly including cases where AI-assist doesn't win.

---

## 13. Persistent Documentation and Progress System

Canonical docs (all under `docs/`, each single-purpose, none a monolith): `ARCHITECTURE.md` + `adr/` (per-decision ADRs — e.g., the three decisions in §3), `DATA_MODEL.md`, `WORKFLOWS_TOOLS.md`, `RAG_INGESTION.md`, `EVALUATIONS.md`, `TELEMETRY_COST.md`, `SECURITY.md`, `API_CONTRACTS.md`, `DESIGN_SYSTEM.md`, `EXPERIMENT_REGISTRY.md`, `RUNBOOKS.md`, `DECISION_LOG.md`.

**`README.md`** stays the human-facing entry point (what/why, quick start, stack table, repo map) — kept current as a required step of every phase's "documentation updates," explicitly to avoid Project WIN's drift (2.4KB boilerplate README vs. a 24.8KB CLAUDE.md that became the de facto architecture doc).

**`CLAUDE.md`** stays a lean index only: durable instructions, critical commands, invariants, and links into the docs above — target budget on the order of a few KB, not a diary, not a duplicate of `ARCHITECTURE.md`.

**`PROGRESS.md`** is the single authoritative ledger — one table: Unit ID, Description, State (**Proposed / Approved / In Progress / Blocked / Implemented / Verified / Superseded**), owner/session ref, entry dependencies, exit-criteria evidence link, last-updated. Every phase in §14 becomes one or more ledger rows at execution time. No status is duplicated in any other document — other docs link to the ledger rather than restating state. Each future session is expected to: inspect `CLAUDE.md` + `PROGRESS.md` + repo/git state, pick the next approved bounded unit, record decisions, implement and verify only that unit, update evidence/status, and leave a resumable repository.

---

## 14. Dependency-Ordered Implementation Phases

No time estimates anywhere, per plan.md. Ordered so the walking skeleton (repo scaffold, Postgres+pgvector+Redis+Ollama, one real ticket flow, real telemetry) exists before RAG sophistication, evaluation rigor, caching, and the ADK specialist are layered on.

### Phase 1 — Repo scaffold & local infra skeleton
- **Objective:** Stand up the empty-but-wired monorepo and Docker Compose topology (Postgres+pgvector, Redis, Ollama, OTel Collector), no product features.
- **User-visible result:** `docker compose up` brings all infra containers to healthy; `GET /health` on FastAPI returns OK; Next.js placeholder loads.
- **Affected components:** repo root, `apps/api` skeleton, `apps/web` skeleton, `infra/docker-compose.yml`, `docs/` skeleton, `CLAUDE.md`, `README.md`.
- **Schema changes:** initial Alembic baseline migration (health-check table only).
- **Tests:** infra smoke test (compose up → health endpoints), CI skeleton (lint + typecheck).
- **Evaluation evidence:** none — explicitly N/A at this stage, documented as such.
- **Observability:** OTel Collector container healthy, receives a startup span from FastAPI.
- **Security:** `.env.example` with no real secrets; pinned container image versions.
- **Documentation:** README quick-start, `ARCHITECTURE.md` stub, `PROGRESS.md` created with this phase as its first row.
- **Entry dependencies:** none (greenfield).
- **Exit criteria:** all containers healthy; health endpoints return 200; CI green; docs skeleton committed.

### Phase 2 — Identity, org data model & seeded directory
- **Objective:** Employee/Department/Team/Location/RoleAssignment schema + deterministic seed generator (small scale, e.g. 50–100 employees first) with manager-hierarchy cycle guard.
- **User-visible result:** Admin can query the employee directory; org chart resolves correctly.
- **Affected components:** `apps/api` (directory endpoints), `packages/authz` skeleton, migrations, `scripts/seed`.
- **Schema changes:** `Employee`, `Department`, `Team`, `Location`, `RoleAssignment` + `managerId` index.
- **Tests:** cycle-guard unit tests (equivalent to `wouldCreateCircle`), seed determinism test (same seed → identical data hash).
- **Evaluation evidence:** N/A (no agent yet).
- **Observability:** DB-query spans emitted.
- **Security:** authz module stub enforces role checks on directory endpoints from day one.
- **Documentation:** `DATA_MODEL.md` org-entities section, `PROGRESS.md` update.
- **Entry dependencies:** Phase 1.
- **Exit criteria:** seed script deterministic and idempotent; cycle-guard tests pass; directory API demonstrably restricts visibility by role (tested).

### Phase 3 — Ticketing core + first authenticated UI flow (no AI yet)
- **Objective:** Ticket/TicketMessage/TicketStateTransition/Queue schema, centralized state-machine module, minimal employee-submit / analyst-triage UI flow.
- **User-visible result:** An employee submits a ticket; an analyst sees it in a queue and transitions its status.
- **Affected components:** `apps/api` ticket routes, `apps/web` employee + analyst surfaces, `packages/contracts` ticket schemas, `packages/authz` row-level checks.
- **Schema changes:** `Ticket`, `TicketMessage`, `TicketStateTransition`, `Queue`, `AuditLog` (hash-chained, introduced here since real mutations begin).
- **Tests:** illegal-transition rejection tests, authz visibility tests, audit hash-chain integrity test.
- **Evaluation evidence:** N/A (no AI yet).
- **Observability:** full web→api→db trace for ticket creation.
- **Security:** input validation, rate limiting on creation, append-only audit verified.
- **Documentation:** `WORKFLOWS_TOOLS.md` ticket-lifecycle section, `DECISION_LOG.md` entry for state-machine approach.
- **Entry dependencies:** Phase 2.
- **Exit criteria:** end-to-end manual create→triage→close works through real UI; state machine cannot be bypassed; every transition audited.

### Phase 4 — Model gateway & first real LLM call
- **Objective:** Self-hosted LiteLLM proxy + Ollama (Qwen3 14B) route; wire `ModelCall`/`PricingVersion`; prove real cost/latency capture on a trivial non-agentic call (ticket summarization).
- **User-visible result:** Analyst sees an AI-generated ticket summary with a real (not mocked) latency/cost badge.
- **Affected components:** `infra` LiteLLM service, `packages/telemetry`, `apps/api` summarize endpoint, `apps/web` ticket detail panel.
- **Schema changes:** `ModelCall`, `PricingVersion`.
- **Tests:** contract test — every LiteLLM call produces exactly one fully-populated `ModelCall` row; pricing-version immutability test.
- **Evaluation evidence:** none formal yet; first documented spot-check of correctness.
- **Observability:** OTel GenAI-convention spans; visible in Langfuse; LiteLLM metrics scraped by Prometheus.
- **Security:** LiteLLM virtual-key scoping, no raw provider keys in app code.
- **Documentation:** `TELEMETRY_COST.md` initial schema, README stack table updated.
- **Entry dependencies:** Phase 3.
- **Exit criteria:** one real call visible with matching trace IDs simultaneously in ResolveGrid's `ModelCall` table, Langfuse, and Grafana; Ollama shows `$0.00` cost with correct latency/tokens.

### Phase 5 — Cloud provider fallback routing
- **Objective:** Add Anthropic (primary) and OpenAI (fallback) behind LiteLLM; capture routing/fallback reasons.
- **User-visible result:** Admin can observe a forced fallback with `routingReason`/`fallbackReason` populated.
- **Affected components:** `infra` LiteLLM config, `packages/telemetry`.
- **Schema changes:** none beyond Phase 4 (populate existing columns).
- **Tests:** forced-failure fallback integration test.
- **Evaluation evidence:** none yet.
- **Observability:** fallback events as distinct span attributes.
- **Security:** separate scoped virtual keys with budgets per provider.
- **Documentation:** `TELEMETRY_COST.md` routing-policy section.
- **Entry dependencies:** Phase 4.
- **Exit criteria:** forced-failure test proves correct fallback + accurate logging; budget enforcement tested.

### Phase 6 — Minimal LangGraph agent workflow (stages 1-2-3-11-12-13 only)
- **Objective:** First real graph: authenticate → `classify_intent` → route (trivial: general-knowledge answer, no retrieval) → `compose_response` → no-op verify → `finalize` → telemetry, with `AsyncPostgresSaver` checkpointing wired up.
- **User-visible result:** Employee asks a free-text question and gets a model answer end-to-end (explicitly labeled "no knowledge base yet"); admin sees the run as a correlated trace.
- **Affected components:** `services/agent-orchestration` (new), `apps/api` chat endpoint, `apps/web` minimal chat surface, `AgentRun`/`Span`.
- **Schema changes:** `AgentRun`, `Span`.
- **Tests:** **checkpoint-restore test** (kill process mid-run at a node boundary, confirm resume from last checkpoint rather than restart), `classify_intent` schema-validity unit tests.
- **Evaluation evidence:** first tiny golden set (10–20 hand-written cases) for schema-validity/classification grading — establishes the eval harness scaffolding.
- **Observability:** per-node spans correlated under one `AgentRun`, visible in Langfuse as a graph trace.
- **Security:** principal resolved once and passed immutably; authz module invoked even though nothing restricts yet, to establish the pattern early.
- **Documentation:** `WORKFLOWS_TOOLS.md` agent-workflow section (stage numbering fixed here), ADR for the checkpointer decision.
- **Entry dependencies:** Phase 5.
- **Exit criteria:** checkpoint-restore test proves durable resumability (not just claimed); one full run correctly correlated across ResolveGrid's own tables and Langfuse.

### Phase 7 — Knowledge ingestion + baseline retrieval (stages 4-5-6-7)
- **Objective:** Document/DocumentVersion/Chunk/Embedding schema, structure-aware parser+chunker, `nomic-embed-text` embeddings, pgvector HNSW, Postgres FTS, RRF fusion, authz-aware filtering, sufficiency check; ingest a bounded real corpus slice (synthetic policies + attributed public vendor docs).
- **User-visible result:** Employee questions now return cited answers from real ingested documents; admin sees retrieval candidates/scores in the trace.
- **Affected components:** `services/retrieval` (new), `services/agent-orchestration` (retrieve subgraph wired in), Arq ingestion job (Redis job queue introduced here — first genuine async need), knowledge tables.
- **Schema changes:** `Document`, `DocumentVersion`, `Chunk`, `Embedding`, `IngestionRun`.
- **Tests:** parser/chunker unit tests, ingestion idempotency test, **adversarial authz-filter leakage test**, RRF fusion unit test.
- **Evaluation evidence:** retrieval-focused golden subset — recall/precision/MRR/nDCG measured, first real baseline numbers recorded.
- **Observability:** retrieval spans include candidate list, scores, filter hash.
- **Security:** cross-user retrieval-leakage adversarial test added to CI now.
- **Documentation:** `RAG_INGESTION.md`, public-source attribution log (URL/publisher/date/checksum/license).
- **Entry dependencies:** Phase 6.
- **Exit criteria:** zero-leakage adversarial retrieval test passes across N trials; baseline recall/precision/MRR recorded with exact versions in `EXPERIMENT_REGISTRY.md`; UI citations verifiably map to real chunks.

### Phase 8 — Reranking, dedup, context budgeting, citation verification
- **Objective:** Add bge-reranker-v2-m3, dedup, token-budget-aware context assembly, deterministic `verify_citations`.
- **User-visible result:** Measurably more precise answers; UI distinguishes verified-citation answers from flagged/abstained ones.
- **Affected components:** `services/retrieval` reranker process, `services/agent-orchestration` `verify_citations` node.
- **Schema changes:** none major (populate existing `rerankerVersion` fields).
- **Tests:** reranker integration test, citation-verification unit tests (fabricated citation correctly rejected).
- **Evaluation evidence:** before/after comparison vs. Phase 7 baseline (same golden subset), stored as a versioned `EvalRun`.
- **Observability:** rerank before/after spans, citation-verification pass/fail metric.
- **Security:** **injected-document adversarial case introduced** — must fail safely.
- **Documentation:** `EXPERIMENT_REGISTRY.md` reranker comparison, `EVALUATIONS.md` citation methodology.
- **Entry dependencies:** Phase 7.
- **Exit criteria:** injected-document case passes; reranking shows measured improvement or an honest documented negative result, reproducible.

### Phase 9 — Tools, approvals, and mutation execution (stages 8-9-10)
- **Objective:** Typed tool contracts, first bounded operational adapters (a read-only lookup + a mutating action), tool allowlist filtering, `ApprovalRequest`/`ApprovalDecision` schema, `interrupt()`/`Command(resume=...)` wiring, staged approval policy.
- **User-visible result:** Analyst gets an agent-proposed mutating action; approver sees a durable request (survives restart) bound to params/evidence/expiry; approved action executes and is audited.
- **Affected components:** `services/operational-adapters` (new), `services/agent-orchestration` (interrupt node, `execute_mutation` post-interrupt), `apps/web` approver surface.
- **Schema changes:** `ApprovalRequest`, `ApprovalDecision`, `ApprovalPolicy`, `ToolCall`, `AccessGroup`/`Entitlement`, `EmployeeEntitlement`.
- **Tests:** **interrupt-survives-restart test** (kill mid-approval, restart, resume, confirm no duplicate side effects — directly exercising the LangGraph re-execution caveat), approval-tamper test, expiry test, duplicate-replay adversarial test, tool-allowlist test.
- **Evaluation evidence:** tool/argument-accuracy and approval-compliance golden cases.
- **Observability:** approval spans, tool-execution spans, dry-run telemetry.
- **Security:** core threat-model phase — approval binding, allowlists, idempotency, error taxonomy, per-tool compensation/rollback guidance documented.
- **Documentation:** `WORKFLOWS_TOOLS.md` tool catalog, `SECURITY.md` approval-binding section, `RUNBOOKS.md` compensation runbook.
- **Entry dependencies:** Phase 8, Phase 3.
- **Exit criteria:** restart-mid-approval test proves zero duplicate mutation side effects; tamper/expiry/replay tests all pass; one real mutating tool works end-to-end with full audit trail.

### Phase 10 — Golden dataset expansion & evaluation harness
- **Objective:** Scale the golden dataset with full version-dimension coverage, deterministic-first grading, calibrated judges, Arq batch eval workers.
- **User-visible result:** Admin console shows a real evaluation dashboard (pass rates by dimension/version), not static numbers.
- **Affected components:** `services/evaluation` (new), Arq infra extended, `apps/web` admin eval views.
- **Schema changes:** `EvalRun` (fixtures versioned as files under `eval/`, results in DB).
- **Tests:** judge-calibration test (human-agreement measured and reported), grader-determinism test.
- **Evaluation evidence:** this phase's deliverable is the first comprehensive multi-dimension report.
- **Observability:** eval-run spans; dashboard sourced from `EvalRun`, not mocks.
- **Security:** full adversarial suite now runs automatically as a CI/eval regression gate.
- **Documentation:** `EVALUATIONS.md` full methodology, `EXPERIMENT_REGISTRY.md` finalized schema.
- **Entry dependencies:** Phase 9.
- **Exit criteria:** adversarial suite runs automatically and is green (or documented known-failures with tracked follow-ups); judge/human agreement reported against a stated threshold.

### Phase 11 — Caching layer
- **Objective:** Redis-backed exact/semantic/retrieval/embedding/reranker/tool/prefix caches with authorization-scoped keys, TTL/event invalidation, cache telemetry.
- **User-visible result:** Measurably lower latency/cost on repeated/shared questions, visible in the admin cost dashboard.
- **Affected components:** `packages/telemetry` cache instrumentation, `services/retrieval` and `services/agent-orchestration` cache lookups.
- **Schema changes:** none new (uses existing `cacheStatus` field).
- **Tests:** **mandatory cross-user cache-leakage adversarial test**, staleness/invalidation test (doc-version bump invalidates dependents), TTL expiry test.
- **Evaluation evidence:** before/after latency and cost comparison on a repeated-question workload slice.
- **Observability:** cache hit/miss/tier metrics in Grafana.
- **Security:** leakage test is the primary artifact — zero-tolerance result reported explicitly.
- **Documentation:** caching section in `RAG_INGESTION.md`, `SECURITY.md` cache-boundary section.
- **Entry dependencies:** Phase 10 (need the eval harness to measure before/after credibly).
- **Exit criteria:** zero cross-user leakage across adversarial trials; documented latency/cost improvement (or honest negative result) with a reproducible command and raw results.

### Phase 12 — ADK diagnostic specialist integration
- **Objective:** Add the bounded, in-process ADK specialist subgraph for diagnostic/triage reasoning, wired into `decide_path`.
- **User-visible result:** Qualifying multi-symptom tickets show a structured diagnostic breakdown, distinctly attributed to the specialist in the trace.
- **Affected components:** `services/agent-orchestration` (new adk-specialist module), routing logic.
- **Schema changes:** none required beyond adding a specialist-identifier span attribute.
- **Tests:** specialist output schema-validity tests, **integration test proving no network boundary is crossed** and control correctly returns to LangGraph state.
- **Evaluation evidence:** golden diagnostic-case subset comparing specialist path vs. baseline single-pass on task-completion/correctness.
- **Observability:** specialist-attributed spans distinguishable in admin trace view.
- **Security:** specialist goes through the same authz/allowlist boundary as any other node — no bypass.
- **Documentation:** ADR for the ADK boundary decision; A2A evaluated-and-deferred note finalized.
- **Entry dependencies:** Phase 9, Phase 10.
- **Exit criteria:** measured task-completion/correctness delta reported (even if modest); code/trace review confirms ADK never becomes a second orchestrator — all invocations originate from one node call site.

### Phase 13 — Performance/load testing & manual-vs-AI benchmark
- **Objective:** Replayable workload generator (1,000+ daily requests, realistic arrivals/bursts/concurrency), run local/paid-provider/offline-eval-throughput/interactive tests separately, run the manual-lookup benchmark protocol.
- **User-visible result:** Admin performance dashboard shows real measured p50/p95/p99, TTFT, cache hit rate, cost/request under realistic load; a written manual-vs-AI benchmark report.
- **Affected components:** `scripts` load-test runner, `eval/workloads`, `apps/web` admin performance dashboard.
- **Schema changes:** possibly a `LoadTestRun` manifest table for reproducibility metadata.
- **Tests:** load-harness correctness test (arrival-pattern simulation validated).
- **Evaluation evidence:** primary output of this phase — raw results stored, comparison against plan.md's target claims, honest reporting of misses.
- **Observability:** dashboard exercised under real load, validated not to fall over under its own instrumentation overhead.
- **Security:** rate-limit behavior validated under burst load (doesn't fail open).
- **Documentation:** `EXPERIMENT_REGISTRY.md` load-test protocol, dedicated benchmark report.
- **Entry dependencies:** Phase 11 (caching should exist before claiming latency numbers), Phase 12 included.
- **Exit criteria:** reproducible load-test command re-runnable; benchmark report published with defined sampling/statistics/limitations; each plan.md Outcomes-section target evaluated against actual measured numbers, met or honestly not met.

### Phase 14 — Admin console polish & full frontend IA
- **Objective:** Apply the chosen visual direction (three-pane workspace, waterfall trace view, purpose-built dashboards) across all role surfaces; finalize accessibility/reduced-motion/keyboard/dense-display requirements.
- **User-visible result:** Complete role-aware product across all four surfaces.
- **Affected components:** `apps/web` across all routes, design-system tokens.
- **Schema changes:** none.
- **Tests:** accessibility pass, reduced-motion behavior test, keyboard-navigation test, virtualization/bulk-multi-select test at seeded volume.
- **Evaluation evidence:** regression check confirming telemetry views are wired to owned schema, not mocks.
- **Observability:** n/a beyond existing.
- **Security:** admin surface authz-gated per route, verified.
- **Documentation:** `DESIGN_SYSTEM.md` finalized, README screenshots updated.
- **Entry dependencies:** Phase 13.
- **Exit criteria:** all four role surfaces implemented and authz-tested; accessibility checks pass; no mock/static charts remain in the admin console.

### Phase 15 (optional, separately gated) — AWS production mapping
- **Objective:** Document, and only if separately approved, implement the ECS/Fargate/RDS/ElastiCache/S3/Secrets-Manager mapping.
- **User-visible result:** none locally; a deployment-mapping document / optional IaC.
- **Entry dependencies:** all prior phases, plus explicit separate approval.
- **Exit criteria:** mapping documented; no cloud resources created without that separate approval.

### Cross-cutting risks and mitigations
- **LangGraph resume re-execution caveat** — mitigated structurally by placing non-idempotent work after `interrupt()` and restart-testing it explicitly (Phase 9).
- **ADK scope creep into orchestration** — mitigated by the single-node call-site rule, verified by code review each time the specialist changes (Phase 12).
- **Langfuse/ClickHouse acquisition risk** — mitigated by owning the normalized schema independently (Phase 4 onward); monitored, not blocking.
- **Documentation drift (Project WIN's failure mode)** — mitigated by the README/CLAUDE.md/PROGRESS.md separation defined in §13, enforced as a required step in every phase's "documentation updates."
- **Synthetic-data honesty** — every evaluation/telemetry/load-test artifact is labeled synthetic; no phase claims real production traffic.

---

**End of plan. Presented for approval — no implementation, scaffolding, or dependency installation has occurred or will occur until approved.**

### Critical Files for Implementation
Once approved, these are the first files/modules to create (none exist yet — this is the proposed starting scaffold):
- `C:\Dev\ResolveGrid\infra\docker-compose.yml`
- `C:\Dev\ResolveGrid\packages\contracts\` (tool/event/eval-case schema definitions)
- `C:\Dev\ResolveGrid\packages\authz\` (centralized authorization module)
- `C:\Dev\ResolveGrid\services\agent-orchestration\` (LangGraph graph/node definitions)
- `C:\Dev\ResolveGrid\docs\PROGRESS.md` (the single progress ledger)
