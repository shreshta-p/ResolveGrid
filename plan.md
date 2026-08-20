# ResolveGrid — Claude Plan Mode Master Prompt

You are the principal architect and technical lead for a greenfield, production-shaped enterprise AI platform. You are currently in **Plan Mode**.

Your task in this session is to investigate, challenge assumptions, research where necessary, and produce a complete dependency-ordered implementation plan. **Do not write application code, scaffold the repository, install dependencies, or begin implementation until I approve the plan.** Do not include timelines, dates, weeks, story points, or delivery-duration estimates.

## Objective

Design and eventually build a realistic internal IT service-management and AI operations platform for a fictional technology company with approximately 1,000 employees. Propose a credible company name and product name before propagating them through the design; `ResolveGrid` is a candidate, not a mandatory choice.

This is not a shallow resume demo. It must behave like a coherent enterprise product with:

- A realistic organizational and managerial hierarchy
- Consistent employees, departments, teams, locations, permissions, analysts, and approvers
- Devices, software entitlements, service catalog items, incidents, and service requests
- Historical and active support tickets
- Versioned policies, runbooks, and knowledge articles with provenance
- Observable, testable AI agents and persistent HITL approvals
- Reproducible evaluations, traces, costs, retrieval evidence, citations, and performance experiments

Initial scope is internal IT support. Do not expand v1 into HR or Finance. Preserve clean extension boundaries without implementing speculative departments.

## Outcomes and evidence

Design the system to test claims such as:

- A Python/FastAPI, LangGraph, Google ADK, PostgreSQL/pgvector agentic RAG system tested at a workload equivalent to 1,000+ requests/day
- Approximately 80% reduction in manual information-gathering time
- End-to-end task-completion accuracy improving from roughly 60% to 85%
- Relative evaluation pass-rate improvement of roughly 50% (for example 56% to 84%)
- Approximately 40% lower p95 latency and 30% lower inference cost

These are targets, not facts. Never fabricate them or label synthetic load as real production traffic. Each metric requires a precise definition, baseline, protocol, reproducible command, stored raw results, comparison report, and limitations. Report actual outcomes even if targets are missed.

## Users and product surfaces

Design role-aware experiences for:

1. **Employee:** ask cited IT questions, create and track requests, provide missing information, and see only authorized personal data.
2. **IT analyst:** work queues, examine evidence and similar incidents, use agent-assisted diagnosis, approve or modify actions, invoke authorized tools, escalate/resolve tickets, and submit feedback.
3. **Approver:** inspect risk, evidence, affected resources, parameters, and rollback guidance; approve, reject, or request changes with durable audit records.
4. **AI/platform admin:** inspect traces, workflow transitions, retrieval candidates and scores, citations, model/tool calls, prompt versions, evaluations, latency, token use, query-level cost, cache behavior, routing decisions, failures, and experiment comparisons.

A limited super-admin may manage identities and authentication but should not become another large UI surface. RBAC and resource authorization must be enforced by backend and tool execution, not merely hidden in the UI. Agents inherit the requesting principal's permissions.

## Local-first runtime

The complete core system must run locally without paid enterprise services. Use Docker Compose or an equally reproducible approach. Plan local PostgreSQL with pgvector, Redis, Next.js, FastAPI, workers, Ollama, telemetry infrastructure, and object/document storage if needed.

The machine has an RTX 4080 with 12 GB VRAM and 32 GB RAM. Select Ollama models through current compatibility research and benchmarks; do not hard-code an arbitrary model.

Provide an optional AWS production mapping (for example ECS/Fargate, RDS, ElastiCache, S3, managed secrets and telemetry), but do not require cloud resources or generate deployment infrastructure unless separately approved.

## Required stack and responsibility boundaries

The intended stack includes Next.js/TypeScript, Python/FastAPI, PostgreSQL/pgvector, Redis, LangGraph, Google ADK, LiteLLM, Ollama, OpenAI, Anthropic, Gemini, and A2A where justified.

Do not use technologies as resume keywords. Give each one a single defensible responsibility. A preferred split is:

- LangGraph: application workflow, state, checkpoints, retries, routing, resumability, and HITL interrupts
- Google ADK: selected specialist-agent/tool packaging where it adds value
- LiteLLM: model gateway, provider abstraction, budgets, normalized usage, fallbacks, and routing hooks
- Ollama: zero-marginal-cost local inference
- A2A: only between independently meaningful agent/service boundaries

Challenge this split if official documentation shows a better design. Do not let LangGraph and ADK own the same orchestration layer. Do not turn ordinary function calls into A2A traffic. Research current primary documentation before fixing APIs, packages, versions, or model names.

## Model routing and cost accounting

Initially use Ollama, one primary cloud provider, and one fallback. Expand routing only after baseline measurements. Support quality-, latency-, privacy-, reliability-, and cost-oriented policies.

Every inference call must persist trace/span ID, protected request/user identity, provider, model, purpose, prompt/workflow version, input/output/cached tokens, latency, retries, cache status, routing and fallback reason, error, timestamp, provider cost, and locally estimated cost. Ollama calls display `$0.00` monetary inference cost while retaining latency and token information. Version pricing so historical costs do not change when current prices change.

## Local operational systems and tools

Do not depend on paid Jira, ServiceNow, or Okta accounts. Build realistic persistent local modules for ticketing, identity/account status, employee directory, assets, entitlements, service health, knowledge, approvals, notifications, and auditing. Add vendor-neutral adapters for future integrations.

Agents interact through typed tool contracts, not arbitrary table access. Tools require authorization, validation, idempotency where needed, timeouts, retry policy, audit events, error taxonomy, safe dry-run support, and compensation/rollback guidance for sensitive mutations.

## Agent workflow

Design a measurable baseline and progressively enhanced workflow covering:

1. Authentication and requesting principal
2. Intent, risk, and data-scope classification
3. Retrieval/tool decision
4. Query rewrite or decomposition when justified
5. Authorization-aware metadata filters
6. Candidate retrieval, fusion, reranking, and deduplication
7. Evidence-sufficiency decision
8. Tool selection and schema validation
9. Read-only execution or persistent approval interrupt
10. Authorized mutation after approval
11. Structured response and citation verification
12. Safe answer, abstention, or escalation
13. Telemetry, evaluation hooks, and feedback

An approval must survive restarts and bind to the exact action, parameters, actor, evidence, risk context, and expiry. Do not expose private chain-of-thought. Store inspectable state transitions, evidence, decision summaries, tool inputs/outputs, and policy results.

## RAG design

Begin with the simplest strong measurable baseline:

- Structure-aware parsing and chunking
- Embeddings and pgvector ANN search
- PostgreSQL lexical/full-text search or a better justified alternative
- Hybrid retrieval with documented fusion
- Authorization and metadata filtering
- Reranking and deduplication
- Context budgeting and citation mapping
- Document, parser, chunk, and embedding versioning

Plan experiments for HNSW versus IVFFlat where dataset scale justifies it, recall/latency trade-offs, filter selectivity, chunk sizing, parent-child retrieval, query expansion, neighboring chunks, reranker choice, freshness, conflicting documents, effective dates, and access-control metadata.

Do not add Self-RAG or CRAG by default. First define measurable triggers such as weak retrieval confidence, conflicting sources, missing evidence, failed citation verification, or high-risk requests. Add corrective loops only if evaluations justify their latency and cost. Bound all loops and escalate safely.

## Caching

Consider exact-response, semantic-response, retrieval, embedding, reranker, safe read-only tool, and provider prompt-prefix caches. Cache keys and validation must account for authorization scope, user/role/department, tenant, knowledge version, permissions, workflow/prompt/model version, locale, and freshness.

Support globally shareable, role-scoped, department-scoped, user-specific, and non-cacheable results. Define TTLs, event invalidation, stale handling, provenance, and cache telemetry. Treat cache authorization as a security boundary and test cross-user leakage.

## Knowledge and enterprise data

Use two clearly labeled sources:

1. **Public attributable knowledge:** legally usable official vendor documentation, operating-system support material, and public security/IT guidance. Record URL, publisher, retrieval date, version/effective date, checksum, license/usage note where identifiable, parser version, chunking version, and embedding version. Do not scrape arbitrary sites without checking terms and reproducibility.
2. **Synthetic private company knowledge:** consistent internal IT/security policies, runbooks, software catalog, access rules, onboarding/offboarding, VPN/network guidance, service levels, approval matrices, known errors, change notices, and archived/superseded documents. Include intentionally stale, conflicting, restricted, and incomplete cases.

Generate about 1,000 coherent employees with deterministic seeds and scenario manifests. Model reporting lines, departments, teams, locations, time zones, employment status, IT roles, approval authority, devices, entitlements, access groups, and lifecycle dates. Generate connected ticket histories covering common and long-tail issues, duplicates, multi-turn cases, escalations, SLA breaches, approvals, outages, missing information, security-sensitive requests, tool failures, and reopened tickets. Use no real PII.

## Golden dataset and evaluations

Use curated synthetic cases and legally usable public-source questions. Do not let one model create unreviewed gold answers and then judge itself against them.

Cases should version identity/permissions, request/context, intent, answerability, relevant evidence, citations, tool and argument constraints, state transition, approval policy, forbidden actions, structured result, rubric, risk/difficulty, provenance, and human-review status.

Prefer deterministic graders before model judges. Measure retrieval recall/precision, MRR or nDCG, reranking, context relevance, citation correctness/completeness, groundedness, faithfulness, correctness, abstention, hallucination, tool/argument accuracy, schema validity, authorization and approval compliance, task completion, latency, tokens, cost, cache safety, resilience, and prompt-injection resistance.

Include adversarial cases: injected documents, cross-user data requests, conflicting/stale policies, unsupported questions, fabricated IDs, malformed or timed-out tools, outages, duplicate/expired approvals, provider failure, and empty retrieval.

Version every run by dataset, prompt, workflow, retriever, reranker, embeddings, generation model, judge, tool schemas, and commit. Calibrate model judges with human-reviewed samples and agreement analysis.

## Observability

Use OpenTelemetry-compatible correlated traces spanning API, auth, workflow, agents, models, retrieval, reranking, caches, tools, approvals, guardrails, evaluations, response, and feedback. The admin UI must display genuine telemetry, not static mock charts.

Research current open-source options such as Langfuse, Phoenix, OpenTelemetry Collector, Prometheus, and Grafana, selecting the smallest non-duplicative combination. Own normalized trace, evaluation, and cost schemas so no vendor becomes the only source of truth. Define redaction, access control, retention, and deletion.

## Performance and metric experiments

Measure baseline p50/p95/p99 latency, TTFT, stage latency, queue time, error rate, throughput, cache hit rate, tokens, cost/request, cost/successful-task, and quality by route/model.

Simulate 1,000+ daily requests with realistic arrivals, bursts, concurrency, repeated shared questions, and personalized requests—not a constant average rate. Separate local load, paid-provider tests, offline evaluation throughput, and interactive performance. Save replayable workloads and raw evidence.

Test caching, routing, context reduction, classification, concurrency, ANN/retrieval tuning, reranking thresholds, skipped planning, prefix caching, connection pooling, streaming, batched embeddings, and batched offline evals. Do not claim online-generation batching improvements unless actually supported.

Design a reproducible manual-lookup benchmark comparing manual and AI-assisted tasks with defined sampling, start/stop rules, correctness threshold, median/percentile time, failure treatment, sample size, bias, and limitations.

## Security and guardrails

Plan input/output validation, prompt-injection boundaries, authorization-aware retrieval, tool allowlists, least privilege, approval policies, rate limits, secrets, PII redaction, audit trails, retention, safe errors, threat modeling, dependency/container scanning, and backup/restore. Test guardrails as executable policies where practical. Expected abstention or escalation counts as success.

## Frontend and product design

Use Next.js/TypeScript, an accessible component foundation, design tokens, responsive layouts, keyboard access, appropriate dense displays, and reduced-motion support.

Inspect installed skills and invoke applicable design skills through their real mechanism, specifically checking for Taste/design-taste, Impeccable, frontend-design, and emil-design-eng. Do not pretend an unavailable skill exists. Research current enterprise IT-support and AI-observability interfaces using primary/product sources where possible. Present two or three distinct directions and recommend one; do not clone another product.

The UI should feel modern, tactile, precise, premium, and operationally dense. Use restrained microinteractions to communicate streaming, workflow state, approval, tool execution, trace expansion, comparisons, cache activity, and recovery. Avoid generic AI gradients, gratuitous glass, decorative dashboards, huge empty cards, repetitive KPI grids, excessive rounding, delayed interactions, fake terminal aesthetics, and inaccessible motion.

## Project WIN reference

Inspect `C:\Dev\Project WIN` read-only. Do not modify or blindly copy it. Extract useful lessons from its manager hierarchy, role-aware IA, audit logs, semantic statuses, dense admin workflows, token discipline, and reduced-motion handling.

Avoid its documentation drift: the README stayed boilerplate while CLAUDE.md grew to roughly 25 KB.

## Cross-session continuity

Keep `CLAUDE.md` lean: durable instructions, critical commands, invariants, and links only. Do not use it as a diary or duplicate architecture.

Propose canonical documents for product scope, architecture/ADRs, data model, workflows/tools, RAG/ingestion, evaluations, telemetry/cost, security, API contracts, design system, experiment registry, runbooks, decision log, and current status.

Maintain one authoritative progress ledger with states such as Proposed, Approved, In Progress, Blocked, Implemented, Verified, and Superseded. Each future session must inspect instructions, ledger, repository, and git state; choose the next approved bounded unit; record decisions; implement and verify only that unit; update evidence and status; and leave a resumable repository. Do not duplicate status across many files.

## Engineering principles

Prefer a monorepo and clear bounded modules over premature microservices. Likely boundaries are web, FastAPI backend, agent orchestration, retrieval/ingestion, evaluation workers, operational adapters, shared contracts, and infrastructure configuration.

Plan typed/versioned contracts, migrations, deterministic seeds, background jobs, idempotency, retries/dead letters where needed, feature flags, configuration validation, health checks, structured logs, and unit/integration/contract/E2E/security/evaluation/load tests. Add queues and events only where asynchronous execution, retries, or isolation require them.

## Required planning process

Before finalizing the plan:

1. Inspect Project WIN read-only.
2. Inspect the target workspace and existing repository state.
3. Inspect installed skills and local capabilities.
4. Research current official documentation for AI libraries and integrations.
5. Research observability/evaluation options and enterprise UI patterns.
6. Identify conflicts, duplicated responsibilities, unsupported assumptions, and overengineering.
7. Propose company and product identity with rationale.
8. Present two or three architecture approaches and recommend one.
9. Ask only questions that materially alter architecture or scope.
10. Do not make me choose low-level details that evidence can resolve.

## Required plan output

Produce a plan containing:

1. Executive summary, assumptions, non-goals, and proposed identity
2. Users, workflows, domain model, and authorization
3. Recommended architecture and alternatives rejected
4. Responsibility of every technology and A2A criteria
5. Monorepo layout, local topology, and optional cloud mapping
6. Agent workflows, tools, approvals, and error handling
7. Baseline RAG, progressive enhancements, ingestion, and caching
8. Enterprise data and golden-dataset strategy
9. Evaluations, telemetry, cost accounting, and admin-console requirements
10. Security and threat model
11. Frontend IA, design research, and recommended visual direction
12. Testing, load testing, metric definitions, and before/after experiments
13. Persistent documentation and progress system
14. Dependency-ordered implementation phases with risks, mitigations, deferrals, and approval decisions

For every phase specify objective, user-visible result, affected components, schema changes, tests, evaluation evidence, observability, security, documentation updates, entry dependencies, and objective exit criteria. Include no time estimates.

The final plan must be locally runnable, enterprise-shaped, reproducible, evaluation-driven, cost-observable, security-conscious, honest about synthetic data, incremental, and explicit about deferrals. When uncertain whether to add infrastructure or another agent, prefer the simpler measurable design.

End by presenting the plan for approval. **Do not begin implementation in the same response.**
