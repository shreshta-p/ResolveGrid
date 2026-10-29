# Phase 14: Performance/Load Testing & Manual-vs-AI Benchmark

**Renumbering note**: this is the approved master plan's original Phase 13
("Performance/load testing & manual-vs-AI benchmark," `docs/PLAN_APPROVED.md`
line 508), renumbered Phase 14 per `docs/DECISION_LOG.md`'s 2026-09-28 entry.
Depends on Phase 12 (Caching, Verified) and Phase 13 (ADK diagnostic
specialist, Verified, `docs/PROGRESS.md`) — both closed. The master plan's
original Phase 15 ("AWS production mapping," renumbered Phase 16) is
**dropped from scope entirely** per the user's explicit 2026-10-28
instruction (`docs/DECISION_LOG.md`'s 2026-10-29 entry) — this phase and
Phase 15 (admin console polish, renumbered from original Phase 14) are the
final two phases of this project.

**Standing directive governing this phase (unchanged since Phase 11)**:
"nothing mocked, this is going to be a properly working project." Every
number this phase reports must come from a real running stack under real
(synthetic but genuinely executed) load — never a fabricated or
back-calculated figure. Per `plan.md`'s own Outcomes section: "These are
targets, not facts. Never fabricate them or label synthetic load as real
production traffic... Report actual outcomes even if targets are missed."
This phase's entire job is to honestly test plan.md's five headline claims
against reality, not to make them come true by construction.

## The five claims this phase must evaluate (plan.md lines 27-31)

1. Tested at a workload equivalent to 1,000+ requests/day.
2. ~80% reduction in manual information-gathering time.
3. End-to-end task-completion accuracy improving from ~60% to ~85%.
4. Relative evaluation pass-rate improvement of ~50% (e.g. 56% → 84%).
5. ~40% lower p95 latency and ~30% lower inference cost.

Each requires: a precise metric definition, a real baseline, a reproducible
protocol/command, stored raw results, a comparison report, and documented
limitations (`plan.md` line 33; `PLAN_APPROVED.md` §12's before/after
protocol: fixed dataset + fixed seed + only the variable under test
changes + paired comparison).

**A real, upfront constraint, reported honestly rather than hidden**: claims
2-4 require a "before" condition. No prior phase recorded a deliberate
pre-RAG/pre-caching/pre-ADK baseline as a formal experiment at project
start — Phases 1-13 were built incrementally, each already measured against
its own immediately-prior state (e.g. Phase 8's reranking before/after, Phase
12's caching before/after), never against a single frozen "v0" snapshot.
This phase cannot retroactively manufacture a baseline that was never run.
The defensible, honest path: reconstruct a genuine "baseline" condition by
**disabling** specific real capabilities this project added (semantic/
retrieval/response caching off, reranker off, ADK diagnostic specialist
unused — i.e. the ticket-creation + `/chat` RAG path exactly as it existed
after Phase 7/8, run fresh against the current golden set and current
model) and compare it against the **current full system**, same dataset,
same seed, same model versions, paired. This is explicitly allowed by
§12's protocol ("only the variable under test changes") — here the
variable is "which Phase 9-13 capabilities are enabled," which is a real,
reproducible, honestly-labeled experiment, not a reconstruction of history.
Claim 2 (manual-lookup time) has no automatable proxy at all — it requires
an actual human performing the task. Task 5 below designs the most
defensible real protocol achievable without a live human-timing study
available in this environment, and reports that limitation explicitly
rather than fabricating a number.

## Real research performed before writing this plan

**Load-generation approach — custom asyncio harness, not Locust/k6.**
Considered Locust and k6 (both real, current, well-documented load-testing
tools) against a small bespoke Python harness. Chose the bespoke harness
because: (a) `PLAN_APPROVED.md` requires "load-harness correctness test
(arrival-pattern simulation validated)" as a real automated test — trivial
to write as a deterministic, fixed-seed unit test against our own sampler,
much harder to validate from outside an external tool's internal
distribution code; (b) this project's results need to join directly against
real `AgentRun`/`Span`/`ModelCall` rows already in Postgres (telemetry
already owned per `ARCHITECTURE.md`, not re-derived from an external tool's
own report format); (c) every other cross-cutting harness in this repo
(`eval_retrieval.py`, `diagnostic_eval.py`, `eval_worker.py`) is already a
bespoke in-repo Python module following the same convention, not an
external tool — consistent, not novel. Locust/k6 remain real options if a
future phase needs distributed/multi-machine load generation; this phase's
"1,000+ requests/day" target is low enough (≈0.012 req/s sustained, bursty)
that a single-process asyncio generator run from the dev machine is
realistic, not a scaling cheat.

**Arrival-pattern modeling.** "Realistic arrivals/bursts/concurrency" (not a
constant rate) is implemented as a non-homogeneous Poisson process: a
real day-shaped intensity curve (e.g. low overnight, business-hours peaks)
sampled via thinning, plus explicit burst injection (a Poisson sub-process
of simultaneous arrivals) at a few scheduled points — both real, standard,
citable techniques (Lewis-Shedler thinning for NHPP simulation), not
invented. Implemented with Python's stdlib `random`/`math`, no new runtime
dependency.

**Separation of test categories, per `PLAN_APPROVED.md` §12**: "Local load,
paid-provider tests, offline eval throughput, and interactive performance
are measured **separately**." This project has no paid-provider path wired
(LiteLLM here only routes to local Ollama — confirmed via
`llm_gateway.py`'s real config; Phase 4/5's cloud-fallback path exists but
is not exercised by default) — so the "paid-provider" category is reported
as **not applicable in this local-first environment**, stated explicitly in
the final report rather than silently omitted.

## Task breakdown

### Task 1 — Load-test harness core + `LoadTestRun` schema
- Build `scripts/load_test.py` (or `apps/api/src/resolvegrid_api/load_test.py`
  if it needs direct DB/model access — decide empirically which import path
  is cleaner, matching `eval_retrieval.py`'s precedent of living inside the
  API package when it needs the same session/engine machinery): the NHPP
  arrival sampler (fixed-seed reproducible), a realistic workload mix drawn
  from real seeded tickets/employees/questions (not synthetic nonsense
  strings), a bounded-concurrency async client against the real running API.
- New Alembic migration: `LoadTestRun` manifest table (id, started_at,
  completed_at, config_json (seed, duration, target rps, burst schedule),
  git_commit, result_summary_json, raw_results_path) — for reproducibility
  metadata, per `PLAN_APPROVED.md`'s schema note.
- Test: a deterministic unit test proving the NHPP sampler's arrival-pattern
  simulation is correct (fixed seed → exact reproducible arrival sequence;
  statistical shape check — e.g. bucketed counts roughly match the intensity
  curve's shape within a documented tolerance) — this is the "load-harness
  correctness test" `PLAN_APPROVED.md` explicitly requires, and it must not
  depend on a real running stack to be meaningful as a correctness test.

### Task 2 — Real local load run against the live stack; latency/cost/throughput claims
- Run the Task 1 harness against the real running Docker Compose stack at a
  workload equivalent to 1,000+ requests/day (compressed into a realistic
  test window — state the compression factor and why it's a fair
  equivalent), hitting real endpoints (`/tickets`, `/chat`, `/tickets/{id}/
  diagnose`, mixed per a realistic usage-ratio assumption, stated
  explicitly).
- Run the SAME workload twice: once against the current full system
  (caching/reranking/ADK all enabled — the real default), once against the
  reconstructed baseline condition described above (caching/reranking
  disabled via real config, ADK path not exercised) — paired comparison,
  same seed, same dataset.
- Capture real p50/p95/p99 latency, TTFT, queue time, error rate, cache hit
  rate (per tier), throughput, tokens, cost/request, cost/successful-task —
  every metric definition from `PLAN_APPROVED.md` §12, computed from real
  `AgentRun`/`Span`/`ModelCall` rows, not the harness's own client-side
  timers alone (both should be captured and cross-checked).
- Store raw results under `eval/results/phase14_load_v1/` (baseline run and
  full-system run, both labeled, both reproducible via one command).
- Validate rate-limiting doesn't fail open under this same burst load
  (`rate_limit.py`'s real behavior, observed, not assumed) — the
  `PLAN_APPROVED.md` §12 security requirement for this phase.
- Report claims 1 and 5 (workload scale, p95 latency delta, cost delta)
  honestly — met, partially met, or not met, with real numbers either way.

### Task 3 — Admin performance dashboard (real data only)
- `apps/web/app/admin/performance`: a new route reading a new real endpoint
  that aggregates `AgentRun`/`Span`/`ModelCall` into p50/p95/p99, TTFT,
  cache hit rate, cost/request, throughput — over a selectable real time
  window, including the Task 2 load-test window as a selectable range.
- Explicit exit-criterion check (`PLAN_APPROVED.md` line 280 and this
  phase's own line 527): **no static/mock charts** — every number traced to
  a real query against owned schema. A regression test proves this (e.g.
  asserting the endpoint's response changes when a new real `AgentRun` row
  is inserted, not a fixture).
- Dashboard validated under real load from Task 2's own run (observability
  "exercised under real load, validated not to fall over under its own
  instrumentation overhead" — `PLAN_APPROVED.md`'s explicit requirement).
- Visual polish is explicitly OUT of scope here (that's Phase 15's job per
  `PLAN_APPROVED.md`'s own phase split) — this task wires real data
  correctly with the existing minimal style (`admin/evals/page.tsx`'s
  precedent), not a finished design.

### Task 4 — Evaluation pass-rate & task-completion accuracy claims (3 and 4)
- Run the existing golden eval suite (`eval/golden/*.jsonl`, all dimensions)
  against both conditions from Task 2 (baseline vs. full system), same
  dataset version, same seed, same judge — a real, paired before/after
  evaluation run, stored under `eval/results/phase14_eval_v1/`.
- Define "task-completion accuracy" precisely before measuring (this exact
  term isn't a field anywhere in the schema yet — likely composed from
  existing per-case `passed` outcomes across the `retrieval`/`tool`/
  `approval`/`mutation_completion_claim`/overall-answer-correctness
  dimensions, weighted by real task scenarios, not invented post hoc to hit
  85%). Document the exact definition in `EVALUATIONS.md` before running.
- Report claims 3 and 4 honestly against the baseline-vs-full-system
  comparison — met, partially met, or not met.

### Task 5 — Manual-vs-AI benchmark protocol + report (claim 2)
- Design the real protocol `PLAN_APPROVED.md` §12 requires: defined
  sampling (which real seeded tickets), start/stop rules, correctness
  threshold, median/percentile completion time, failure treatment, sample
  size, documented bias and limitations.
- Honest, stated constraint: no live human subject is available in this
  agent-only development environment to time genuine "manual" lookups.
  Design the most defensible real proxy achievable and label it as exactly
  that — e.g. a structured walkthrough protocol where a human (the
  project's own operator, if willing, or documented as unavailable) times
  themselves manually resolving a real sample of seeded tickets using only
  the existing ticketing UI and KB search (no AI assist), compared against
  real measured `/tickets/{id}/diagnose` wall-clock time on the same
  tickets. If no human timing session is actually feasible in this
  environment, the report states that explicitly as a real, unmet
  limitation of claim 2 — rather than estimating or fabricating a number.
  This must be decided and disclosed, not quietly dropped.
- Report claim 2 honestly, including stating directly if it could not be
  measured and why.

### Task 6 (closing) — Final benchmark report, docs, fresh-state verify, push, CI
- `EXPERIMENT_REGISTRY.md` gains the Phase 14 load-test protocol entry
  (following the exact field convention of the existing Phase 7/8 entries
  read directly from the current file) plus the baseline-vs-full-system
  before/after entries from Tasks 2 and 4.
- A single dedicated benchmark report (new doc, e.g.
  `docs/benchmarks/phase14-benchmark-report.md`) walks through all five
  `plan.md` claims one by one: definition, protocol, reproducible command,
  raw-results pointer, measured result, met/partially-met/not-met verdict,
  limitations — mirroring the honesty standard already set by Phase 12
  Task 4's prompt-caching negative result and Phase 13 Task 5's 2/6
  content-quality finding.
- `docs/PROGRESS.md` gains the Phase 14 closing row.
- Fresh-state verification (teardown/rebuild containers, full test suite,
  seed/ingest, real browser walkthrough of the new performance dashboard),
  push, blocking-foreground CI poll to green — mirroring every prior phase's
  closing-task convention (most recently Phase 13 Task 6).

## Entry dependencies
Phase 12 (Caching, Verified) and Phase 13 (ADK diagnostic specialist,
Verified) — both closed, confirmed via `docs/PROGRESS.md`.

## Exit criteria
Reproducible load-test command re-runnable; benchmark report published with
defined sampling/statistics/limitations; each of `plan.md`'s five Outcomes
claims evaluated against actual measured numbers, each explicitly met or
honestly not met; admin performance dashboard shows only real data; rate
limiting validated not to fail open under real burst load; `docs/PROGRESS.md`
and `EXPERIMENT_REGISTRY.md` updated; CI green on `master`.
