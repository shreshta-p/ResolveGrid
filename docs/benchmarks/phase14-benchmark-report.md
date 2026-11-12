# Phase 14 Benchmark Report

**Primary deliverable of Phase 14** (`docs/superpowers/plans/2026-10-29-phase14-performance-load-testing.md`,
Task 6). This document is the single place that walks through all five of
`plan.md`'s Outcomes-section headline claims, one at a time, with: a precise
definition, the real protocol used, the reproducible command, a pointer to
the raw results, the measured number, a verdict (**met** / **partially met**
/ **not met** / **unmeasured**), and honestly-stated limitations.

**Standing rule governing every number in this report** (unchanged since
Phase 11): "nothing mocked, this is going to be a properly working project."
Every figure below comes from a real, locally running Docker Compose stack
(Postgres, Redis, Ollama, LiteLLM) under real synthetic-but-genuinely-executed
load, never fabricated, back-calculated, or rounded favorably. **Of the five
claims, only one (claim 1) was cleanly met.** Two (claims 3 and 4) were
measured and clearly not met. One (claim 5) is mixed — cost met, latency not
met and reversed. One (claim 2) could not be measured at all in this
environment and is reported as unmeasured, not assumed. This report does not
soften any of that.

A real, upfront methodological constraint applies to claims 2-4 and is
repeated here because it shapes how every verdict below should be read:
no prior phase recorded a deliberate pre-RAG/pre-caching/pre-ADK "v0"
baseline as a formal experiment — this project was built incrementally, each
phase measured against its own immediately-prior state. This phase cannot
retroactively manufacture a baseline that was never run. The defensible
substitute used throughout (claims 1, 3, 4, 5): a **reconstructed baseline**
condition — disable the real caching/reranking capabilities Phases 9-12
added via real config toggles, exclude the ADK diagnostic specialist (Phase
13) from the baseline workload — compared against the **current full
system**, same dataset, same seed, same model, paired. This is a real,
reproducible, honestly-labeled experiment (`PLAN_APPROVED.md` §12 explicitly
permits it: "only the variable under test changes"), but it is **not** a
reconstruction of the pre-AI/manual process plan.md's original claims likely
had in mind. See each claim's "Limitations" section for what this does and
does not prove.

---

## Claim 1 — "Tested at a workload equivalent to 1,000+ requests/day"

**Verdict: MET.**

- **Definition**: the load-test harness's configured NHPP (non-homogeneous
  Poisson process) arrival schedule must sample a mean arrival rate
  equivalent to 1,000+ requests/day, and the real stack must actually
  complete that many real HTTP requests during a real run — not just
  configure the rate.
- **Protocol**: Task 1's `scripts/run_phase14_load_test.py` NHPP sampler
  (fixed seed 42, Lewis-Shedler thinning), configured with
  `duration_seconds=86400, base_rate_rps=0.005, peak_rate_rps=0.03,
  burst_schedule=(8h,13h,20h)`, compressed 48x into a ~30-minute real
  wall-clock window (documented compression factor, chosen for practicality,
  not to inflate throughput — the harness's bottleneck is real local-Ollama
  service capacity, not the nominal schedule).
- **Reproducible command**: `uv run --package resolvegrid-api python -m
  resolvegrid_api.run_phase14_load_test --condition full_system --seed 42`
  (and `--condition baseline` for the paired run).
- **Raw results**: `eval/results/phase14_load_v1/full_system_raw.json`,
  `baseline_raw.json`; manifest rows `LoadTestRun.id = 16, 17`.
- **Measured result**: the fixed-seed schedule sampled **1,151 total
  arrivals** over the simulated 24h day (mean rate 0.01283 req/s =
  **1,109 requests/day-equivalent**). Both real runs (full_system and
  baseline) **actually fired and received responses for all 1,151 scheduled
  requests** against the real live stack — not a subset, not a dry run.
- **Limitations**: this is a single-process, single-machine generator run
  from the dev host, not a distributed load-test; "requests/day" here is a
  configured-and-executed equivalent rate under 48x time compression, not a
  genuine 24-hour production day. Both are disclosed explicitly in
  `eval/results/phase14_load_v1/comparison_report.md` §1 rather than
  presented as literal production traffic.

---

## Claim 2 — "~80% reduction in manual information-gathering time"

**Verdict: UNMEASURED.** (Not met, not partially met — genuinely could not
be assessed.)

- **Definition**: the wall-clock time for a human to manually resolve a
  real ticket (no AI assist, using only the existing ticketing UI and KB
  search) versus the real measured wall-clock time of
  `/tickets/{id}/diagnose` on the same ticket, same sample.
- **Protocol**: designed in full per `PLAN_APPROVED.md` §12 — defined
  sampling (5 real tickets materialized from this project's own existing
  golden diagnostic-eval fixtures, IDs 1457-1461), start/stop rules (no
  `/chat`, no "Diagnose," no "Summarize" during the timed manual window),
  correctness thresholds, median/percentile completion time methodology,
  failure treatment, documented bias and limitations. Full protocol:
  `eval/results/phase14_manual_benchmark_v1/protocol.md`.
- **Reproducible command**: none — no run was completed to reproduce.
- **Raw results pointer**: `eval/results/phase14_manual_benchmark_v1/`
  (`protocol.md`, `kb_export_for_manual_lookup.txt`,
  `comparison_report.md` — the last documents the abandonment, not results).
- **Measured result**: none. The human operator began the timed session,
  reached ticket 1458 (a fabricated "coffee machine" ticket), accidentally
  clicked an AI-assisted "Summarize" button (contaminating that attempt
  under the protocol's own rules), and on reflection identified the real
  underlying problem: **a human cannot meaningfully "resolve" a
  synthetic/fabricated ticket the way they would resolve a real one** —
  there is no genuine stake, no real requester, no real-world context. No
  valid elapsed time was recorded for any of the 5 tickets before the
  session was deliberately stopped.
- **Limitations**: this is a real, substantive constraint of this project's
  synthetic-data-only environment (`docs/SECURITY.md`'s "not real data"
  disclosure applies here too), not a reflection on the operator or a sign
  the protocol was poorly designed. It is a *more specific* finding than the
  protocol's own already-flagged "no human subject available in an
  agent-only environment" constraint — it shows that even a willing human
  does not solve the underlying problem, because the ticket corpus itself is
  fabricated. A future phase with access to a real (non-synthetic) ticket
  corpus with genuine task stakes could re-attempt this using the same kept
  protocol. No number is estimated or substituted here in its place.

---

## Claim 3 — "End-to-end task-completion accuracy improving from ~60% to ~85%"

**Verdict: NOT MET.**

- **Definition**: `task_completion_accuracy(run)` — defined in
  `docs/EVALUATIONS.md`'s "Phase 14 Task 4" section **before** either
  comparison run was executed — is the unweighted pooled pass rate over
  `EvalCaseResult` rows in the `chat`, `retrieval`, `tool`, and `approval`
  dimensions for a given `EvalRun`, excluding `adversarial` (a documented
  structural pass-through) and `diagnostic` (no honest "disable ADK"
  baseline exists — the specialist didn't exist before Phase 13 at all).
  Unweighted pooling was chosen specifically because it is the one
  construction that cannot be tuned post hoc to hit a target number.
- **Protocol**: the same full_system/baseline toggle pair from claim 1,
  applied to the existing golden eval suite (`eval/golden/v2/*.jsonl` +
  `eval/adversarial/v1.jsonl`, 49 cases, same dataset version, same judge
  model `qwen3:14b`, same seed), run as two real `EvalRun`s via the
  existing Phase 10 `eval_worker.py`.
- **Reproducible command**: `uv run --package resolvegrid-api python -m
  resolvegrid_api.eval_worker` (run once per condition, each its own fresh
  process so the module-level env-var reads take effect).
- **Raw results**: `eval/results/phase14_eval_v1/baseline_raw.json`,
  `full_system_raw.json`; `EvalRun.id = 133` (baseline), `134`
  (full-system).
- **Measured result**: composite `task_completion_accuracy`: baseline
  **36/39 = 92.3%** -> full-system **38/39 = 97.4%**. Both numbers sit far
  above the claimed 60%/85% reference points in absolute terms.
- **Limitations — read before trusting the headline numbers**: tracing
  each dimension's real execution path (done during Task 4, not assumed)
  found that **36 of the 39 composite cases are mechanically incapable of
  being moved by the toggle under test at all.** The `chat` dimension (18
  cases) runs through `classify_intent`, a graph node that never consumes
  retrieval output — its grading depends only on `input_text`, independent
  of caching/reranking. The `retrieval` dimension (18 cases) runs through a
  separate, older Phase 7 module (`eval_retrieval.py`) that never imports
  the real `agent_retrieval.py` production path at all, and always scores
  the un-reranked fused order regardless of
  `RESOLVEGRID_RERANK_ENABLED`/`RESOLVEGRID_RETRIEVAL_CACHE_ENABLED`. Only
  `tool` (1 case) and `approval` (2 cases) are wired through the real
  toggled code path — and both showed **zero change** between conditions.
  The entire observed 2.6-point gap is in `chat`: two genuinely ambiguous
  inputs (`"ticket"`, `"asdf1234 blah blah nothing useful here"`) were
  classified differently between runs by the same real LLM making a
  different real call-to-call judgment — sampling variance on a dimension
  that cannot be affected by the variable under test, not a causal effect
  of caching or reranking. **The honest conclusion: this comparison
  measured zero causally-attributable difference between conditions on this
  golden suite.** This also does not refute plan.md's original claim about
  going from a pre-AI/manual process to this system — that comparison was
  never possible to run honestly here (no frozen "v0" baseline was ever
  recorded) and nothing in this measurement speaks to it either way; it
  only shows the specific Phase 9-13 capabilities this phase could honestly
  toggle don't measurably change golden-eval correctness versus having them
  off, consistent with caching/reranking being latency/relevance
  optimizations layered on an already-correct pipeline by design, not
  correctness-changing features.

---

## Claim 4 — "Relative evaluation pass-rate improvement of ~50% (e.g. 56% → 84%)"

**Verdict: NOT MET.**

- **Definition**: relative improvement = `(full_system_rate -
  baseline_rate) / baseline_rate`, using the same composite
  `task_completion_accuracy` as claim 3.
- **Protocol / reproducible command / raw results**: identical to claim 3
  — same two `EvalRun`s (133, 134), same single comparison.
- **Measured result**: `(0.9744 - 0.9231) / 0.9231 = 5.6%` relative
  improvement — about 1/9th of the ~50% target.
- **Limitations**: identical caveat to claim 3 applies with full force —
  this 5.6% is driven entirely by two ambiguous-input sampling-variance
  cases in a dimension (`chat`) structurally incapable of reflecting the
  toggle under test; the 3 cases that genuinely do exercise the toggled
  code path showed no change at all. `docs/EVALUATIONS.md`'s Task 4
  section, written *before* either run executed, already predicted that
  "a small or even zero measured delta here would not be surprising" given
  the reconstructed baseline is cache/rerank-off on an already-fully-
  agentic system, not a pre-AI starting point — this prediction held. What
  was not predicted in advance was the specific mechanical finding above
  (two of four composite dimensions cannot be moved by this toggle at all)
  — a real, previously-undocumented gap in this harness's coverage of its
  own stated before/after variable, worth a dedicated fix in a future phase
  (e.g. making `eval_retrieval.py`'s retrieval dimension route through the
  real `agent_retrieval.py` production path).

**A related, genuinely concerning finding surfaced by the same Task 4 run,
independent of claims 3/4's scope**: the ADK diagnostic specialist's own
golden suite (full-system only — no honest baseline exists, since the
specialist didn't exist before Phase 13) scored **safety 6/6, quality
0/6** — every one of 6 real diagnosis runs produced a response the judge
correctly failed on content grounds (3/6 defaulted to a VPN-themed answer
regardless of the ticket's actual subject; 2/6 called the wrong tool
entirely). This reproduces, more severely, the same pre-existing,
already-documented `qwen3:14b` content-grounding weakness Phase 13's own
ADR 0005 first found (2/6) — a third independent real data point in the
same direction, not a new regression. Safety held perfectly across all
three runs; quality did not. See `eval/results/phase14_eval_v1/comparison_report.md`
§3 for the full case-by-case trace.

---

## Claim 5 — "~40% lower p95 latency and ~30% lower inference cost"

**Verdict: MIXED — cost component met (on the real measurable proxy); the
literal dollar-cost claim is not applicable here; latency component NOT
MET and reversed.**

- **Definition**: p95 latency and inference cost per request, measured
  server-side from real `AgentRun`/`Span`/`ModelCall` Postgres rows (cross-
  checked against the harness's own client-side timers), full_system vs.
  baseline, same paired protocol as claim 1.
- **Protocol / reproducible command**: identical to claim 1
  (`run_phase14_load_test.py`, `--condition full_system` / `--condition
  baseline`, seed 42).
- **Raw results**: `eval/results/phase14_load_v1/full_system_server_metrics.json`,
  `baseline_server_metrics.json`, `full_system_raw.json`, `baseline_raw.json`.

### Cost

- **Measured result**: real dollar cost was **$0.00 in both conditions**
  — 100% of successful LLM calls routed to local Ollama
  (`provider=ollama, model=local-qwen3`), which this project correctly
  treats as $0/token (self-hosted, not metered). The real cloud-fallback
  path (Phase 5) *was* genuinely exercised under this load (36 attempts in
  full_system, 84 in baseline) but **100% of those attempts failed** in
  both conditions — consistent with no outbound network path to Anthropic's
  API from this sandboxed dev environment, not an application bug. **This
  means the literal "~30% lower dollar cost" claim cannot be evaluated
  here (0 vs. 0 is not a percentage).** Using real token count as the
  measurable substitute cost proxy (per `PLAN_APPROVED.md` §12's own
  "tokens" metric): full-system used **716.8 tokens/chat-request** vs.
  baseline's **1,339.8 tokens/chat-request** — a **46.5% reduction**,
  exceeding the ~30% target, directly attributable to the measured 35.3%
  real exact-cache hit rate (0% semantic) — roughly one in three chat
  requests skipped the real LLM call entirely.
- **Limitations**: this proxy result is real and exceeds the target, but
  it answers a different question than the original claim literally asked
  ("lower dollar cost") because this environment has no metered inference
  path that actually fires successfully. The embedding cache was not
  toggled off in baseline (only response/semantic/retrieval-result caches
  and the reranker were), and Redis was not flushed between runs, so
  baseline's embedding layer specifically started warm — disclosed in
  `eval/results/phase14_load_v1/comparison_report.md` §1 as a real reason
  baseline is not a pristine cold-cache baseline for that one layer,
  though it does not affect the token-count finding (every baseline chat
  call still ran a full real LLM generation with zero response/semantic/
  retrieval-result shortcuts, confirmed directly).

### Latency

- **Measured result**: full-system chat p95 was **45.58s server-side /
  48.40s client-side**, versus baseline chat p95 of **27.10s server-side
  / 27.11s client-side** — full-system was **~68% higher**, not ~40%
  lower. **NOT MET, and reversed from the claimed direction.**
- **Root cause (evidenced, not guessed)**: two real, measured, non-caching
  causes. (1) **Resource contention**: full-system's realistic workload mix
  includes `diagnose` (164 real ADK multi-turn calls, p50=45.3s,
  p99=141.4s) — absent from baseline by design. Both kinds share the same
  bounded-concurrency pool (max 6) against the *same single local Ollama
  GPU instance*, so full-system's chat requests genuinely queue behind
  much longer-running diagnose calls — real head-of-line blocking, not a
  caching artifact. (2) **Real reranker compute cost**: the `retrieve`
  span shows full_system p99=1,803ms vs. baseline p99=92ms — baseline's
  reranker is disabled (free pseudo-score fallback), while full_system
  pays real reranker-model inference cost on every retrieval-cache miss
  (smaller effect than cause 1, but real and measured).
- **Limitations**: neither cause is a defect in the caching layer itself —
  the caching layer demonstrably works (35.3% real hit rate, 46.5% fewer
  tokens, zero measurable correctness regression per claim 3/4). The
  finding is that plan.md's claim 5 framing confounds two independent
  effects: it credits "caching + reranking" as a single bundle with a
  latency improvement, but the realistic workload used to test it also
  introduces a new, expensive, GPU-contending capability (the ADK
  diagnostic specialist, Phase 13) only in the condition being measured
  for latency — and that capability's queuing cost dominates any caching
  benefit. A cleaner future measurement would isolate caching/reranking's
  latency effect from an identical workload mix (i.e. also exclude
  `diagnose` from the full_system latency comparison, not just from
  baseline), which this run did not do and discloses as a real scope gap,
  not a hidden one.

---

## Cross-cutting findings (not tied to one claim)

- **Rate limiting validated not to fail open under real burst load**
  (`PLAN_APPROVED.md` §12 security requirement): a direct, focused
  verification (13 real rapid-fire `POST /tickets` calls from one real
  employee) showed requests 1-10 → `201`, 11-13 → `429` — the per-employee
  sliding-window limiter correctly rejects once its density condition is
  met. **Real gap found and disclosed**: `check_ticket_creation_rate_limit`
  is wired only to `POST /tickets` — `/chat` and `/tickets/{id}/diagnose`,
  the two most expensive LLM-bound endpoints (and the ones that produced
  real transport errors/timeouts under this load), have no rate limiting
  at all. Under this run's realistic workload shape no single employee's
  traffic ever clustered densely enough to exercise this gap into an actual
  failure, but it is a real, unmitigated exposure, not a hypothetical one.
- **Ollama cold-start cost is severe**: the very first probe call after an
  idle period took 47-65s; warm back-to-back calls ran 7-25s. Any future
  deployment needs a keep-warm strategy before latency numbers here are
  production-representative.
- **The paid-provider load-test category is not applicable in this
  local-first environment** — LiteLLM here only routes to local Ollama by
  default; the real cloud-fallback path exists (Phase 4/5) but every real
  attempt during this phase's load failed (no outbound network from this
  sandboxed environment), stated explicitly rather than silently omitted,
  per `PLAN_APPROVED.md` §12's requirement to measure local/paid-provider/
  offline-eval/interactive categories separately.
- **The ADK diagnostic specialist is measurably the single most expensive
  real capability in the system** (6,346 tokens/call average, 45-141s
  latency) and currently has zero rate limiting — the same gap noted above.

## Summary table

| Claim | Target | Measured | Verdict |
|---|---|---|---|
| 1. Workload scale | 1,000+ req/day | 1,109 req/day-equivalent, 1,151/1,151 requests completed | **MET** |
| 2. Manual-lookup time reduction | ~80% | no valid human-timing data collected | **UNMEASURED** |
| 3. Task-completion accuracy | ~60% → ~85% | 92.3% → 97.4% (not causally attributable to the toggle under test) | **NOT MET** |
| 4. Relative eval pass-rate improvement | ~50% | 5.6% | **NOT MET** |
| 5. p95 latency | ~40% lower | 68% **higher** (reversed) | **NOT MET** |
| 5. Inference cost | ~30% lower | $0 vs $0 (not applicable); token proxy: 46.5% fewer tokens/chat-request | **MET on proxy only; dollar claim not applicable** |

## Source documents

- Task 1 (harness + schema): commit `359c9a0`
- Task 2 (load run, claims 1/5): commit `d46ce65`,
  `eval/results/phase14_load_v1/comparison_report.md`
- Task 3 (admin performance dashboard): commit `61d819f`
- Task 4 (eval run, claims 3/4): commit `74f80f1`,
  `eval/results/phase14_eval_v1/comparison_report.md`
- Task 5 (manual-vs-AI protocol, claim 2): commit `27528f3`,
  `eval/results/phase14_manual_benchmark_v1/comparison_report.md`
- Metric definitions: `docs/EVALUATIONS.md`'s "Phase 14 Task 4" section
- Registry entry: `docs/EXPERIMENT_REGISTRY.md`'s "Phase 14 Tasks 2/4" entry
- Ledger row: `docs/PROGRESS.md`'s Phase 14 row
