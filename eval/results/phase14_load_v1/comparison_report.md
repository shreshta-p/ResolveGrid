# Phase 14 Task 2 — Real Load-Test Comparison (full_system vs baseline)

Generated from two real runs of `scripts/run_phase14_load_test.py` against
the real, locally running Docker Compose stack (Postgres/Redis/Ollama/
LiteLLM) plus a real `uvicorn` API process started via
`scripts/run_api_dev.py`. Nothing in this document is estimated,
back-calculated, or fabricated — every number below is read directly from
either the harness's own client-side `RequestResult` timers (`*_raw.json`)
or real `AgentRun`/`Span`/`ModelCall` Postgres rows (`*_server_metrics.json`),
both captured and cross-checked against each other per
`PLAN_APPROVED.md` §12.

- `git_commit` at run time: `2402a2a` (plus this task's own uncommitted
  changes — see "Code changes made during this task" below).
- Full-system `LoadTestRun.id = 16`, baseline `LoadTestRun.id = 17`
  (real rows in the `load_test_run` table).
- Raw per-request results: `full_system_raw.json`, `baseline_raw.json`.
- Server-side query output: `full_system_server_metrics.json`,
  `baseline_server_metrics.json`.

## 1. Protocol

**Same NHPP arrival schedule, same seed (42), for both runs.**
`NHPPConfig(duration_seconds=86400, base_rate_rps=0.005, peak_rate_rps=0.03,
burst_schedule=(8h,13h,20h), burst_mean_size=20, burst_jitter_seconds=5)`
sampled **1,151 total arrivals** over a simulated 24h day — mean rate
0.01283 req/s = **1,109 requests/day-equivalent**, comfortably over the
phase's "1,000+ requests/day" target.

**Compression**: arrival times divided by **48x** (24h → 30 real minutes
nominal), preserving the exact relative arrival shape (two business-hours
peaks, overnight trough, three scheduled bursts) while being practical to
run. Documented in full in `scripts/run_phase14_load_test.py`'s module
docstring, including why the *actual* wall-clock duration was expected to
exceed the 30-minute nominal window (real local-Ollama service capacity,
not the nominal schedule, is the real bottleneck — confirmed below).

**Only variable changed between runs**: which Phase 9-13 capabilities are
enabled, via real config:

| | full_system | baseline |
|---|---|---|
| `RESOLVEGRID_RESPONSE_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_SEMANTIC_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_RETRIEVAL_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_RERANK_ENABLED` | true (default) | **false** |
| workload mix | chat 0.50 / create_ticket 0.35 / **diagnose 0.15** | chat 0.50 / create_ticket 0.35 / diagnose **excluded** |

Each condition ran against its own freshly started `uvicorn` process (the
env vars above are read once at module import, so a live process can't be
flipped mid-run) — full_system first, then baseline.

**Documented scope limitation**: the lower-level embedding cache (Phase 12
Task 1, `get_or_compute_embeddings`) was **not** toggled off in baseline —
only the response/semantic/retrieval-*result* caches and the reranker were
disabled, per the task's literal scope ("caching and reranking"). Redis was
also not flushed between runs, so baseline's embedding cache started warm
from full_system's run (`redis_keyspace_hits_delta=694, misses=0` in
baseline — i.e. **every** embedding lookup during baseline hit a
leftover-warm cache entry from full_system, since only 18 unique chat
questions exist in the real golden set). This does not affect the core
finding (every baseline chat call still ran a full real LLM generation
with zero response/semantic/retrieval-result shortcuts — confirmed
directly below), but it is a real reason baseline's measured latency is
*not* a pristine cold-cache baseline for the embedding layer specifically.

**Real seeded data**: 58 active employees, 7 queues, 5 pre-existing
tickets (grew to 13+ during the run from real `create_ticket` items), 18
unique real `/chat` golden questions (`eval/golden/v2/chat_v1.jsonl`), 6
unique real ticket descriptions (`eval/golden/diagnostic_v1.jsonl`).

**Real gap found and fixed while running this** (`load_test.py`): the
original `build_workload_mix` drew every kind's `employee_id`, including
`diagnose`, from the full active-employee pool. `/tickets/{id}/diagnose`
is `_STAFF_ONLY_ACTIONS` in `packages/authz/policy.py` — only 6 of 58
active employees hold any role grant, so ~90% of sampled diagnose calls
would have 403'd. Confirmed via a real `curl` (employee 2 → 403 "not
authorized to diagnose this ticket"). Fixed by adding
`fetch_real_diagnose_capable_employee_ids` (real global-admin employees,
ids 1473/1488) and routing diagnose items through that pool instead.

## 2. Real measured numbers

### 2.1 Client-side (harness `perf_counter` timers)

| Metric | full_system | baseline |
|---|---|---|
| Total requests fired | 1,151 | 1,151 |
| Actual wall-clock duration | 3,206.3s (53.4 min) | 2,242.6s (37.4 min) |
| Client throughput | 0.359 req/s | 0.513 req/s |
| Workload mix (actual) | chat 580 / create_ticket 407 / diagnose 164 | chat 694 / create_ticket 457 |
| Overall p50 / p95 / p99 (s) | 0.036 / 55.85 / 68.82 | 10.54 / 24.72 / 29.51 |
| Overall mean latency | 14.07s | 9.21s |
| Transport errors | 6 (all diagnose timeouts, 120s cap) | 0 |
| Non-2xx | 7 (chat → 502) | 0 |
| 429s on create_ticket | 0 | 0 |

**Chat-only** (the only kind present in both runs — the fair comparison):

| Metric | full_system chat | baseline chat |
|---|---|---|
| n | 580 | 694 |
| p50 (s) | 11.49 | 13.86 |
| p95 (s) | 48.40 | 27.11 |
| p99 (s) | 60.81 | 30.63 |
| mean (s) | 15.27 | 15.27 (coincidental match) |

**diagnose** (full_system only — not run in baseline by design): n=164,
p50=44.79s, p95=71.40s, p99=91.13s, mean=45.83s, 6 transport timeouts
(3.7% of diagnose calls — the ADK specialist's multi-turn tool-calling
loop occasionally exceeded the 120s client timeout under concurrent GPU
load).

### 2.2 Server-side (real `AgentRun`/`Span`/`ModelCall` rows, time-window queried)

| Metric | full_system (chat+diagnose, n=744) | baseline (chat only, n=696*) |
|---|---|---|
| AgentRun p50/p95/p99 (ms) | 19,576 / 60,853 / 93,890 | 13,846 / 27,102 / 30,560 |
| AgentRun status | 737 completed, 7 error | 696 completed, 0 error |
| cache_hit_rate | **35.3%** (263/744, all `exact`; 0 `semantic`) | **0.0%** (confirms toggle worked) |

*baseline n=696 includes 2 manual toggle-verification `/chat` calls fired
immediately before the real run started (both fell inside the query
window) — immaterial at this sample size, disclosed for full transparency.

**Chat-only server-side split** (queried separately by `input_text`
pattern, for the fair like-for-like comparison claim 5 actually needs):

| Metric | full_system chat (n=580) | baseline chat (n=694) |
|---|---|---|
| p50 (ms) | 10,432 | 13,846 |
| p95 (ms) | **45,578** | **27,102** |
| p99 (ms) | 60,263 | 30,560 |
| mean (ms) | 14,657 | 15,215 |

**diagnose-only server-side** (full_system only, n=164): p50=45,270ms,
p95=84,509ms, p99=141,440ms, mean=49,225ms.

**Span-level, `retrieve` stage** (isolates the retrieval+rerank pipeline
specifically): full_system p50=4ms/p95=1,493ms/p99=1,803ms/mean=459ms
(bimodal — cache hits near 0ms, cache misses pay real rerank cost) vs.
baseline p50=57ms/p95=76ms/p99=92ms/mean=60ms (uniformly fast — every call
is a real but small KB corpus search with reranking degraded to a free
pseudo-score, never the real reranker model).

### 2.3 Tokens / cost

**Real dollar cost for both conditions: $0.00.** Every successful LLM
call in both runs routed to local Ollama (`provider=ollama,
model=local-qwen3`), which this project's pricing accounting correctly
treats as $0/token (self-hosted, not metered). The real cloud-fallback
path (Phase 5) *was* genuinely exercised under this load — 36 real
attempts in full_system, 84 in baseline, all routed to
`provider=anthropic, model=cloud-primary` — but **100% of those cloud
attempts failed** (0 successes in either run; `error_count` equals `n`
for every such row). This points to no real outbound network path to
Anthropic's API from this environment (sandboxed/offline dev box), not an
application bug — client-facing chat requests still returned 200 in
nearly every case because the local Ollama path independently succeeded
in parallel/retry. **This means claim 5's "~30% lower cost" cannot be
evaluated as a dollar figure here** (0 vs 0 is not a percentage) — token
count is used below as the real, measured substitute cost proxy, per
`PLAN_APPROVED.md` §12's own "tokens" metric.

| Metric | full_system | baseline |
|---|---|---|
| Real tokens consumed, chat only | 415,733 | 929,847 |
| Tokens / chat request | **716.8** | **1,339.8** |
| Tokens / chat request, reduction | **46.5% fewer tokens/chat-request in full_system** | — |
| Real tokens consumed, diagnose only | 1,040,735 (164 calls, 6,346 tokens/call avg) | n/a (not run) |
| Real tokens consumed, ALL purposes | 1,456,468 | 929,847 |
| Tokens / total request (1,151 denom., both conditions) | 1,265.4 | 807.9 |

The "tokens / total request" row is **not** the fair comparison — it's
dominated by diagnose's real per-call cost (6,346 tokens/call) being
present in full_system and structurally absent from baseline (not a
caching effect). The "tokens / chat request" row above is the correct,
caching-isolated comparison.

### 2.4 Rate limiting (`PLAN_APPROVED.md` §12 security requirement)

- **Natural observation during both real runs**: 0 of 864 total
  `create_ticket` requests (407 + 457) hit `429`, across either run.
  `check_ticket_creation_rate_limit` (`rate_limit.py`) is a 10-req/60s
  **per-employee** sliding window; with items spread randomly across 58
  active employees by the realistic workload-mix generator, no single
  employee's `create_ticket` draws ever clustered densely enough within
  any 60s window to trip it — a real, measured consequence of this
  specific realistic arrival/employee distribution, not evidence the
  limiter is broken.
- **Direct focused verification** (13 real, rapid-fire `POST /tickets`
  calls from a single real employee, same live API process, run
  immediately after the baseline load test): requests 1-10 → `201`,
  requests 11-13 → `429`. **The limiter does correctly reject once its
  real per-employee density condition is actually met — it does not fail
  open.**
- **Real gap, confirmed by reading `tickets.py`/`chat.py`/`rate_limit.py`
  directly**: `check_ticket_creation_rate_limit` is wired to **only**
  `POST /tickets`. `/chat` and `/tickets/{id}/diagnose` — the two most
  expensive, LLM-bound, GPU-contending endpoints, and the ones that
  actually produced real transport errors/timeouts under this load — have
  **no rate limiting at all**. Under this specific realistic workload
  shape, that gap was never exercised into a failure (no single employee
  ever burst enough `/chat`/`/diagnose` traffic to matter), but it is a
  real, documented, unmitigated exposure for a burst concentrated on one
  employee's chat/diagnose traffic specifically.

## 3. Claims 1 and 5 — verdicts

### Claim 1 — "Tested at a workload equivalent to 1,000+ requests/day"

**MET.** The configured NHPP arrival schedule samples 1,151 total arrivals
per simulated day (mean rate 1,109 req/day-equivalent) from a fixed seed,
and **both real runs actually fired and received responses for all 1,151
scheduled requests** against the real live stack (not a subset, not a
dry run) — comfortably exceeding the 1,000+/day target in both the
configured-rate sense and the literal-requests-actually-executed sense.

### Claim 5 — "~40% lower p95 latency and ~30% lower inference cost"

**Cost component: substantially exceeded on the real measurable proxy,
but the literal dollar claim is not applicable.** Real dollar cost was
$0.00 in both conditions (100% local Ollama; all cloud-fallback attempts
failed in this environment), so "~30% lower cost" cannot be verified as a
dollar figure. Using real token count (the measurable substitute `tokens`
metric from `PLAN_APPROVED.md` §12) as the cost proxy, caching produced a
**46.5% reduction in tokens/chat-request** (716.8 vs. 1,339.8) — exceeding
the ~30% target on the one real resource dimension that genuinely has a
non-zero cost here. This is directly attributable to the measured 35.3%
real exact-cache hit rate (0% semantic): roughly one in three chat
requests skipped the real LLM call entirely.

**Latency component: NOT MET — and reversed.** Full-system chat p95 was
**45.58s server-side / 48.40s client-side**, versus baseline chat p95 of
**27.10s server-side / 27.11s client-side** — full-system was **~68%
higher**, not ~40% lower. This is a real, honestly-measured negative
result, with two plausible, evidenced (not merely guessed) causes:

1. **Resource contention from a concurrent, more expensive workload
   absent from baseline.** Full-system's workload mix includes
   `diagnose` (164 real ADK multi-turn calls, p50=45.3s, p99=141.4s) —
   baseline has none. Both kinds share the same bounded-concurrency pool
   (max 6) against the *same single local Ollama GPU instance*, so chat
   requests in full_system were real-world queued behind much
   longer-running diagnose calls — a real head-of-line-blocking effect,
   not a caching artifact. Baseline's chat requests only compete with
   near-free `create_ticket` calls, so they queue far less.
2. **Real reranker compute cost, paid only in full_system.** The
   `retrieve` span shows full_system's p99=1,803ms vs. baseline's
   p99=92ms — baseline's reranker is disabled (free pseudo-score
   fallback), while full_system pays a real reranker-model inference
   cost on every retrieval-cache miss. Measurable, but small relative to
   cause (1).

Neither cause is a caching *defect* — the caching layer itself clearly
works (35.3% real hit rate, 46.5% fewer tokens) — but the phase plan's
literal "~40% lower p95 latency" claim, measured exactly as specified
(real stack, paired runs, same seed), is **not met** in this environment,
primarily because this run's realistic workload mix puts an expensive,
GPU-contending new capability (the ADK diagnostic specialist) in the same
condition being credited with a *latency* improvement — the two effects
are confounded in the plan's own claim 5 framing, and this run surfaces
that honestly rather than hiding it.

## 4. Other genuinely surprising/concerning findings

- **Ollama cold-start cost is severe and real**: idle-unload (Ollama's
  default `keep_alive`) made the very first probe call take 47-65s
  (`agent run failed: timed out` on the very first attempt); warm
  back-to-back calls ran 7-25s. Any production deployment of this stack
  needs either a keep-warm ping or a much longer `OLLAMA_KEEP_ALIVE`
  before latency numbers are production-representative.
- **All real cloud-fallback attempts failed (100%, both runs, 120 total
  attempts)** — a real infrastructure constraint of this environment
  (likely no outbound network), not previously surfaced by this
  project's prior phases, since this is the first phase to generate
  enough real load for the fallback path to fire this many times.
- **The ADK diagnostic specialist is measurably the most expensive real
  capability in the system** (6,346 tokens/call average, 45-141s
  latency) and is not rate-limited at all.
