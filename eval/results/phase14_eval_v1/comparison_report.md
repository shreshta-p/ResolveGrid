# Phase 14 Task 4 — Golden-Eval Comparison (full_system vs baseline) and Claims 3/4

Generated from two real runs of the existing Phase 10 evaluation harness
(`uv run --package resolvegrid-api python -m resolvegrid_api.eval_worker`)
against the real, locally running Docker Compose stack (Postgres/Redis/
Ollama), plus one additional real run of Phase 13's standalone diagnostic
eval runner (`diagnostic_eval.run_diagnostic_eval_suite`). Nothing below is
estimated, back-calculated, or fabricated — every number is read directly
from real `EvalRun`/`EvalCaseResult` Postgres rows (dumped verbatim to the
`*_raw.json` files in this directory) or from the diagnostic runner's own
real `DiagnosticCaseOutcome` objects.

The metric definitions used below (`task_completion_accuracy`, which
dimensions are pooled, and why `adversarial`/`diagnostic` are excluded from
the paired composite) were written into `docs/EVALUATIONS.md`'s "Phase 14
Task 4" section **before** either run below was executed — see that
section for the full reasoning; it is not repeated here.

- git commit at run time: `61d819f` (clean tree; this task's own doc/raw
  -result files are the only uncommitted changes).
- Dataset: `eval/golden/v2/*.jsonl` (`dataset_version="v2"`) +
  `eval/adversarial/v1.jsonl`, 49 cases total — unchanged between runs.
- Judge/generation model: `qwen3:14b` via the live Ollama container —
  unchanged between runs.
- **Baseline `EvalRun.id = 133`**, **Full-system `EvalRun.id = 134`** (real
  rows in the `eval_run` table).
- Raw per-case results: `baseline_raw.json`, `full_system_raw.json`,
  `full_system_diagnostic_raw.json`.

## 1. Protocol

Only variable changed between runs: Task 2's four real env-var toggles,
each run as its own fresh OS process (module-level `os.environ.get` reads
require a process restart, exactly like Task 2's load-test precedent):

| | full_system (`EvalRun 134`) | baseline (`EvalRun 133`) |
|---|---|---|
| `RESOLVEGRID_RESPONSE_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_SEMANTIC_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_RETRIEVAL_CACHE_ENABLED` | true (default) | **false** |
| `RESOLVEGRID_RERANK_ENABLED` | true (default) | **false** |

Baseline ran first, full-system second; both completed with
`EvalRun.status="completed"`, no run-level errors.

## 2. A mechanical finding that must be reported before the numbers: most of this suite cannot be moved by the toggle under test

Read in full before trusting the headline numbers below — this materially
changes what the comparison can prove, and was discovered only while
tracing each dimension's real execution path during this task, not assumed
in advance:

- **`chat` dimension (18 cases)**: executed via `eval_worker.
  _execute_graph_dimension_cases` -> `classify_intent`
  (`services/agent-orchestration/graph.py:538`). `classify_intent` is a
  separate graph node from the `retrieve` node and does not consume
  `retrieve_fn`'s output — its real LLM call is a function of `input_text`
  alone. The dimension is graded purely on `expected_intent`/
  `expected_risk_level` (`chat_expected_match` in the raw JSON), never on
  retrieval/citation quality. **This dimension is structurally incapable of
  being moved by caching or reranking.**
- **`retrieval` dimension (18 cases)**: executed via `eval_worker.
  _execute_retrieval_case` -> `resolvegrid_api.eval_retrieval.evaluate_case`
  — a *different, older (Phase 7) module* that never imports
  `agent_retrieval.py` at all. `evaluate_case` has its **own**, separate
  `reranker_model` keyword (default `None` = Phase 7's plain fused-RRF
  order), and `_execute_retrieval_case` never passes it. **The retrieval
  dimension always scores the un-reranked fused order, in both conditions,
  completely independent of `RESOLVEGRID_RERANK_ENABLED`/
  `RESOLVEGRID_RETRIEVAL_CACHE_ENABLED`.** Those two env vars only gate
  `agent_retrieval.py`'s `retrieve_for_agent` (the real `/chat`/`/diagnose`
  production path) — a module this dimension's execution path never calls.
- **`tool` (1 case) / `approval` (2 cases)**: executed via the real
  `build_graph(..., retrieve_fn=retrieve_for_agent)` / tool-invocation
  graph — these **are** wired through `agent_retrieval.py` and therefore
  are the only cases in the entire 39-case composite mechanically capable
  of reflecting the toggle under test. Three cases.
- **`adversarial` (10 cases)**: already documented (see `docs/
  EVALUATIONS.md`'s "critical distinction" section) as a structural
  pass-through with no real action execution — excluded from the composite
  for that reason, confirmed unaffected (10/10 both runs).

Net effect: **36 of the 39 composite cases (chat + retrieval) are
mechanically guaranteed to be identical between conditions** regardless of
system behavior, because neither dimension's real execution path in this
harness touches the toggled code. Only 3 cases (tool + approval) could, in
principle, differ. This is a real gap in this harness's coverage of its own
stated before/after variable, surfaced by this task, not a limitation
specific to this run.

## 3. Measured results

| Dimension | Baseline (`run 133`) | Full-system (`run 134`) |
|---|---|---|
| chat | 15/18 = 0.833 | 17/18 = 0.944 |
| retrieval | 18/18 = 1.000 | 18/18 = 1.000 |
| tool | 1/1 = 1.000 | 1/1 = 1.000 |
| approval | 2/2 = 1.000 | 2/2 = 1.000 |
| adversarial (excluded from composite) | 10/10 = 1.000 | 10/10 = 1.000 |
| **Composite `task_completion_accuracy`** (chat+retrieval+tool+approval, per `docs/EVALUATIONS.md`) | **36/39 = 0.9231** | **38/39 = 0.9744** |

**The entire observed 2-case delta is in `chat`, not in the three
toggle-sensitive cases (tool/approval, unchanged 100% both runs) or in
retrieval (unchanged 100% both runs).** Tracing the two differing cases
directly (`chat.unclear.003` input `"ticket"`, `chat.unclear.004` input
`"asdf1234 blah blah nothing useful here"`, both `expected_intent="unclear"`):

- Baseline: both classified `actual_intent="general_question"` (fail).
- Full-system: both classified `actual_intent="unclear"` (pass).

Per §2 above, `classify_intent` never consumes retrieval output and is
unaffected by any of the four toggled flags — this is a real LLM making a
different real call-to-call judgment on two genuinely ambiguous, borderline
short inputs, not a caused effect of caching or reranking. **The honest
conclusion is that this comparison measured zero causally-attributable
difference between the two conditions on this 39-case golden suite** — the
2-case gap is sampling noise on the one dimension that can't even be
affected by the variable under test, and the three cases that *can* be
affected showed no change at all.

### Diagnostic dimension (full-system only, not part of the paired comparison)

Per `docs/EVALUATIONS.md`'s pre-committed exclusion (no honest "disable ADK"
toggle exists — the diagnostic specialist did not exist in the
reconstructed baseline state at all), run once, full-system only, via
`diagnostic_eval.run_diagnostic_eval_suite` (all 6 real `diagnostic_v1.jsonl`
cases, no sampling):

| | Result |
|---|---|
| `safety_passed` (zero-tolerance: forbidden_actions + judge-based mutation-completion-claim check) | **6/6 = 1.000** |
| `quality_passed` (tool-selection correctness + citation presence + judge content-quality verdict) | **0/6 = 0.000** |
| `passed` (safety AND quality) | 0/6 |

**This is a genuinely concerning, surprising result worth flagging on its
own, independent of claims 3/4** (it doesn't feed the composite either
way): every one of the 6 real diagnosis runs produced a response the judge
correctly failed on content grounds — 3 of 6 defaulted to a VPN-themed
answer regardless of the ticket's actual subject (a coffee-machine
complaint, an employee-departure audit-history question, an email-lockout
ticket), and 2 of 6 called the wrong tool entirely (`retrieve_ticket_kb_
evidence` instead of the expected `find_similar_tickets`/
`get_employee_entitlements`). This reproduces, more severely (0/6 here vs.
a non-zero fraction in Phase 13's own prior real runs), the exact
pre-existing, already-documented weakness `docs/EVALUATIONS.md`'s
diagnostic section and ADR 0005 describe ("real local-qwen3 is NOT reliably
consistent... e.g. defaulting to a VPN-flavored query regardless of a
ticket's actual, very different subject") against this dev environment's
large, heavily VPN-skewed KB corpus. Safety held perfectly (no forbidden
action, no fabricated completion claim in any of the 6 real runs) — the
zero-tolerance guarantee this phase cares about most was never at risk —
but the specialist's actual diagnostic quality, measured honestly today, is
worse than this project's own prior documentation already warned it might
be.

**This is now the third independent real data point showing the same
weakness, not a one-off result from this task's single run.** The
diagnostic suite was independently reproduced during spec-compliance review
of this task, with the same `safety_passed=6/6`, `quality_passed=1/6`
outcome — consistent with (and slightly less severe than, but in the same
direction and magnitude as) this task's own `0/6`. Taken together with
Phase 13's original finding (ADR 0005, `2/6`), all three real runs agree:
`safety_passed` holds at `6/6` every time (the zero-tolerance guarantee is
robust), while `quality_passed` is consistently low (`0/6`-`2/6` out of 6)
against this dev environment's VPN-heavy KB corpus. This is not three
separate mysteries — it is the same already-documented ADR 0005 content
-grounding weakness, now corroborated a third time under slightly different
run conditions, which makes it materially less likely to be a fluke of any
one run's sampling.

## 4. Claims 3 and 4 — verdict

**Claim 3** ("end-to-end task-completion accuracy improving from ~60% to
~85%"): **NOT MET.**

Measured: baseline 92.3% -> full-system 97.4%. Both numbers are far above
the claimed 60%/85% reference points, and — per §2/§3 above — the measured
2.6-point gap between them is not causally attributable to the Phase 9-13
capabilities under test at all; it is LLM sampling variance on two
genuinely ambiguous inputs in a dimension (`chat`) that structurally cannot
be affected by caching or reranking. The three cases that genuinely do
exercise the toggled code path (`tool`/`approval`) showed zero change. This
reconstructed baseline — the only honest "before" available without
fabricating history (see this plan's own "real, upfront constraint"
section) — simply is not the ~60%-accuracy system the original claim
describes; disabling caching/reranking on an already-fully-agentic
RAG+tool-calling system leaves it performing nearly identically to the
full system on this golden set, because this golden set's deterministic
grading criteria (intent string, chunk ids, tool name/params, approval
status) mostly don't depend on whether results were cached or reranked.

**Claim 4** ("relative evaluation pass-rate improvement of ~50%, e.g. 56%
-> 84%"): **NOT MET.**

Measured relative improvement: `(0.9744 - 0.9231) / 0.9231 = 5.6%` — about
1/9th of the ~50% target, and (same caveat as Claim 3) not demonstrated to
be a real causal effect of the system capabilities under test rather than
LLM-call noise on two borderline cases.

### Why this is the expected, honestly-predicted outcome, not a new surprise

`docs/EVALUATIONS.md`'s Task 4 section, written before either run executed,
already flagged that this comparison's "before" (cache/rerank off on an
already-fully-agentic system) is a materially weaker reconstruction than
plan.md's original claims 3/4 likely intended (reading in context as a
comparison against a pre-AI/manual process), and that "a small or even zero
measured delta here would not be surprising." That is exactly what was
measured. What *was* a genuine, not-predicted-in-advance finding is §2's
mechanical result: two of the four composite dimensions (`chat`,
`retrieval`) turn out to be wired through code paths this harness's own
retrieval dimension (a separate, older Phase 7 module) and classify_intent
node never route through the toggled flags at all — so even the *narrow*
question this reconstructed baseline could answer ("does disabling caching/
reranking measurably change golden-eval pass rate?") was only actually
testable on 3 of 39 cases, and those 3 showed no change.

## 5. What this does and does not tell us

- It does **not** refute plan.md's original claims about going from a
  pre-AI/manual process to this system — that comparison was never
  possible to run honestly in this project (no frozen "v0" baseline was
  ever recorded; see this phase's own plan doc) and nothing here speaks to
  it either way.
- It **does** show that, on this golden set, the specific Phase 12/13
  additions this phase can honestly toggle (response/semantic/retrieval
  caching, reranking) do not measurably change correctness outcomes versus
  having them off — consistent with their original design intent (caching
  and reranking are latency/relevance-quality optimizations layered on an
  already-correct pipeline, not correctness-changing features by design).
  Task 2's own finding (full-system p95 latency *regressed* under this
  same toggle set) plus this task's finding (no measurable correctness
  change) together suggest these Phase 12 features' real, demonstrated
  value in this project to date is smaller than plan.md's original claims
  implied, not larger.
- It **does** surface a real, previously-undocumented gap in this harness's
  own test coverage of its stated before/after variable (§2) — a decent
  follow-up for a future phase would be making `eval_retrieval.py`'s
  retrieval-dimension execution route through the real `agent_retrieval.py`
  production path (or explicitly documenting, as this report now does,
  that it deliberately doesn't).
- It **does** surface a real, concerning regression in the diagnostic
  specialist's measured answer quality (0/6 this run) worth a dedicated
  look in a future phase, independent of this task's claims-3/4 scope.
