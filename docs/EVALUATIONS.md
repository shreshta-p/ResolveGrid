# Evaluations

Full methodology for Phase 10's unified evaluation harness. Golden-dataset and grading
methodology was originally sketched in the approved architecture plan (§8); this document is
the as-built record, written at Phase 10's close (Task 9) against the real, measured system.

## The `EvalCase` schema

One shape (`resolvegrid_evaluation.schema.EvalCase`, pydantic, frozen) covers every evaluation
dimension instead of five bespoke per-phase case shapes. Most fields are optional, since a given
case only exercises the fields relevant to its `dimension`:

- `case_id` (stable, human-readable, e.g. `"chat.greeting.001"`), `dataset_version`,
  `dimension: Literal["chat", "retrieval", "tool", "approval", "adversarial"]`.
- `principal_fixture` — a `{role, department, scope}` dict the harness resolves into a real
  `Principal` backed by a real, idempotently get-or-created `Employee`/`RoleAssignment`.
- `input_text`, `expected_intent`, `expected_risk_level` — chat-dimension (Phase 6 shape).
- `answerable`, `relevant`, `distractor`, `must_not_appear`, `authz` — retrieval-dimension
  (Phase 7/8 shape, reused verbatim, including the VPN v1/v2 distractor and authz-leakage cases).
- `expected_tool_name`, `expected_tool_params` — tool-dimension.
- `expected_approval_decision`, `expected_final_status` — approval-dimension.
- `forbidden_actions` — must never appear in `actual_result["actions_taken"]`, regardless of
  dimension; this is the field the adversarial suite's zero-tolerance guarantee is built on.
- `rubric`, `judge_dimensions` — set only on cases that need judge grading (see below). None of
  `eval/golden/v2/*.jsonl`'s 39 cases set these — every one of them is deterministically graded.
  Only the separate 18-case `eval/golden/judge_calibration_v1.jsonl` sample exercises the judge.
- `provenance`, `human_review_status` — every shipped case in `eval/golden/v2/*.jsonl` and
  `eval/adversarial/v1.jsonl` is `human_review_status == "approved"`, enforced by
  `apps/api/tests/test_golden_v2_dataset_shape.py`, not just convention.

`load_eval_cases(path)` (`services/evaluation/src/resolvegrid_evaluation/schema.py`) is the JSONL
loader, mirroring `resolvegrid_api.eval_retrieval.load_golden_cases`'s established shape.

## Grading order: deterministic first, judge only for genuinely fuzzy dimensions

`apps/api/src/resolvegrid_api/eval_worker.py`'s `_grade_case_safe` (around line 965) checks
`case.judge_dimensions` first: if unset, the case is graded entirely by
`services/evaluation`'s deterministic graders (`grader_type="deterministic"`); only if
`judge_dimensions` is set does it call `judge_response_real` (`grader_type="judge"`). This is
not incidental — every one of the 49 cases in the real eval run recorded below graded
deterministically, because chat/retrieval/tool/approval cases all have objectively checkable
expected values (intent string, risk-level string, chunk ids, tool name/params, approval
decision/status). Judge grading exists specifically for dimensions no deterministic check can
cover — free-text groundedness, free-text correctness, and abstention-appropriateness — and is
currently only exercised by the calibration harness itself (see below), not by the main golden
set. A future phase adding free-text-graded chat cases would set `judge_dimensions` on those
specific cases; nothing about the harness's structure needs to change.

Deterministic graders (`services/evaluation/src/resolvegrid_evaluation/graders.py`), each a pure
`(case, actual_result) -> GradeResult` function:

- `grade_schema_validity` — output conforms to its dimension's expected structured shape (tool
  dimension validates against `packages/contracts`'s `TOOL_REGISTRY` JSON Schema directly, not a
  reimplementation).
- `grade_citation_correctness` — every cited chunk id was actually retrieved (re-checks the eval
  harness's own recorded `actual_result`, independently of `verify_citations`'s in-graph check).
- `grade_tool_argument_accuracy` — tool name and params match `case.expected_tool_name`/
  `expected_tool_params` exactly.
- `grade_approval_compliance` — approval decision/final status match expectations.
- `grade_forbidden_actions` — zero-tolerance: any `forbidden_actions` entry appearing in
  `actual_result["actions_taken"]` fails, regardless of anything else.

Every grader has a determinism test (same inputs twice → byte-identical `GradeResult`) in
`services/evaluation/tests/test_graders.py` — trivial for pure functions, written anyway per the
exit criterion, not skipped as "obviously true."

Retrieval IR metrics (`recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`) were
consolidated (Task 3) from `apps/api/src/resolvegrid_api/eval_retrieval.py`'s formerly-private
functions into `services/evaluation/src/resolvegrid_evaluation/retrieval_metrics.py` as a public,
independently-tested API. `eval_retrieval.py` is now a thin wrapper (real SQL/authz/reranker
invocation stays in `apps/api`, per this repo's established dependency-direction rule); this was
a pure refactor-for-reuse, verified by re-running `test_eval_retrieval.py` unmodified after the
move (see the "confirming the numbers still match" section below).

## The adversarial suite: zero-tolerance policy, and a critical distinction

`eval/adversarial/v1.jsonl` holds 10 cases covering every one of plan.md §8's 9 adversarial
categories (the "duplicate/expired approval replay" category contributes 2 cases — one duplicate,
one expired). Some cases are newly authored; others deliberately wrap an already-passing
Phase 8/9 pytest assertion (extracted into a shared, reusable callable so both the original test
and the new adversarial-suite wrapper call the exact same logic, rather than reimplementing it a
second time that could silently drift from the original) — see `docs/DECISION_LOG.md`'s
2026-09-11 entry for which cases reuse vs. author fresh, and why.

**There are two, deliberately different, adversarial-grading mechanisms in this codebase, and
conflating them would be a real mistake:**

1. **`apps/api/tests/test_adversarial_suite.py`** — the REAL, load-bearing guarantee. Every one
   of the 10 scenarios is genuinely re-executed here: real DB, real HTTP-equivalent in-process
   calls, real Ollama completions where relevant (the injected-document case), real graph
   invocations. This file asserts **zero failures, hard** — a single failing case fails the whole
   `apps/api` pytest suite, which already runs in CI on every push. This is this phase's core
   security-regression gate.
2. **The batch eval runner's `adversarial` dimension** (`eval_worker.py`'s
   `_grade_adversarial_case`) — a **structural pass-through**, not real re-execution. It grades
   each adversarial case's `forbidden_actions` field against an `actions_taken` list that is
   *always empty* in this harness's own execution (the harness performs no actions for these
   cases), so the grade is near-tautologically `passed=True`. This produces a real, genuine
   `EvalCaseResult` row for every adversarial case — not a fabricated pass — but it is
   intentionally shallow, and `eval_worker.py`'s own module docstring says so explicitly.

**Do not read a green `EvalRun.dimension="adversarial"` result as proof the system passed the
adversarial suite in that run.** It proves the harness recorded a case for that run; it does
*not* re-verify the injected-document, cross-user, malformed-tool-call, or replay defenses. The
real proof is exclusively `test_adversarial_suite.py`'s separate, zero-tolerance, CI-gated pytest
suite, re-run independently of any batch eval run. This distinction is deliberate (re-executing
the same scenarios a second time inside the batch runner would itself be the "duplicated
assertion that can silently drift apart" risk plan.md's Task 6 notes warn against, generalized
one level further) and is recorded here so nobody mistakes the batch runner's pass-through for a
security proof.

## The model-judge grader and its calibration

`services/evaluation/src/resolvegrid_evaluation/judge.py`'s `judge_response` sends a structured
grading prompt per `judge_dimensions` entry (`groundedness`, `correctness`,
`abstention_appropriateness`) and parses a `{"passed": bool, "reasoning": str}` response,
rejecting/retrying once on unparseable output before falling back to `passed=False` rather than
raising — a judge failure must never crash a batch run. `JudgeVerdict` carries a `case_id` field
beyond the plan doc's original literal spec — a deliberate, documented bridge letting
`calculate_agreement` correlate pooled verdicts against per-case `human_labels`; see
`docs/DECISION_LOG.md`'s 2026-09-11 entry for why this was necessary (the plan doc never checked
`JudgeVerdict`'s and `calculate_agreement`'s signatures against each other, and omitting
`case_id` would have made the stated `human_labels` shape impossible to correlate against).

**Calibration harness**: `eval/golden/judge_calibration_v1.jsonl` (18 hand-labeled cases: 6 per
dimension) + `calculate_agreement(judge_verdicts, human_labels) -> AgreementReport`. Chosen
threshold: **≥0.80 exact agreement** per dimension (a common practical bar for LLM-judge
calibration, justified in `apps/api/tests/test_eval_judge_calibration.py`'s module docstring).
Per this phase's explicit design principle, the threshold is a documented target, not something
silently lowered to match whatever number came back — the test asserts the calibration harness
runs end to end and reports a real, bounded (`0.0`-`1.0`) number; it does **not** assert the
number meets 0.80, precisely so a real low measurement would be visible, not hidden by a failing
CI gate.

**Real measured result** (re-confirmed during this task's fresh-state verification, run against
the live `qwen3:14b`, commit `e442893543d444f0dce571699e2ace2f31a022fa`):

| Dimension | Agreement |
|---|---|
| groundedness | 1.000 (6/6) |
| correctness | 1.000 (6/6) |
| abstention_appropriateness | 1.000 (6/6) |
| overall | 1.000 (18/18) |

**This must be read honestly, not as strong evidence the judge is well-calibrated for hard
real-world cases.** Independent review of `judge_calibration_v1.jsonl`'s 18 cases (during Task
4's own review cycle) found: the abstention-appropriateness cases are genuinely good, hard test
cases (real ambiguity about whether abstaining was the right call). The groundedness and
correctness cases, however, are clean, template-shaped contradiction-spotting exercises — each
"false" variant directly contradicts a single short retrieved chunk sitting immediately next to
it in the prompt, not a subtle near-miss requiring the judge to reconcile competing signals or
catch a plausible-sounding but wrong claim. A perfect 1.000 here mainly demonstrates the judge
can spot a flat, adjacent contradiction when ground truth is quoted right next to the claim — a
real, useful, but narrow capability. It should **not** be cited as evidence the judge would catch
subtler fabrications, partial-truth answers, or well-hedged incorrect claims in production
traffic. A future phase wanting a stronger calibration signal should add harder, more adversarial
groundedness/correctness cases (paraphrased contradictions, multi-hop reasoning, plausible-but-
wrong claims not directly adjacent to the contradicting evidence) rather than trusting this
number as a ceiling on judge quality.

## Methodology notes on the golden dataset and its self-checks

- `eval/golden/v2/chat_v1.jsonl` (18 cases), `retrieval_v1.jsonl` (18 cases),
  `tools_approvals_v1.jsonl` (3 cases) re-express Phase 6/7/8's already-measured golden cases in
  the unified schema (same queries/expectations, `human_review_status="approved"` carried over as
  a legitimate direct carry-over, not a fresh unreviewed draft) and add new tool/approval cases
  exercising Phase 9's VPN-grant flow (one `tool` case, one `approval.approved` case, one
  `approval.rejected` case — both approval branches covered).
- Several adversarial cases (conflicting/stale policy, simulated provider outage, duplicate/
  expired approval replay) are thin wrappers around **existing** Phase 5/8/9 test assertions,
  extracted into shared callables so both the original test and the new adversarial-suite wrapper
  call the identical logic — this prevents the two from silently drifting apart over time, at the
  cost of these specific cases not being "fresh" re-implementations. This is a deliberate,
  plan-endorsed reuse decision (plan.md's own "Notes" section calls it out explicitly), not
  under-building.
- Two adversarial-suite test bodies were initially found (in code review) to be gaming a
  structural, AST-based completeness self-check — loading an `EvalCase` only to assert an
  already-globally-checked trivial property, rather than genuinely exercising that case's own
  data. Fixed to be genuinely case-driven (`git log` for the exact commit:
  `fix(eval): make cross-user and fabricated-ticket-id cases genuinely case-driven`). Worth
  noting here because it's a real example of the suite's own self-verification catching a
  shortcut in itself, not just in the system under test.

## Running the suite locally

```
uv run --package resolvegrid-api python -m resolvegrid_api.eval_worker
```

Mirrors `ingestion_worker.py`'s `main()` pattern exactly: opens a plain synchronous session
against `DATABASE_URL`, calls `run_eval_suite(session)`, commits, and prints a one-line summary
(`EvalRun <id>: status=<status> dataset_version=<version> git_commit=<sha>`). Verified working as
documented during this task's fresh-state verification — see `docs/PROGRESS.md`'s Phase 10 row
for the real run this produced (`EvalRun` id 12, `status=completed`, 49 cases across 5
dimensions, real per-dimension pass rates visible at `/admin/evals`).

Requires: a running Postgres with migrations applied through `0013_eval_runs_results`, a running
Ollama with `qwen3:14b`/`nomic-embed-text` pulled, the seed corpus ingested (`python -m
resolvegrid_api.ingestion_worker`), and employees seeded (`scripts/seed_employees.sh`) — the
tool/approval-dimension cases construct their own fixture principals on demand, but the
retrieval-dimension cases need the real corpus already ingested.

The Arq-backed path (`run_eval_suite_task`, registered in `WorkerSettings.functions`) is the
production path a scheduled/triggered job would use; `test_eval_worker.py`'s
`test_arq_worker_processes_run_eval_suite_task_via_real_redis` proves it end to end against real
Redis, mirroring `test_ingestion_worker.py`'s identical precedent.

## Known, deliberate simplifications (stated plainly, not buried)

- **`dataset_version` filtering is asymmetric**: `eval/golden/v2/*.jsonl` cases are filtered by
  the passed-in `dataset_version` argument (default `"v2"`); `eval/adversarial/v1.jsonl`'s own
  `"v1"`-versioned cases are **always** included regardless of that argument, since the
  adversarial suite's zero-tolerance policy is deliberately independent of which golden-dataset
  version is under test.
- **Tool/approval cases submit `expected_tool_params` directly**, not derived from `input_text`
  via NLU argument-extraction — this system has no such component yet. This harness proves the
  real allowlist → schema-validate → approval-pause → decide → resume → execute pipeline works
  end to end for an already-correct proposed call; it does not (and today cannot) test whether a
  model would correctly parse free text into that same call.
- **The adversarial dimension's batch-runner grading is a structural pass-through**, not real
  re-execution — see the "critical distinction" section above. This is the single most important
  caveat in this document; repeated here because it is the one most likely to be misread later.
- **Session/commit boundary**: `run_eval_suite`'s own `EvalRun`/`EvalCaseResult` writes are
  flush()-only on the caller's session (mirroring `run_seed_corpus_ingestion`'s contract), but
  principal/employee/approval fixture rows the tool/approval-dimension cases need are committed
  on independent sessions, because the existing graph-adapter modules (`agent_retrieval.py`,
  `approval_service.py`, `agent_mutation_execution.py`) already open and commit their own
  sessions per call — an architectural constraint from before this phase, not a new choice.

## Phase 14 Task 4: defining "task-completion accuracy" (plan.md claims 3 and 4), written before running anything

This section is written and committed to disk **before** the baseline-vs-full-system comparison
run below is executed, specifically so the definition cannot be shaped by having already seen the
numbers — the standing project rule against inventing metrics post hoc to hit a target.

**The two claims being tested** (`plan.md`'s Outcomes section, reproduced verbatim):
- Claim 3: "end-to-end task-completion accuracy improving from ~60% to ~85%."
- Claim 4: "relative evaluation pass-rate improvement of ~50% (e.g. 56% → 84%)."

**Neither term exists as a field anywhere in this schema today.** Both must be composed from the
real `EvalCaseResult.passed` booleans this harness already produces — nothing new is added to
`EvalCase`, `EvalRun`, or `EvalCaseResult` for this task.

### The composite metric

For a given real `EvalRun` (produced by `run_eval_suite`, one row per graded case in
`EvalCaseResult`):

```
task_completion_accuracy(run) =
    count(EvalCaseResult WHERE eval_run_id = run.id
                           AND dimension IN ('chat','retrieval','tool','approval')
                           AND passed = true)
    -----------------------------------------------------------------------------
    count(EvalCaseResult WHERE eval_run_id = run.id
                           AND dimension IN ('chat','retrieval','tool','approval'))
```

i.e. a single **pooled** pass rate (not a per-dimension average) across the four dimensions whose
cases each represent one real, checkable way the system can complete a user's actual task:
chat (did it classify intent/risk correctly — the entry point to acting on a request at all),
retrieval (did it find the correct grounding evidence an answer depends on), tool (did it select
and call the correct tool with correct parameters — the actual "doing" of a task), approval (did
it correctly gate/resolve an approval-required action end to end). Pooling (not per-dimension
weighting) is deliberate: the four dimensions have very different case counts in the current
golden set (chat 18, retrieval 18, tool 1, approval 2 — see below), and choosing dimension weights
after the fact is exactly the "invented post hoc to hit 85%" failure mode `plan.md` warns against.
An unweighted pool over every real case is the one construction that cannot be tuned toward a
target number.

**Two dimensions are deliberately excluded from this composite, for stated reasons, not silently
dropped:**

- **`adversarial` (10 cases, always included in every `run_eval_suite` call regardless of
  `dataset_version`)** is excluded because this document's own "critical distinction" section
  above already establishes that the batch runner's adversarial grading is a **structural
  pass-through** — `actions_taken` is always empty in this harness's own execution, so
  `grade_forbidden_actions` passes near-tautologically. Folding a near-100%, not-really-measuring-
  anything dimension into the composite would inflate it without reflecting any real capability.
- **`diagnostic` (6 cases, `eval/golden/diagnostic_v1.jsonl`, Phase 13's ADK specialist)** is
  excluded from the **paired baseline-vs-full-system delta** used for claims 3/4, for a different
  reason than adversarial: per ADR 0005, `diagnostic` cases are not wired into `EvalRun`/
  `EvalCaseResult` at all (no `EvalRun` row is produced for them — they run via the standalone
  `diagnostic_eval.run_diagnostic_eval_suite`), and more importantly, **the ADK diagnostic
  specialist has no real "disable" toggle** — unlike response/semantic/retrieval caching and
  reranking (Task 2's real env-var flags), it simply did not exist before Phase 13, and this
  phase's reconstructed baseline condition (plan.md's "ticket-creation + `/chat` RAG path as it
  existed after Phase 7/8") is a state in which the diagnostic specialist is structurally absent,
  not merely switched off. Inventing a synthetic "ADK off" condition to force a paired number here
  would be fabricating a baseline that never existed, which the project's standing rule forbids.
  `diagnostic`'s pass rate is still measured and reported (full-system only), as a clearly
  separate, non-paired, descriptive data point — never folded into the claim-3/4 composite.

### Mapping onto claims 3 and 4

Both claims are read as two framings of the **same** underlying composite metric, not two
different metrics (plan.md gives no separate definition for "evaluation pass-rate" vs
"task-completion accuracy", and this project's precedent is to avoid inventing a second parallel
mechanism when one already covers the question — see this doc's "two adversarial-grading
mechanisms" section for the exact failure mode being avoided):

- **Claim 3** = `task_completion_accuracy(baseline_run)` vs `task_completion_accuracy(full_system_run)`,
  compared against the absolute target (~60% → ~85%).
- **Claim 4** = the **relative** change of that same pair,
  `(full_system - baseline) / baseline`, compared against the ~50% relative-improvement target
  (plan.md's own worked example, 56% → 84%, is a 50.0% relative increase).

### Protocol (fixed per `PLAN_APPROVED.md` §12: same dataset, same seed, only the variable under test changes)

- Dataset: `eval/golden/v2/*.jsonl` + `eval/adversarial/v1.jsonl`, `dataset_version="v2"` (the
  harness's own default — unchanged between runs).
- Judge model: `qwen3:14b` via the live Ollama container (unchanged between runs).
- Variable under test: Task 2's four real env-var toggles only —
  `RESOLVEGRID_RESPONSE_CACHE_ENABLED`, `RESOLVEGRID_SEMANTIC_CACHE_ENABLED` (`routers/chat.py`),
  `RESOLVEGRID_RETRIEVAL_CACHE_ENABLED`, `RESOLVEGRID_RERANK_ENABLED` (`agent_retrieval.py`).
  Because these are read once at module-import time (not re-checked per request), each condition
  is run as its own fresh OS process invoking
  `uv run --package resolvegrid-api python -m resolvegrid_api.eval_worker`
  — baseline with all four set `false`, full-system with all four left unset (real default
  `true`) — never flipped mid-process, exactly mirroring Task 2's own precedent.
- Same git commit for both runs (recorded on each `EvalRun.git_commit` automatically).
- Both runs produce one real, committed `EvalRun` + its `EvalCaseResult` rows in Postgres; raw
  per-case JSON exports and a comparison report are stored under
  `eval/results/phase14_eval_v1/`.

### A reasoning note on what this comparison can and cannot show

Because the only variable under test is caching/reranking (not the presence of RAG/tool-calling
itself — both conditions still run the full Phase 6-9 chat graph with real retrieval, tool calls,
and approval gating; only whether results are cached and whether retrieval is reranked differs),
this comparison is a materially *weaker* "before" than plan.md's original claims 3/4 likely
intended (which read, in context, as comparing against a pre-AI/manual process, not against an
already-fully-agentic system with two optional optimizations switched off). A small or even zero
measured delta here would not be surprising, and should not be read as evidence the project's
overall before/after improvement is small — only that *this specific, honestly-reconstructable*
slice of "before" (cache/rerank off) is a narrow one. This is stated here, before running
anything, precisely so a small-delta result cannot later be read as a surprise or explained away.
