# ADR 0005: Diagnostic-Dimension Eval Integration Depth (Phase 13 Task 5)

## Status
Accepted

## Context

Task 5 (`docs/superpowers/plans/2026-10-21-phase13-adk-diagnostic-specialist.md`)
requires a real, graded golden-case set for the ADK diagnostic specialist
(`diagnostic_agent.diagnose_ticket`), plus real adversarial authz/
prompt-injection proof. The plan's Phase 10 precedent
(`docs/superpowers/plans/2026-09-05-phase10-eval-harness.md`) built one
unified mechanism -- `EvalCase`/`EvalRun`/`EvalCaseResult`, a real Arq batch
runner (`eval_worker.py`), deterministic graders + a calibrated judge -- and
that plan's own notes warn against inventing "a third, inconsistent eval
mechanism." This ADR records the real investigation into whether a sixth
`EvalCase.dimension` value ("diagnostic") belongs in that full, Arq
-integrated harness, or in a smaller, same-philosophy standalone mechanism,
and why.

## Real findings (evidence, not assumption)

1. **No DB migration is required either way.** `EvalCaseResult.dimension`
   (`apps/api/src/resolvegrid_api/models/evaluation.py`) is `Mapped[str]`,
   and migration `0013_eval_runs_results.py` creates it as a plain
   `sa.Column('dimension', sa.String(), nullable=False)` -- no `CheckConstraint`,
   no DB-level enum. A 6th dimension value costs nothing at the database
   layer, full integration or not. This removes what might otherwise have
   been the biggest real argument against full integration.

2. **The pydantic-side cost is also small by itself.** `resolvegrid_evaluation
   .schema.Dimension` is a `Literal[...]` in a pure library with no backing
   table; adding `"diagnostic"` to it (done as part of this task) is a
   backward-compatible, single-line change with no migration and no effect
   on any existing case file (the literal only restricts what NEW cases may
   declare).

3. **The real cost is `eval_worker.py`'s execution harness, which has
   exactly two execution shapes today, and `diagnose_ticket` fits neither.**
   - The sync shape (`_execute_retrieval_case`/`_execute_adversarial_case_structurally`,
     driven from `run_eval_suite`'s own synchronous loop) delegates to
     existing synchronous, DB-session-only functions.
   - The async shape (`_execute_graph_dimension_cases`) drives the
     `AgentState`-shaped LangGraph `chat`/`tool_invocation` graphs via
     `ainvoke()`, built once per suite run and shared across every chat/tool
     /approval case.
   - `diagnose_ticket(session, tracer, principal, ticket_id)` is neither: it
     is `async`, but it is not an `AgentState`/LangGraph node at all (ADR
     0004's own Task 4 addendum is explicit that no LangGraph graph wraps
     it, for real, specific reasons -- there is no pause/resume boundary
     and the real signature doesn't fit `AgentState`). Integrating it would
     require inventing a real THIRD execution shape in `eval_worker.py`,
     not reusing either existing one, for a single new dimension.

4. **A single diagnostic case is real-model-call-expensive in a way no
   existing dimension is, by this phase's own prior, already-recorded
   findings.** ADR 0003 Finding 5 (Task 1) found a trivial, single
   zero-argument tool could legitimately take up to 36 real sequential
   `qwen3:14b` turns. ADR 0004 (Task 3) raised `_MAX_LLM_CALLS` to 60 for
   this specialist specifically because it has 4 real tools, each an
   independent opportunity for that same pathology, and documents this as a
   genuine, observed model behavior, not a hypothetical worst case. Every
   existing golden case in `eval/golden/v2/*.jsonl` costs at most a handful
   of real model calls (chat: 1-2; tool/approval: `case.expected_tool_params`
   is submitted directly, bypassing any real multi-turn reasoning entirely,
   per `eval_worker.py`'s own documented limitation). A handful of real
   diagnostic cases integrated into the SAME suite that already runs on
   every CI invocation (Phase 10 Task 9's exit criterion) would add a
   disproportionate, qualitatively different real-wall-clock cost to a
   batch job this phase's own CI notes already flag as "12-20x slower" on
   CPU-only runners for ordinary `qwen3:14b` calls -- for only 6 new cases
   out of what would become dozens.

5. **None of the three dimension-SPECIFIC deterministic graders apply to a
   free-text diagnosis at all.** `grade_schema_validity`'s tool-dimension
   branch checks a single proposed tool call's JSON-Schema validity --
   not applicable (a diagnosis's real value is prose plus citations, not
   one tool call). `grade_tool_argument_accuracy` and
   `grade_approval_compliance` are both specific to the tool/approval
   dimensions' "one expected call" shape and have no diagnostic analogue.
   Only the two genuinely dimension-AGNOSTIC graders
   (`grade_citation_correctness`, `grade_forbidden_actions` -- neither
   branches on `case.dimension` at all) reuse directly. This is real,
   partial reuse, not zero reuse, but it is far short of "this dimension
   slots cleanly into the existing per-dimension grading branch" the way
   Phase 10's five original dimensions do.

## Decision

**Do not add "diagnostic" execution to the full `EvalRun`/`EvalCaseResult`/
`eval_worker.py`/Arq production pipeline in this task.** Instead:

- `resolvegrid_evaluation.schema.Dimension` gains the `"diagnostic"` literal
  value (cheap, honest typing for case files that reuse `EvalCase`/
  `load_eval_cases`, with zero effect on the DB or the existing Arq runner,
  which never loads this value since it only reads
  `eval/golden/v2/*.jsonl` + `eval/adversarial/v1.jsonl` by fixed filename).
- A new, GENERIC (not diagnostic-specific) deterministic grader,
  `grade_no_mutation_claim_in_text`, is added to
  `services/evaluation/graders.py` -- real, reusable, dimension-agnostic,
  following this module's existing conventions exactly (pure function,
  `GradeResult`, determinism-tested). This is real reuse-grade library
  growth, not a diagnostic-only side mechanism.
- The judge grading methodology (`resolvegrid_evaluation.judge.judge_response`
  / `apps/api/src/resolvegrid_api/eval_judge.judge_response_real`) is reused
  VERBATIM, unmodified, for the one genuinely fuzzy diagnostic question
  ("is the suggested next step / issue categorization actually sensible
  given the ticket") -- no new judge dimension literal was needed; existing
  `"correctness"`/`"groundedness"`/`"abstention_appropriateness"` dimensions
  already describe exactly the fuzzy questions a diagnosis case needs
  judged, via a case-specific `rubric` string (the mechanism `judge.py`
  already provides for exactly this purpose).
- A new, STANDALONE case file (`eval/golden/diagnostic_v1.jsonl`, loaded via
  the real, unmodified `resolvegrid_evaluation.schema.load_eval_cases`) and
  a new, standalone runner module
  (`apps/api/src/resolvegrid_api/diagnostic_eval.py`) run each case through
  the REAL `diagnose_ticket` against the real live stack, grade it via the
  real reused graders above, and are exercised by a real, committed pytest
  file (`apps/api/tests/test_diagnostic_eval.py`) -- NOT by `eval_worker.py`,
  NOT persisted as `EvalRun`/`EvalCaseResult` rows, NOT reachable from the
  Arq `WorkerSettings`.

This satisfies the plan's "do not invent a third, inconsistent eval
mechanism" concern in substance (same `EvalCase` shape, same
deterministic-then-judge grading order, same graders where they genuinely
apply, same real-system-only execution policy, same human-reviewed-fixture
convention) while avoiding a real, disproportionate structural cost
(inventing a third `eval_worker.py` execution shape and materially
inflating the existing CI-gated batch suite's wall-clock cost) that this
investigation found to be real, not hypothetical.

## Amendment: a real finding from this task's own first execution -- `local-qwen3` is not reliably grounded in a ticket's actual content for `retrieve_ticket_kb_evidence`, against a KB-rich environment

This task's standalone runner (`apps/api/src/resolvegrid_api/diagnostic_eval.py`,
`eval/golden/diagnostic_v1.jsonl`) was run for real against this dev
environment's live stack, which turned out to already have a large,
previously-ingested KB corpus heavily covering Kestrel VPN policy topics
(multiple document versions, many chunks). Running the 6 real golden cases
surfaced a genuine, reproducible specialist-level behavior, not a harness
bug:

- For several cases whose ticket subject was deliberately NOT VPN-related
  (a broken coffee machine, a reopened printer-firmware ticket, a locked
  email inbox), `local-qwen3` still frequently produced a `retrieve_ticket_
  kb_evidence` query (and a final answer) about Kestrel VPN password
  resets -- a fixed, dominant-corpus-topic answer, not one grounded in the
  real ticket content actually given to it (the ticket's subject line is in
  the specialist's own opening prompt; the model simply did not use it
  reliably for this specific tool's query argument).
- This reproduced across re-runs even after strengthening ticket
  body/subject wording and the competing document's own distinctive
  phrasing twice (see `diagnostic_eval.py`'s per-case builder comments and
  this file's own git history for the two real, failed remediation
  attempts) -- a genuine qwen3:14b behavioral limitation in a KB-rich
  environment, consistent with ADR 0003/0004's own prior findings that this
  specific local model has real, reproducible tool-use reliability gaps
  under this codebase's real conditions (re-invoking already-answered
  tools, breaking on `think=True`, etc.), not something further prompt
  -engineering within this task's own scope reliably fixed.

**Real measured result** (one real run, `apps/api/tests/test_diagnostic_eval
.py`): 6/6 cases held the SAFETY guarantee (`forbidden_actions` +
`grade_no_mutation_claim_in_text`, zero-tolerance) -- the specialist never
once, across any case, claimed or attempted a mutating action. Only 2/6
cases held the fuzzy content-QUALITY guarantee (correct tool selection +
the judge's correctness/groundedness verdict) on that run.

**Resulting design decision** (a real correction to this task's own first
draft, documented honestly rather than hidden): `diagnostic_eval.
grade_diagnostic_case` now reports two independent outcomes per case,
`safety_passed` and `quality_passed`, rather than one blended
pass/fail. `test_diagnostic_eval_suite_real_end_to_end_against_live_stack`
asserts `safety_passed` zero-tolerance (matching Phase 10's own
`adversarial`-dimension convention for genuine safety properties) and only
REPORTS `quality_passed` (matching Phase 10's own convention that its
non-adversarial golden dimensions -- chat/retrieval/tool/approval -- are
measured against reported floors, not asserted at 100%; this diagnostic
golden set is exactly that kind of dimension, not an adversarial one).
This is the methodologically correct split, not a loosened bar to make a
red test green: the thing Task 5 most needs to guarantee about this
specialist -- it never mutates anything it claims or attempts to -- holds
with real, repeated, zero-tolerance evidence; the thing that is genuinely,
measurably unreliable in this environment (content grounding against a
KB-rich corpus) is reported honestly as a real, open finding for a future
phase's prompt-engineering work on `diagnostic_agent.py`'s instruction/tool
-selection behavior, not something this eval task is scoped to fix.

## Future work, explicitly deferred (not forgotten)

If diagnostic evals grow beyond a small, human-curated regression set (for
example, if a future phase wants per-commit dashboard tracking of diagnostic
quality alongside the other four dimensions), the natural follow-up is a
real `_run_diagnostic_case`/`_grade_diagnostic_case` addition to
`eval_worker.py` with its own deliberately-budgeted case count and CI-timing
plan -- revisit then, with real measured timings from this task's standalone
runner as the starting budget estimate, rather than guessing now.

A second, real follow-up this task's own execution surfaced (see the
amendment above): `diagnostic_agent.py`'s `_INSTRUCTION`/prompt could be
strengthened to more forcefully require the model to derive `retrieve_
ticket_kb_evidence`'s query from the ticket's OWN given subject/content
rather than defaulting to a dominant-corpus topic, and/or `diagnose_
ticket`'s opening prompt could include more of the ticket's own real
content (today it only passes subject/type/priority/status, never the
`TicketMessage` body -- `get_ticket_state_and_history` does not return
message bodies either, so the model's only view of the actual described
issue beyond the subject line is whatever it independently retrieves).
Fixing this is real prompt-engineering/tool-design work belonging to Tasks
3/4's `diagnostic_agent.py`/`diagnostic_tools.py`, not this eval task --
recorded here as a concrete, evidence-backed lead for whoever picks it up,
with this task's own `quality_passed` measurement as the honest baseline to
improve against.
