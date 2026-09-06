"""Thin adapters for adversarial `EvalCase`s that *reuse* an existing,
already-passing Phase 7/8/9 pytest assertion instead of re-implementing its
logic a second time (Phase 10 Task 6).

Why this module exists (reuse over reimplementation)
------------------------------------------------------------------------
plan.md's Task 6 requires 9 adversarial cases; 3 of them already have a
real, passing assertion elsewhere in this codebase:

- the conflicting/stale VPN v1/v2 policy pair (already covered by
  `apps/api/tests/test_eval_retrieval.py`'s zero-tolerance
  `any_distractor_beats_relevant is False` check),
- the simulated provider-outage/fallback case (already covered by
  `apps/api/tests/test_ticket_summarize.py`'s
  `assert_ticket_summarize_reports_fallback_and_records_model_call`,
  extracted from that file's own regression test for this exact purpose), and
- the duplicate/expired approval-replay cases (already covered by
  `apps/api/tests/test_mutation_execution.py`'s
  `assert_duplicate_replay_creates_no_second_grant_or_tool_call` and
  `assert_expired_approval_raises_and_blocks_execution`, extracted the
  same way).

Per this phase's plan ("Notes for whoever executes this plan" -- Task 6):
wrapping these as a SECOND, independent assertion mechanism risks the two
silently drifting apart over time (one gets updated, the other doesn't).
The correct fix, which this phase applied, is extracting each of those
three tests' core assertion into a plain callable that BOTH the original
regression test and the new adversarial case call directly -- so there is
exactly one place each assertion is written.

`apps/api/tests/test_adversarial_suite.py` is the only caller of the real,
DB/HTTP-touching extracted callables above (they live in `apps/api`, which
this package must never import -- see this module's dependency-direction
note below). It calls each one directly, catches whether it raised, and
packages the *outcome* into a plain, JSON-serializable dict (`actual_result
= {"passed": bool, ...whatever else is useful for debugging}`). This
module's job is only to turn that already-computed real outcome into the
harness's uniform `GradeResult` shape -- it performs no DB/HTTP work
itself, ever.

Dependency direction (must read before adding any import here)
------------------------------------------------------------------------
Same rule as `schema.py`/`graders.py` (see their module docstrings): this
package must NEVER import `resolvegrid_api`, SQLAlchemy, or anything
DB-shaped. This module in particular must never import
`apps/api/tests/test_ticket_summarize.py` or
`apps/api/tests/test_mutation_execution.py` directly -- doing so would
both violate that rule (those modules pull in SQLAlchemy/FastAPI) and
invert the intended calling direction (`apps/api` calls into
`resolvegrid_evaluation`, never the reverse). This module only ever
receives an already-built plain dict describing what happened; it never
reaches back into `apps/api` to compute it.
"""

from __future__ import annotations

from resolvegrid_evaluation.graders import GradeResult
from resolvegrid_evaluation.schema import EvalCase

__all__ = ["grade_reused_pytest_assertion"]


def grade_reused_pytest_assertion(case: EvalCase, actual_result: dict) -> GradeResult:
    """Uniform `GradeResult` adapter for an adversarial case that wraps an
    existing, already-passing Phase 7/8/9 pytest assertion rather than
    re-implementing it (see module docstring for which 3 of the 9
    adversarial cases this applies to).

    `actual_result` is expected to carry `{"passed": bool, ...}` -- the
    real outcome of calling the extracted assertion callable directly (or,
    for the conflicting/stale-policy reference case, of checking the
    referenced case_id's own data), computed entirely in `apps/api` before
    this function ever sees it. Every other key in `actual_result` is
    carried through into `GradeResult.details` verbatim, for debugging,
    with no reinterpretation -- this function does not know or care what
    those keys mean; it only reads `"passed"`.

    `case` is accepted (matching every other grader's signature in this
    package, see `graders.py`) but not consulted -- unlike the Task 2
    deterministic graders, which each check specific `EvalCase` fields
    against `actual_result`, this adapter's whole point is that the real
    check already happened elsewhere (inside the reused pytest assertion),
    so there is nothing left for this function to independently verify
    against `case`. Kept as a parameter anyway for a uniform `(case,
    actual_result) -> GradeResult` call signature across every grader in
    this harness, so a caller (e.g. a future batch runner) can call every
    case's grader the same way without special-casing reused cases.
    """
    passed = bool(actual_result.get("passed"))
    details = {key: value for key, value in actual_result.items() if key != "passed"}
    return GradeResult(passed=passed, details=details)
