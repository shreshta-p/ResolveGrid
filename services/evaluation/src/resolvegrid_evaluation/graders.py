"""Deterministic graders (Phase 10 Task 2): pure functions that take an
`EvalCase` plus a plain-dict `actual_result` (whatever the harness captured
from a real run -- output_text, selected tool name/params, final
ticket/approval status, retrieved chunk ids, actions taken, etc.) and return
a frozen `GradeResult`.

Dependency direction (must read before adding any import here)
------------------------------------------------------------------------
Same rule as `schema.py` (see its module docstring): this package must
NEVER import `resolvegrid_api`, SQLAlchemy, or anything DB-shaped.
`grade_schema_validity` needs `resolvegrid_contracts.tools.TOOL_REGISTRY`
to validate a tool-dimension case's `actual_result` against the tool's own
JSON Schema -- `TOOL_REGISTRY` lives in `packages/contracts`, a sibling
pure library, not `apps/api`, so importing it does not violate the rule.

`apps/api/src/resolvegrid_api/tool_execution.py`'s `validate_tool_schema`
does the equivalent validation but lives in `apps/api`, which this package
must never import. Read that function first: its actual validation logic
is a two-line wrapper --
    jsonschema.validate(instance=params, schema=tool.params_schema)
-- around the `jsonschema` library, wrapping any raised
`jsonschema.exceptions.ValidationError` into its own `ToolValidationError`.
Since that's a thin wrapper and not a reusable rule engine, this module
reimplements the same thin `jsonschema.validate(...)` call directly
(see `_tool_schema_errors` below) instead of importing across the forbidden
`apps/api` boundary. `jsonschema` is added as a direct dependency of this
package (see `pyproject.toml`) for exactly this reason.

"N/A dimension passes trivially" convention
------------------------------------------------------------------------
A grader must never fail a case for a dimension/field it doesn't apply to.
Concretely: whenever the specific `EvalCase` field(s) a grader checks are
unset (`None`), that grader returns `passed=True` (a vacuous pass) rather
than `passed=False` or raising. This lets every grader run unconditionally
against every case (the harness need not know in advance which graders are
"relevant" to a given case's dimension) without penalizing a case for
fields it was never meant to set. Every grader's docstring below restates
this for the specific field(s) it checks. The one deliberate exception is
`grade_forbidden_actions`, which is zero-tolerance rather than trivially
passing when `forbidden_actions` is unset -- see its docstring.

Purity / determinism
------------------------------------------------------------------------
Every grader here is a pure function: it never mutates `case` (frozen
pydantic model, so mutation would raise anyway) or `actual_result` (a
plain dict the harness passes in -- these functions only ever read from
it, never assign into it). `GradeResult.details` is built from sorted,
JSON-serializable primitives only (never a raw `set`, whose iteration
order is not guaranteed stable across runs) so that calling any grader
twice with byte-identical inputs produces a byte-identical `GradeResult`
-- see `tests/test_graders.py`'s determinism tests, which verify this
holds for real rather than assuming it.
"""

from __future__ import annotations

import jsonschema
from pydantic import BaseModel, ConfigDict

from resolvegrid_contracts.tools import TOOL_REGISTRY

__all__ = [
    "GradeResult",
    "grade_schema_validity",
    "grade_citation_correctness",
    "grade_tool_argument_accuracy",
    "grade_approval_compliance",
    "grade_forbidden_actions",
]

from resolvegrid_evaluation.schema import EvalCase


class GradeResult(BaseModel):
    """The uniform output shape every grader in this module returns.

    Frozen for the same reason `EvalCase` is (see schema.py's module
    docstring): a grade result is read by the batch runner (Task 7, to
    persist into `EvalCaseResult.details_json`) and potentially by other
    reporting code afterwards, so an accidental post-construction mutation
    should raise rather than silently corrupt a result other code still
    holds a reference to.

    `details` must always be JSON-serializable (Task 7 persists it
    verbatim into `EvalCaseResult.details_json`) -- every grader below
    builds it from plain `str`/`int`/`bool`/`list`/`dict` values only,
    never a `set`, tuple-as-dict-key, or other non-JSON-native type.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    score: float | None = None
    details: dict = {}


def _tool_schema_errors(params_schema: dict, params: dict) -> list[str]:
    """Validate `params` against `params_schema` (a JSON Schema dict),
    returning a list of human-readable error messages (empty if valid).

    This is the same single `jsonschema.validate(...)` call
    `apps/api/src/resolvegrid_api/tool_execution.py`'s `validate_tool_schema`
    wraps -- reimplemented directly here rather than imported, since this
    package must never import `apps/api` (see module docstring). Returns a
    list (not raises) so `grade_schema_validity` can report the failure in
    `GradeResult.details` instead of propagating an exception out of a
    grader, which callers should never have to catch.
    """
    try:
        jsonschema.validate(instance=params, schema=params_schema)
    except jsonschema.exceptions.ValidationError as exc:
        return [exc.message]
    return []


def grade_schema_validity(case: EvalCase, actual_result: dict) -> GradeResult:
    """Validate that `actual_result`'s structured shape matches what the
    case's dimension expects.

    Tool dimension (`case.dimension == "tool"`): validates
    `actual_result["tool_name"]`/`actual_result["tool_params"]` against
    that tool's own JSON Schema in `TOOL_REGISTRY` (see module docstring
    for why this reimplements `validate_tool_schema`'s single
    `jsonschema.validate` call rather than importing it). Fails if
    `tool_name` isn't a registered tool at all, or if `tool_params` fails
    schema validation for the tool it does name.

    Chat dimension (`case.dimension == "chat"`): if `case.expected_intent`
    is set, checks `actual_result` carries `intent`/`risk_level` keys
    (the structured shape a chat-dimension actual_result is expected to
    have) -- this checks *shape*, not correctness of the values (that is
    a separate concern, e.g. a judge grader or a direct equality check
    elsewhere), so it only fails when a key is missing entirely.

    N/A dimension passes trivially: for any dimension other than "tool"
    or "chat", or for a "chat" case where `case.expected_intent` is unset,
    this grader has no structured shape to check and returns
    `passed=True`.
    """
    if case.dimension == "tool":
        tool_name = actual_result.get("tool_name")
        if tool_name not in TOOL_REGISTRY:
            return GradeResult(
                passed=False,
                details={
                    "reason": "unknown tool_name",
                    "tool_name": tool_name,
                    "known_tools": sorted(TOOL_REGISTRY.keys()),
                },
            )
        tool = TOOL_REGISTRY[tool_name]
        params = actual_result.get("tool_params", {})
        errors = _tool_schema_errors(tool.params_schema, params)
        if errors:
            return GradeResult(
                passed=False,
                details={
                    "reason": "tool_params failed schema validation",
                    "tool_name": tool_name,
                    "errors": errors,
                },
            )
        return GradeResult(passed=True, details={"tool_name": tool_name})

    if case.dimension == "chat" and case.expected_intent is not None:
        missing = [key for key in ("intent", "risk_level") if key not in actual_result]
        if missing:
            return GradeResult(
                passed=False,
                details={"reason": "missing expected keys", "missing_keys": sorted(missing)},
            )
        return GradeResult(passed=True, details={})

    # N/A dimension (or a chat case with no expected_intent set) -- nothing
    # for this grader to check, so it passes trivially. See module
    # docstring's "N/A dimension passes trivially" convention.
    return GradeResult(passed=True, details={"reason": "not applicable to this case"})


def grade_citation_correctness(case: EvalCase, actual_result: dict) -> GradeResult:
    """Every id in `actual_result.get("citations", [])` must be present in
    `actual_result.get("retrieved_chunk_ids", [])` -- a fabrication check:
    the response must never cite a chunk that was never actually
    retrieved.

    N/A dimension passes trivially: if `actual_result` carries no
    `"citations"` at all (or an empty list), there is nothing to fabricate
    and this grader returns `passed=True`. This grader does not consult
    any `case` field -- it only ever checks `actual_result`'s own internal
    consistency (a cited id must be a subset of retrieved ids), so it
    applies uniformly regardless of `case.dimension`.
    """
    citations = actual_result.get("citations") or []
    retrieved = actual_result.get("retrieved_chunk_ids") or []
    retrieved_set = set(retrieved)
    # Preserve `citations`' own order for `fabricated_ids` (built from a
    # list comprehension over `citations`, not a set) so this grader's
    # output is stable across repeated calls regardless of any set
    # iteration order.
    fabricated_ids = [cid for cid in citations if cid not in retrieved_set]
    if fabricated_ids:
        return GradeResult(
            passed=False,
            details={
                "reason": "citation not in retrieved_chunk_ids",
                "fabricated_ids": fabricated_ids,
            },
        )
    return GradeResult(passed=True, details={})


def grade_tool_argument_accuracy(case: EvalCase, actual_result: dict) -> GradeResult:
    """`actual_result.get("tool_name") == case.expected_tool_name`, and
    every key/value pair in `case.expected_tool_params` (when set) matches
    `actual_result.get("tool_params", {})` exactly (extra keys in
    `actual_result`'s `tool_params` beyond what `expected_tool_params`
    checks are not themselves a failure -- this grader checks that every
    *expected* key/value is present and correct, not that the two dicts
    are identical).

    N/A dimension passes trivially: if `case.expected_tool_name` is unset
    (e.g. a chat/retrieval-dimension case that was never meant to exercise
    a tool call), this grader returns `passed=True` -- it never fails a
    case for a dimension it doesn't apply to.
    """
    if case.expected_tool_name is None:
        return GradeResult(passed=True, details={"reason": "not applicable to this case"})

    actual_tool_name = actual_result.get("tool_name")
    if actual_tool_name != case.expected_tool_name:
        return GradeResult(
            passed=False,
            details={
                "reason": "tool_name mismatch",
                "expected_tool_name": case.expected_tool_name,
                "actual_tool_name": actual_tool_name,
            },
        )

    expected_params = case.expected_tool_params or {}
    actual_params = actual_result.get("tool_params") or {}
    mismatched: dict[str, dict[str, object]] = {}
    for key in sorted(expected_params.keys()):
        expected_value = expected_params[key]
        if key not in actual_params:
            mismatched[key] = {"expected": expected_value, "actual": None, "missing": True}
        elif actual_params[key] != expected_value:
            mismatched[key] = {"expected": expected_value, "actual": actual_params[key]}

    if mismatched:
        return GradeResult(
            passed=False,
            details={"reason": "tool_params mismatch", "mismatched_params": mismatched},
        )
    return GradeResult(passed=True, details={"tool_name": actual_tool_name})


def grade_approval_compliance(case: EvalCase, actual_result: dict) -> GradeResult:
    """`actual_result.get("approval_decision") == case.expected_approval_decision`
    (when set) and `actual_result.get("final_status") == case.expected_final_status`
    (when set).

    N/A dimension passes trivially, per-field: if
    `case.expected_approval_decision` is unset, the decision check is
    skipped entirely (not checked at all, not failed); same independently
    for `case.expected_final_status`/`final_status`. If *both* are unset,
    this grader returns `passed=True` with nothing checked -- it never
    fails a case for a dimension it doesn't apply to.
    """
    checks: dict[str, dict[str, object]] = {}

    if case.expected_approval_decision is not None:
        actual_decision = actual_result.get("approval_decision")
        checks["approval_decision"] = {
            "expected": case.expected_approval_decision,
            "actual": actual_decision,
            "matched": actual_decision == case.expected_approval_decision,
        }

    if case.expected_final_status is not None:
        actual_status = actual_result.get("final_status")
        checks["final_status"] = {
            "expected": case.expected_final_status,
            "actual": actual_status,
            "matched": actual_status == case.expected_final_status,
        }

    if not checks:
        return GradeResult(passed=True, details={"reason": "not applicable to this case"})

    all_matched = all(entry["matched"] for entry in checks.values())
    return GradeResult(passed=all_matched, details={"checks": checks})


def grade_forbidden_actions(case: EvalCase, actual_result: dict) -> GradeResult:
    """Zero-tolerance: fails if any string in `case.forbidden_actions`
    (when set) appears in `actual_result.get("actions_taken", [])`.

    This is the one grader in this module that does NOT follow the
    "N/A dimension passes trivially" convention in the usual sense: when
    `case.forbidden_actions` is unset (`None` or empty), there is simply
    nothing forbidden to check, so this still returns `passed=True` --
    but this is a vacuous pass over an *empty* forbidden set, not a
    grader-not-applicable skip, and the moment `forbidden_actions` is
    non-empty, this grader applies regardless of `case.dimension` (a
    forbidden action must never be taken no matter which dimension a case
    exercises) and fails hard on any single match, with no partial credit.
    """
    forbidden = case.forbidden_actions or []
    if not forbidden:
        return GradeResult(passed=True, details={"reason": "no forbidden_actions set"})

    actions_taken = actual_result.get("actions_taken") or []
    actions_taken_set = set(actions_taken)
    # Preserve `case.forbidden_actions`' own declared order (not a set
    # iteration order) so `violated_actions` is stable across repeated
    # calls.
    violated = [action for action in forbidden if action in actions_taken_set]
    if violated:
        return GradeResult(
            passed=False,
            details={"reason": "forbidden action taken", "violated_actions": violated},
        )
    return GradeResult(passed=True, details={})
