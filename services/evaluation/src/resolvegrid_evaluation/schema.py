"""`EvalCase`: one unified shape for every evaluation dimension this project
grades (chat, retrieval, tool, approval, adversarial), plus a JSONL loader.

Phase 10 Task 1 replaces the ad hoc, per-phase eval case shapes this
codebase accumulated (`resolvegrid_api.eval_retrieval.GoldenCase` for
Phase 7 retrieval cases, a bespoke inline shape for Phase 6 chat-intent
cases) with one schema covering every dimension plan.md's golden-dataset
section requires. Most fields are optional since a given case only
exercises some of them -- a `dimension="chat"` case sets
`expected_intent`/`expected_risk_level` and leaves the retrieval/tool/
approval fields `None`, a `dimension="retrieval"` case sets `relevant`/
`distractor`/`must_not_appear`/`authz` and leaves the rest `None`, etc.

Dependency direction (must read before adding any import here)
------------------------------------------------------------------------
This package (`services/evaluation`, importable as `resolvegrid_evaluation`)
is a pure library, mirroring the same rule already established for
`services/retrieval` and `services/agent-orchestration` (see their
`pyproject.toml` comments and `graph.py`'s module docstring): it must
NEVER import SQLAlchemy, `resolvegrid_api`, or anything DB-shaped. Only
`apps/api` may depend on `resolvegrid-evaluation`, never the reverse. This
is why `EvalCase` stores `relevant`/`distractor`/`must_not_appear` as
plain `(title, ordinal)` tuples (matching
`resolvegrid_api.eval_retrieval.GoldenCase`'s existing convention -- see
that module's docstring for why identity is keyed by
`(Document.title, Chunk.ordinal)` rather than a real DB id) instead of
resolving them to real `Chunk` rows here -- that resolution is
`apps/api`'s job, against whatever corpus is actually loaded in its DB.

Frozen, like `resolvegrid_contracts.tools.ToolContract`
------------------------------------------------------------------------
`EvalCase` is `frozen=True` for the same reason `ToolContract` is: a
golden/adversarial case is a fixed fixture, loaded once and then read by
many parts of the harness (graders, the batch runner, dashboard queries).
A frozen model turns any accidental post-construction mutation into a
raised `ValidationError` instead of silently corrupting a shared case
object other code still holds a reference to.

JSONL loading convention
------------------------------------------------------------------------
`load_eval_cases` mirrors `resolvegrid_api.eval_retrieval.load_golden_cases`:
one JSON object per non-blank line, parsed with `json.loads`, one
`EvalCase` per line, in file order.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Dimension = Literal["chat", "retrieval", "tool", "approval", "adversarial"]
ApprovalDecisionLiteral = Literal["approved", "rejected"]
JudgeDimension = Literal["groundedness", "correctness", "abstention_appropriateness"]
Provenance = Literal["hand_written", "model_drafted"]
HumanReviewStatus = Literal["draft", "approved", "rejected"]
# Matches `services/agent-orchestration/src/resolvegrid_agent_orchestration/
# graph.py`'s `_VALID_RISK_LEVELS = {"low", "medium", "high"}` -- the exact
# set `classify_intent` ever assigns and `eval/golden/phase6_chat_v1.jsonl`'s
# real cases already use for `expected_risk_level`. Reused verbatim for the
# cross-cutting `risk_level` field too, rather than inventing a second,
# looser vocabulary for what is the same underlying concept.
RiskLevel = Literal["low", "medium", "high"]
Difficulty = Literal["easy", "medium", "hard"]
# Matches `ApprovalRequest.status`'s value set
# (`apps/api/src/resolvegrid_api/models/approvals.py`) verbatim -- an
# approval-dimension case's expected final status is, by definition, one of
# the same four values that column can ever actually hold.
FinalStatus = Literal["pending", "approved", "rejected", "expired"]


class EvalCase(BaseModel):
    """One gradeable evaluation case, covering any combination of the
    chat/retrieval/tool/approval/adversarial dimensions.

    See module docstring for the frozen-instance rationale and the
    dependency-direction rule this package must never violate.
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity -----------------------------------------------------
    case_id: str  # stable, human-readable, e.g. "chat.greeting.001"
    dataset_version: str
    dimension: Dimension

    # --- Principal / input (shared across dimensions) ------------------
    # Identity/permissions fixture ref (role, department, entitlements)
    # -- enough for the harness to construct or look up a real
    # Principal/Employee. Not resolved to a real DB row here (dependency
    # direction rule) -- apps/api resolves it at run time.
    principal_fixture: dict | None = None
    input_text: str | None = None  # the request/context

    # --- Chat dimension (Phase 6 shape) --------------------------------
    expected_intent: str | None = None
    expected_risk_level: RiskLevel | None = None

    # --- Retrieval dimension (Phase 7 shape, reused verbatim) ----------
    answerable: bool | None = None
    # (Document.title, Chunk.ordinal) pairs -- see module docstring for
    # why identity is keyed this way rather than a real Chunk.id.
    relevant: list[tuple[str, int]] | None = None
    distractor: list[tuple[str, int]] | None = None
    must_not_appear: list[tuple[str, int]] | None = None
    authz: dict | None = None

    # --- Tool dimension -------------------------------------------------
    expected_tool_name: str | None = None
    expected_tool_params: dict | None = None

    # --- Approval dimension ---------------------------------------------
    expected_approval_decision: ApprovalDecisionLiteral | None = None
    expected_final_status: FinalStatus | None = None

    # --- Cross-cutting (any dimension) ----------------------------------
    # Must never be taken, regardless of dimension -- zero-tolerance,
    # checked by grade_forbidden_actions (Task 2).
    forbidden_actions: list[str] | None = None

    # --- Judge grading (only set when this case needs judge grading) ---
    rubric: str | None = None
    judge_dimensions: list[JudgeDimension] | None = None

    risk_level: RiskLevel | None = None
    difficulty: Difficulty | None = None

    # --- Provenance / review gate ----------------------------------------
    provenance: Provenance
    human_review_status: HumanReviewStatus

    note: str | None = None


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Parse a JSONL file of `EvalCase` records into `EvalCase` objects,
    one per non-blank line, in file order.

    Mirrors `resolvegrid_api.eval_retrieval.load_golden_cases`'s existing
    JSONL-parsing convention: open with utf-8, strip each line, skip blank
    lines, `json.loads` each remaining line. Unlike `load_golden_cases`
    (which resolves `authz` into a real `AuthzFilter` dataclass, a
    DB-adjacent concept `resolvegrid_api.retrieval_authz` owns), this
    loader keeps `authz`/`principal_fixture` as plain dicts -- resolving
    them to real objects is `apps/api`'s job, per this package's
    dependency-direction rule.

    No manual coercion of `relevant`/`distractor`/`must_not_appear` is
    needed here: Pydantic v2 already coerces a JSON list-of-lists (e.g.
    `[["title", 0]]`) directly into `list[tuple[str, int]]` on model
    construction, verified empirically against this exact field type --
    `EvalCase(**raw)` alone produces the identical tuples a hand-written
    conversion loop would, so that loop would have been untested,
    redundant code.
    """
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(EvalCase(**raw))
    return cases
