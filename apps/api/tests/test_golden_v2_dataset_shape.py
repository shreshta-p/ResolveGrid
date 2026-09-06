"""Phase 10 Task 5 shape test: the re-expressed/new `eval/golden/v2/*.jsonl`
golden dataset loads cleanly into `EvalCase` and satisfies the exit
criteria this task's plan section requires.

Covers `eval/golden/v2/chat_v1.jsonl` (Phase 6 chat cases re-expressed),
`eval/golden/v2/retrieval_v1.jsonl` (Phase 7 retrieval cases re-expressed,
including the VPN v1/v2 distractor and authz-leakage cases), and
`eval/golden/v2/tools_approvals_v1.jsonl` (new Phase 9 tool/approval
cases). Deliberately does NOT touch `eval/golden/phase6_chat_v1.jsonl` or
`eval/golden/phase7_retrieval_v1.jsonl` -- those remain untouched source
material for `test_eval_retrieval.py`/the classify_intent golden test.
"""

from pathlib import Path

from resolvegrid_evaluation.schema import EvalCase, load_eval_cases

_V2_DIR = Path(__file__).resolve().parents[3] / "eval" / "golden" / "v2"

_CHAT_PATH = _V2_DIR / "chat_v1.jsonl"
_RETRIEVAL_PATH = _V2_DIR / "retrieval_v1.jsonl"
_TOOLS_APPROVALS_PATH = _V2_DIR / "tools_approvals_v1.jsonl"


def _load_all() -> tuple[list[EvalCase], list[EvalCase], list[EvalCase]]:
    return (
        load_eval_cases(_CHAT_PATH),
        load_eval_cases(_RETRIEVAL_PATH),
        load_eval_cases(_TOOLS_APPROVALS_PATH),
    )


def test_v2_files_parse_cleanly_into_eval_cases():
    """Explicit sanity check that `load_eval_cases` parses all three new
    files without error -- each returns a non-empty list of real `EvalCase`
    instances (frozen pydantic models, so construction itself validates
    every Literal field)."""
    chat_cases, retrieval_cases, tools_approvals_cases = _load_all()

    assert len(chat_cases) == 18
    assert all(isinstance(c, EvalCase) for c in chat_cases)

    assert len(retrieval_cases) == 18
    assert all(isinstance(c, EvalCase) for c in retrieval_cases)

    assert len(tools_approvals_cases) == 3
    assert all(isinstance(c, EvalCase) for c in tools_approvals_cases)


def test_every_v2_case_is_human_review_approved():
    """Fails loudly if a 'draft' (or 'rejected') case is ever checked in --
    every shipped v2 case must be `human_review_status == 'approved'`, per
    this task's exit criterion."""
    chat_cases, retrieval_cases, tools_approvals_cases = _load_all()
    all_cases = chat_cases + retrieval_cases + tools_approvals_cases

    not_approved = [c.case_id for c in all_cases if c.human_review_status != "approved"]
    assert not_approved == [], f"non-approved v2 cases found: {not_approved}"


def test_every_required_dimension_has_at_least_one_case():
    """At this point in Phase 10 (Task 5, before Task 6's adversarial
    cases), exactly the chat/retrieval/tool/approval dimensions must each
    have at least one case across the three v2 files."""
    chat_cases, retrieval_cases, tools_approvals_cases = _load_all()
    all_cases = chat_cases + retrieval_cases + tools_approvals_cases

    dimensions_present = {c.dimension for c in all_cases}
    for required_dimension in ("chat", "retrieval", "tool", "approval"):
        assert required_dimension in dimensions_present, (
            f"no case with dimension={required_dimension!r} found across v2 files"
        )
        count = sum(1 for c in all_cases if c.dimension == required_dimension)
        assert count >= 1


def test_case_ids_are_globally_unique_across_all_three_files():
    chat_cases, retrieval_cases, tools_approvals_cases = _load_all()
    all_cases = chat_cases + retrieval_cases + tools_approvals_cases

    case_ids = [c.case_id for c in all_cases]
    duplicates = {case_id for case_id in case_ids if case_ids.count(case_id) > 1}
    assert duplicates == set(), f"duplicate case_ids found: {duplicates}"
    assert len(case_ids) == len(set(case_ids))


def test_retrieval_file_preserves_distractor_and_authz_leakage_cases():
    """Security-critical regression guard: the VPN v1/v2 distractor cases
    and the authz-leakage (`must_not_appear`) cases must survive the
    re-expression from `phase7_retrieval_v1.jsonl` into the new `EvalCase`
    shape, not get silently dropped."""
    _, retrieval_cases, _ = _load_all()

    distractor_cases = [c for c in retrieval_cases if c.distractor]
    must_not_appear_cases = [c for c in retrieval_cases if c.must_not_appear]

    assert len(distractor_cases) >= 1
    assert len(must_not_appear_cases) >= 1


def test_tool_and_approval_cases_cover_both_approval_decision_branches():
    """Both the approve and reject branches of Phase 9's real VPN-grant
    flow need coverage, per this task's plan -- not just one happy path."""
    _, _, tools_approvals_cases = _load_all()

    tool_cases = [c for c in tools_approvals_cases if c.dimension == "tool"]
    approval_cases = [c for c in tools_approvals_cases if c.dimension == "approval"]

    assert len(tool_cases) >= 1
    assert any(c.expected_tool_name == "grant_vpn_access" for c in tool_cases)

    approved_cases = [c for c in approval_cases if c.expected_approval_decision == "approved"]
    rejected_cases = [c for c in approval_cases if c.expected_approval_decision == "rejected"]
    assert len(approved_cases) >= 1
    assert len(rejected_cases) >= 1
    assert any(c.expected_final_status == "approved" for c in approved_cases)
    assert any(c.expected_final_status == "rejected" for c in rejected_cases)
