"""Golden-set test for `classify_intent`'s parsing/validation path.

**What this test does and does not prove (read before extending it):**
This test loads `eval/golden/phase6_chat_v1.jsonl` -- a small, hand-written
set of realistic chat inputs, each paired with the intent/risk_level a
real classification of that input would plausibly produce -- and for
each case, mocks `complete_fn` to return a scripted JSON response
matching that case's expected classification. It then asserts
`classify_intent` (built via `make_classify_intent_node`) parses and
validates that response correctly and returns the expected fields.

This proves the golden-set harness and `classify_intent`'s JSON
parsing/`IntentClassification` validation logic work correctly across a
realistic variety of well-formed response shapes (all 3 valid intents,
all 3 valid risk levels, in combination). It does **not** evaluate LLM
judgment quality -- no real model is called here, and nothing in this
test proves a real LLM would actually classify any given golden input
the way its `expected_intent`/`expected_risk_level` says. That kind of
live judgment-quality evaluation is out of scope for this task ("this
task is scaffolding for that, not that itself" -- Phase 6 Task 6) and
would require a future phase to actually call a live model against each
golden case and grade the real output.
"""

import json
from pathlib import Path

from resolvegrid_agent_orchestration.graph import make_classify_intent_node

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[3] / "eval" / "golden" / "phase6_chat_v1.jsonl"
)

_BASE_STATE = {
    "thread_id": "golden-test",
    "principal_employee_id": None,
    "input_text": "placeholder",
    "intent": None,
    "risk_level": None,
    "output_text": None,
    "error": None,
}


def _load_golden_cases() -> list[dict]:
    cases = []
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def test_golden_file_exists_and_has_a_realistic_number_of_cases():
    cases = _load_golden_cases()
    assert 10 <= len(cases) <= 20
    for case in cases:
        assert "input" in case
        assert "expected_intent" in case
        assert "expected_risk_level" in case


def test_classify_intent_parses_scripted_response_for_every_golden_case():
    cases = _load_golden_cases()
    assert cases, "golden set must not be empty"

    for case in cases:
        expected_intent = case["expected_intent"]
        expected_risk_level = case["expected_risk_level"]

        # Mock complete_fn to return a scripted JSON response matching
        # this case's expected classification -- we are testing that
        # classify_intent's parsing/validation path correctly turns that
        # scripted response into the same fields, not that a real model
        # would produce this response for this input.
        def fake_complete(prompt: str, _intent=expected_intent, _risk=expected_risk_level) -> str:
            return json.dumps({"intent": _intent, "risk_level": _risk})

        node = make_classify_intent_node(fake_complete)
        state = {**_BASE_STATE, "input_text": case["input"]}
        result = node(state)

        assert result == {
            "intent": expected_intent,
            "risk_level": expected_risk_level,
        }, f"mismatch for golden case input={case['input']!r}"
