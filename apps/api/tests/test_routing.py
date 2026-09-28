"""CI-safe unit test of the real model-routing DECISION LOGIC only (Phase 11
Task 4). Zero real HTTP calls, zero credentials needed -- `select_model_for_
risk_level` is a bare `str -> str` pure function with no I/O (see
`routing.py`'s module docstring).

This is deliberately NOT where Task 4's "nothing mocked" real-cloud-call
proof lives -- CI's `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are always empty
strings (`.github/workflows/ci.yml`), so a real Anthropic/OpenAI round trip
can never run here. That proof is a separate, real, LOCAL verification step
against this machine's real `.env` keys, documented as evidence in this
task's own report -- not a pytest file, per the plan doc's explicit
"CI-vs-local proof split" instruction.
"""

from resolvegrid_api import llm_gateway
from resolvegrid_api.routing import HIGH_RISK_MODEL, select_model_for_risk_level


def test_high_risk_routes_to_cloud_primary():
    assert select_model_for_risk_level("high") == "cloud-primary"
    assert select_model_for_risk_level("high") == HIGH_RISK_MODEL


def test_low_risk_routes_to_default_local_model():
    assert select_model_for_risk_level("low") == llm_gateway.DEFAULT_MODEL
    assert select_model_for_risk_level("low") == "local-qwen3"


def test_medium_risk_routes_to_default_local_model():
    # Deliberately not treated as a third tier -- see routing.py's module
    # docstring for why medium shares low's zero-cost-default outcome.
    assert select_model_for_risk_level("medium") == llm_gateway.DEFAULT_MODEL


def test_unrecognized_risk_level_degrades_to_default_local_model_rather_than_raising():
    # An unrecognized/malformed risk_level must never accidentally route to
    # the paid cloud model -- the safe degrade is the cheap default, same
    # direction classify_intent's own soft-degrade already takes for an
    # unparseable classification (see graph.py's IntentClassification
    # handling).
    assert select_model_for_risk_level("not-a-real-risk-level") == llm_gateway.DEFAULT_MODEL
    assert select_model_for_risk_level("") == llm_gateway.DEFAULT_MODEL
