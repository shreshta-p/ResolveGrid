"""Real model-routing policy (Phase 11 Task 4).

Until this task, every real call path defaulted to `local-qwen3`
unconditionally (see `docs/superpowers/plans/2026-09-27-phase11-observability-
routing.md`'s "Context" section) -- the cloud fallback config in
`infra/litellm/config.yaml` (`cloud-primary` -> `cloud-fallback`) was real but
unreachable in practice. This module is the one place that decides WHEN a
real call should leave the zero-marginal-cost local model and pay for a real
cloud completion.

**Policy** (a real, justified design decision, not an arbitrary toggle):
`risk_level == "high"` -> `cloud-primary` (Anthropic Haiku, per
`infra/litellm/config.yaml`); anything else (`"low"`, `"medium"`, or an
unrecognized/missing value) -> `llm_gateway.DEFAULT_MODEL` (`local-qwen3`).
Cheap-by-default, cloud-for-high-risk is the plan's stated intent: "zero-
marginal-cost local inference" is the default, and cloud spend is reserved
for classifications that matter enough to justify real per-call cost. This
deliberately does NOT special-case `"medium"` -- a three-tier policy
(low/medium/high each routing differently) was considered and rejected as
unjustified complexity; `classify_intent` only ever produces one of exactly
three risk_level strings (see `graph.py`'s `_VALID_RISK_LEVELS`), and this
policy only needs a two-way split to satisfy the plan's exit criterion ("at
least one classified risk_level value results in a cloud-primary call... and
at least one other value stays on local-qwen3").

**Where this is called from, and why it's a pure function**: this is called
from `model_call_logging.py`'s `make_compose_routing_complete_fn` (the real
closure `main.py` builds for `compose_response`'s `ComposeCompleteFn`
parameter -- see `graph.py`'s `ComposeCompleteFn` docstring for why
`compose_response` is the one node that receives `risk_level` as a real call
argument, not just `classify_intent`'s output baked into a prompt string).
Kept here, in its own module, as a bare `str -> str` function with zero
I/O and zero dependencies beyond `llm_gateway.DEFAULT_MODEL` specifically so
it is trivially unit-testable with zero real HTTP calls and zero
credentials -- this is the CI-safe half of Task 4's required "CI-vs-local
proof split" (see `apps/api/tests/test_routing.py`); the real, cloud-reaching
half (an actual `ModelCall` row with `provider="anthropic"`) is proven
separately, as real local evidence, since CI's `ANTHROPIC_API_KEY`/
`OPENAI_API_KEY` are always empty strings (`.github/workflows/ci.yml`).
"""

from resolvegrid_api import llm_gateway

# The LiteLLM `model_name` (from `infra/litellm/config.yaml`) that a
# `risk_level == "high"` classification routes to. Not `DEFAULT_MODEL` --
# that constant is deliberately the *fallback* value this policy routes
# everything else to, so this needs its own name rather than reusing/
# repurposing an existing "default" constant that means the opposite thing.
HIGH_RISK_MODEL = "cloud-primary"


def select_model_for_risk_level(risk_level: str) -> str:
    """Return the LiteLLM `model_name` a real completion should target,
    given a classified `risk_level` ("low" | "medium" | "high", or any
    other/unrecognized string -- see module docstring for why unrecognized
    values degrade to the same safe, zero-cost default as "low"/"medium"
    rather than raising).
    """
    if risk_level == "high":
        return HIGH_RISK_MODEL
    return llm_gateway.DEFAULT_MODEL
