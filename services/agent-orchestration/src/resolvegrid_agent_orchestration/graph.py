"""The Phase 6 agent graph: classify_intent -> compose_response -> finalize.

Dependency-direction note (read this before adding an import to this
module): `services/agent-orchestration` is a workspace *library* that
`apps/api` depends on (see `apps/api/pyproject.toml`'s
`resolvegrid-agent-orchestration` dependency, added in Phase 6 Task 1).
`apps/api/src/resolvegrid_api/llm_gateway.py` lives in the *app*, not a
shared package. If this module imported `resolvegrid_api.llm_gateway`
directly, the dependency graph would be circular in spirit (an app
importing a library that imports back into the app) and this package
would gain a hard dependency on `resolvegrid_api` -- which doesn't even
exist from this package's workspace position without also depending on
FastAPI, SQLAlchemy, alembic, etc., none of which a pure orchestration
graph needs.

Instead, every node here is built by a factory function that takes a
plain callable, `CompleteFn = Callable[[str], str]` -- prompt text in,
completion text out. `apps/api` is the one that knows about
`llm_gateway.complete()` and its richer `CompletionResult` (tokens,
latency, provider, fallback info); Task 3 wires it in with something
like `complete_fn = lambda prompt: llm_gateway.complete(prompt).text`
when it calls `build_graph(checkpointer, complete_fn)`. This package
never imports `resolvegrid_api` and has no knowledge of
`CompletionResult` at all -- it only needs response *text*. That also
makes every node here trivially unit-testable with a bare lambda/fake,
no mocking of `apps/api` internals required.
"""

import json
from typing import Callable

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from resolvegrid_agent_orchestration.state import AgentState

# A completion function: takes a prompt string, returns the model's raw
# response text. Deliberately the narrowest possible interface -- see the
# module docstring for why this package doesn't touch `CompletionResult`.
CompleteFn = Callable[[str], str]

_VALID_INTENTS = {"general_question", "greeting", "unclear"}
_VALID_RISK_LEVELS = {"low", "medium", "high"}

_FALLBACK_MESSAGE = (
    "Sorry, something went wrong while processing your request. Please try again."
)

_CLASSIFY_PROMPT_TEMPLATE = """Classify the following user message.

Respond with ONLY a JSON object of exactly this shape, no other text, no markdown fences:
{{"intent": "<one of: general_question, greeting, unclear>", "risk_level": "<one of: low, medium, high>"}}

User message:
{input_text}
"""

_COMPOSE_PROMPT_TEMPLATE = """You are a helpful internal IT service-desk assistant. \
Answer the user's message clearly and concisely. There is no ticket/company-specific \
knowledge base available yet -- answer from general knowledge only, and say so if the \
question clearly requires company-specific information you don't have.

Classified intent: {intent} (risk_level: {risk_level})

User message:
{input_text}
"""


class IntentClassification(BaseModel):
    intent: str
    risk_level: str


def make_classify_intent_node(complete_fn: CompleteFn):
    """Build the `classify_intent` node bound to a given completion function.

    Prompts for a JSON classification, parses it with `json.loads`, and
    validates its shape with `IntentClassification`. Any failure along
    that path (malformed JSON, wrong shape, unrecognized intent/risk_level
    value) degrades softly to `intent="unclear", risk_level="low"` rather
    than raising -- a classification node failing softly is more honest
    than making the whole chat feature fragile against LLM output
    formatting variance.
    """

    def classify_intent(state: AgentState) -> dict:
        prompt = _CLASSIFY_PROMPT_TEMPLATE.format(input_text=state["input_text"])
        intent = "unclear"
        risk_level = "low"
        try:
            raw_text = complete_fn(prompt)
            parsed = json.loads(raw_text)
            classification = IntentClassification.model_validate(parsed)
            if classification.intent in _VALID_INTENTS:
                intent = classification.intent
            if classification.risk_level in _VALID_RISK_LEVELS:
                risk_level = classification.risk_level
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass
        return {"intent": intent, "risk_level": risk_level}

    return classify_intent


def make_compose_response_node(complete_fn: CompleteFn):
    """Build the `compose_response` node bound to a given completion function.

    Calls `complete_fn` again with the original input text (plus the
    classified intent/risk_level as context) to produce the actual answer.
    Any failure here is recorded on `state["error"]` rather than raised,
    so `finalize` can still produce a clean fallback message instead of
    the whole graph run blowing up.
    """

    def compose_response(state: AgentState) -> dict:
        prompt = _COMPOSE_PROMPT_TEMPLATE.format(
            intent=state.get("intent") or "unclear",
            risk_level=state.get("risk_level") or "low",
            input_text=state["input_text"],
        )
        try:
            text = complete_fn(prompt)
        except Exception as exc:  # noqa: BLE001 -- any completion failure, of
            # whatever concrete exception type `complete_fn` raises (this
            # package doesn't know or care -- see module docstring), should
            # degrade to a recorded error rather than crash the graph.
            return {"error": str(exc)}
        return {"output_text": text}

    return compose_response


def finalize(state: AgentState) -> dict:
    """Phase 6's "verify" stage is explicitly a no-op per the approved plan;
    `finalize` folds directly into it: pass through `compose_response`'s
    output, or substitute a safe fallback message if an earlier node
    recorded an error.
    """
    if state.get("error"):
        return {"output_text": _FALLBACK_MESSAGE}
    if state.get("output_text"):
        return {"output_text": state["output_text"]}
    return {"output_text": _FALLBACK_MESSAGE}


def build_graph(checkpointer, complete_fn: CompleteFn):
    """Build and compile the classify_intent -> compose_response -> finalize
    graph, wired to `checkpointer` for persistence and `complete_fn` for
    all LLM calls (see module docstring for why `complete_fn` is injected
    rather than this module importing `resolvegrid_api.llm_gateway`
    directly).
    """
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", make_classify_intent_node(complete_fn))
    builder.add_node("compose_response", make_compose_response_node(complete_fn))
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "compose_response")
    builder.add_edge("compose_response", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
