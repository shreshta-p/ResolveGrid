"""The agent graph: classify_intent -> retrieve -> compose_response -> finalize.

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
graph needs. The exact same reasoning applies to Phase 7's retrieval
pipeline (`resolvegrid_api.retrieval`/`retrieval_authz`, which themselves
import SQLAlchemy) -- see `RetrieveFn` below.

Instead, every node here is built by a factory function that takes a
plain callable, `CompleteFn = Callable[[str], str]` -- prompt text in,
completion text out. `apps/api` is the one that knows about
`llm_gateway.complete()` and its richer `CompletionResult` (tokens,
latency, provider, fallback info); Task 3 wires it in with something
like `complete_fn = lambda prompt: llm_gateway.complete(prompt).text`
when it calls `build_graph(checkpointer, complete_fn, retrieve_fn)`. This
package never imports `resolvegrid_api` and has no knowledge of
`CompletionResult` at all -- it only needs response *text*. That also
makes every node here trivially unit-testable with a bare lambda/fake,
no mocking of `apps/api` internals required.

Phase 7 Task 7 -- retrieval wiring and its scope limit
--------------------------------------------------------------------------
A new `retrieve` node sits between `classify_intent` and `compose_response`,
calling an injected `RetrieveFn` (same DI pattern as `CompleteFn`) to fetch
hybrid-search chunks and a deterministic sufficiency verdict (both computed
in `apps/api`, via `resolvegrid_api.retrieval`). `compose_response` then
branches its prompt on `retrieval_sufficient`: sufficient -> build a
citation-ready context block from `retrieved_chunks` and instruct the model
to answer from it, citing `[chunk:<id>]` inline; insufficient/empty ->
fall back to the original general-knowledge-only prompt.

Deliberate scope limit: this is NOT the plan doc's full "sufficient ->
answer with citations, insufficient -> abstain/clarify" routing. There is
no separate abstain/clarify graph branch, node, or forced refusal message
here -- an insufficient retrieval result degrades to *today's* Phase 6
general-knowledge answer, not a hard "I don't know" abstention. Rationale:
`assess_sufficiency`'s bar (see `resolvegrid_api/retrieval.py`) answers
"is there company-specific evidence worth citing," not "is this question
answerable at all" -- plenty of legitimate questions (e.g. "what's 2+2")
have no KB match and are still answerable from general knowledge. Gating
*whether to cite* on sufficiency, while leaving *whether to answer*
ungated, is judged the more useful behavior for this phase. A stricter
policy (e.g. always abstain when `risk_level` is high AND retrieval is
insufficient) is real future work once more of stage 11/12's upstream
verification/risk signals exist -- tracked as a documented gap, not
silently skipped.
"""

import json
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from resolvegrid_agent_orchestration.state import AgentState, RetrievedChunk

# A completion function: takes a prompt string, returns the model's raw
# response text. Deliberately the narrowest possible interface -- see the
# module docstring for why this package doesn't touch `CompletionResult`.
CompleteFn = Callable[[str], str]


class RetrievalOutcome(TypedDict):
    """Return shape `RetrieveFn` must produce -- exactly what the
    `retrieve` node attaches to state (`retrieved_chunks`,
    `retrieval_sufficient`).
    """

    chunks: list[RetrievedChunk]
    sufficient: bool


# A retrieval function: takes the query text and the opaque
# `retrieval_scope` dict already sitting in state (see state.py's module
# docstring for why it's a plain dict, not a typed `AuthzFilter`), returns
# a `RetrievalOutcome`. Unlike `CompleteFn`, this isn't shrunk down to a
# single bare value -- retrieval's genuinely graph-relevant signal *is*
# structured (which chunks, for `compose_response`'s citation context; was
# retrieval judged sufficient, for its prompt-branch decision), so both
# are kept rather than stripped the way `CompletionResult`'s token
# counts/provider/latency are for `CompleteFn`.
RetrieveFn = Callable[[str, dict | None], RetrievalOutcome]

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

_COMPOSE_PROMPT_NO_CONTEXT_TEMPLATE = """You are a helpful internal IT service-desk assistant. \
Answer the user's message clearly and concisely. No relevant company-specific knowledge-base \
article was found for this question -- answer from general knowledge only, and say so if the \
question clearly requires company-specific information you don't have.

Classified intent: {intent} (risk_level: {risk_level})

User message:
{input_text}
"""

_COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE = """You are a helpful internal IT service-desk assistant. \
Answer the user's question using the knowledge-base context below. Each context entry is \
labeled with a citation id in the form [chunk:<id>] and the title of the source document it \
came from -- when you use information from a specific entry, cite it inline using that exact \
bracketed id (for example: "...per the VPN policy [chunk:123]."). Only rely on context that is \
actually relevant to the question; if the context doesn't fully answer it, say so honestly \
rather than inventing details beyond what the context supports.

Classified intent: {intent} (risk_level: {risk_level})

Knowledge-base context:
{context_block}

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


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as citation-ready prompt context -- each
    entry prefixed with `[chunk:<id>] (from "<document title>"):` so the
    model has a concrete, copyable citation token per chunk, and so a
    citation in the model's answer can later be mapped back to a real
    `Chunk.id` (see `state.py`'s `RetrievedChunk` docstring).
    """
    return "\n\n".join(
        f'[chunk:{chunk["chunk_id"]}] (from "{chunk["document_title"]}"):\n{chunk["text"]}'
        for chunk in chunks
    )


def make_compose_response_node(complete_fn: CompleteFn):
    """Build the `compose_response` node bound to a given completion function.

    Calls `complete_fn` again with the original input text (plus the
    classified intent/risk_level as context) to produce the actual answer.
    Any failure here is recorded on `state["error"]` rather than raised,
    so `finalize` can still produce a clean fallback message instead of
    the whole graph run blowing up.

    Phase 7 Task 7: branches its prompt on `state["retrieval_sufficient"]`
    -- see this module's docstring for the documented scope limit on how
    far this drives abstention (it doesn't; insufficient retrieval falls
    back to the original general-knowledge prompt, not a refusal).
    """

    def compose_response(state: AgentState) -> dict:
        chunks = state.get("retrieved_chunks") or []
        if state.get("retrieval_sufficient") and chunks:
            prompt = _COMPOSE_PROMPT_WITH_CONTEXT_TEMPLATE.format(
                intent=state.get("intent") or "unclear",
                risk_level=state.get("risk_level") or "low",
                context_block=_build_context_block(chunks),
                input_text=state["input_text"],
            )
        else:
            prompt = _COMPOSE_PROMPT_NO_CONTEXT_TEMPLATE.format(
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


def make_retrieve_node(retrieve_fn: RetrieveFn):
    """Build the `retrieve` node bound to a given retrieval function.

    Calls `retrieve_fn(input_text, retrieval_scope)` and attaches the
    result to state as `retrieved_chunks`/`retrieval_sufficient`. Any
    failure degrades softly to "no chunks, not sufficient" (mirroring
    `classify_intent`'s soft-degrade philosophy) rather than raising -- a
    retrieval backend outage (DB down, embedding service unreachable)
    should degrade the chat to a general-knowledge answer via
    `compose_response`'s fallback branch, not take the whole endpoint
    down. This node's only job is populating state; `compose_response` is
    what decides how to use it.
    """

    def retrieve(state: AgentState) -> dict:
        try:
            outcome = retrieve_fn(state["input_text"], state.get("retrieval_scope"))
            chunks = outcome.get("chunks") or []
            sufficient = bool(outcome.get("sufficient"))
        except Exception:  # noqa: BLE001 -- any retrieval failure, of
            # whatever concrete exception type the injected `retrieve_fn`
            # raises (this package doesn't know or care what backend it
            # calls), should degrade softly rather than crash the graph.
            chunks = []
            sufficient = False
        return {"retrieved_chunks": chunks, "retrieval_sufficient": sufficient}

    return retrieve


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


def build_graph(checkpointer, complete_fn: CompleteFn, retrieve_fn: RetrieveFn):
    """Build and compile the classify_intent -> retrieve -> compose_response
    -> finalize graph, wired to `checkpointer` for persistence,
    `complete_fn` for all LLM calls, and `retrieve_fn` for knowledge
    retrieval (see module docstring for why both are injected rather than
    this module importing `resolvegrid_api` directly).
    """
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", make_classify_intent_node(complete_fn))
    builder.add_node("retrieve", make_retrieve_node(retrieve_fn))
    builder.add_node("compose_response", make_compose_response_node(complete_fn))
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "retrieve")
    builder.add_edge("retrieve", "compose_response")
    builder.add_edge("compose_response", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
