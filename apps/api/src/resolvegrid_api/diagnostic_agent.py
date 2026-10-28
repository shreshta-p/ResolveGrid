"""Phase 13 Task 3: real Google ADK `Agent`/`Runner` wiring for the
diagnostic specialist (see `docs/superpowers/plans/2026-10-21-phase13-adk-
diagnostic-specialist.md`'s "Task 3" section and `docs/adr/0004-adk-
diagnostic-specialist-wiring.md` for the real decisions this module makes
and why).

Real currying finding (empirically verified against the installed
`google-adk==1.14.1` package's real source -- see ADR 0004's "Currying vs.
ToolContext" section for the exact commands/output this cites):

- ADK's native injected-parameter mechanism (`FunctionTool._ignore_params =
  ['tool_context', 'input_stream']`, confirmed by reading `google.adk.tools.
  function_tool.FunctionTool.__init__`) is a fixed, ADK-owned
  `ToolContext` object (invocation/session/memory/auth access) -- NOT a
  generic hook for injecting an arbitrary caller-supplied object like this
  codebase's own `Session`/`Principal`. It is the wrong mechanism for this
  module's problem and was rejected after reading its real source, not
  assumed unsuitable.
- `functools.partial(tool_fn, session, principal)` IS correctly introspected
  by `inspect.signature()` -- confirmed empirically: a real `inspect.
  signature()` call on a `functools.partial`-bound multi-argument function
  automatically drops the already-bound leading positional arguments and
  reports only the remaining ones (`ticket_id`, `limit`, ...), with zero
  manual `__signature__` surgery needed for THAT part. This is exactly what
  `google.adk.tools._automatic_function_calling_util.build_function_
  declaration` (via `FunctionTool._get_declaration`) calls to build the
  model-visible JSON schema.
- BUT a bare `functools.partial` object has no `__name__` (confirmed:
  `hasattr(partial_obj, '__name__')` is `False`) and its `__doc__` is the
  generic `functools.partial` docstring, not the wrapped function's --
  `FunctionTool.__init__` falls back to `func.__class__.__name__` ==
  `"partial"` for EVERY such tool (a real, confirmed tool-name collision
  across all 4 tools) and loses the real docstring the model needs to know
  when to call each tool. This is the real, correctable gap `_curry_tool`
  below fixes.
- This module uses a closure (not a bare `functools.partial`), for one
  additional real reason beyond the name/doc fix: it also needs to catch
  this module's own `DiagnosticToolError` subclasses and degrade them to a
  `{"error": ...}` dict ADK's own `FunctionTool.run_async` already returns
  for ITS missing-mandatory-argument case (see that method's real source) --
  letting an authz/not-found exception propagate unhandled out of a tool
  call would crash the ENTIRE multi-turn run over one adversarial/mistaken
  tool argument (e.g. a model-guessed out-of-scope `employee_id`), rather
  than letting the model see a safe, non-data-leaking denial and recover
  (Task 5's adversarial authz tests target exactly this path). A closure
  needs its OWN explicit `__signature__ = <original sig, minus session/
  principal>` assignment (confirmed empirically: a bare `def wrapper(*args,
  **kwargs)` closure's own signature is NOT automatically narrowed the way
  `functools.partial`'s is) -- `_curry_tool` sets this explicitly, verified
  against the real installed `FunctionTool` (its schema-declaration AND its
  `_get_mandatory_args()`/`valid_params` invocation-time filtering both read
  `inspect.signature(self.func)`, so one manual override is consistently
  honored everywhere ADK itself introspects this closure).

SessionService choice: `InMemorySessionService`, not `DatabaseSessionService`
-- see ADR 0004. This is a bounded, single-shot diagnostic request (not a
persistent multi-day conversation an analyst resumes later), so there is no
real need to pay `DatabaseSessionService`'s two confirmed real costs (ADR
0003 Finding 6): Alembic-bypassing auto-created tables, and permanently
orphaned `app_states`/`user_states` rows with no first-party cleanup API.

Model routing: reuses the existing, already-reviewed `resolvegrid_api.
routing.select_model_for_risk_level` (not a new, parallel routing
mechanism) -- a ticket's own `priority` is mapped to a `risk_level` string
(`"high"` for `high`/`urgent`/`critical` priority, `"medium"` otherwise;
see `_risk_level_for_ticket`), so a high-stakes ticket's diagnosis can route
to `cloud-primary` exactly like `compose_response`'s own real routing
policy does, while most diagnoses stay on the zero-marginal-cost
`local-qwen3` default.

Built PER CALL, not once at app startup (the Architecture decision section's
explicit "your call" point): the tool list MUST be curried with THIS
request's real `session`/`principal` (Task 2's docstring), which only exist
for the lifetime of one request -- there is no way to build a reusable,
app-lifetime `Agent` whose `tools=[...]` are bound to one specific request's
session. Combined with choosing `InMemorySessionService` (no DB connection
pool to reuse across calls, unlike `DatabaseSessionService`), there is no
real benefit left to splitting "build the model" from "build the agent"
across two different lifetimes -- building the whole `Agent`/`Runner`/
`InMemorySessionService` fresh per call is cheap (confirmed: no real network
calls happen during construction itself, only when `run_async` actually
runs) and mirrors Task 1's own spike shape (`_build_agent_and_plugin`/
`_run_spike`), which also builds everything fresh per call.
"""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from opentelemetry.trace import Tracer
from resolvegrid_retrieval.citation_verification import strip_fabricated_spans, verify_citations
from sqlalchemy.orm import Session

from resolvegrid_api import llm_gateway
from resolvegrid_api.diagnostic_tools import (
    DiagnosticToolError,
    find_similar_tickets,
    get_employee_entitlements,
    get_ticket_state_and_history,
    retrieve_ticket_kb_evidence,
)
from resolvegrid_api.model_call_logging import log_adk_model_call
from resolvegrid_api.routing import select_model_for_risk_level
from resolvegrid_authz import Principal

# The real tool name whose `function_response` event this module inspects
# to recover STRUCTURED evidence (chunk_id/document_title pairs) for the
# citations Task 4's endpoint surfaces to the UI -- see `DiagnosisResult.
# cited_chunks`'s docstring below for the full "why this is a Task 4
# addition, not something Task 3 already did" rationale.
_KB_EVIDENCE_TOOL_NAME = "retrieve_ticket_kb_evidence"

# Ticket.priority values (free-text column, no DB-level enum -- see
# models/ticketing.py) that map to risk_level "high" for `routing.select_
# model_for_risk_level`'s existing two-tier policy. "medium"/"low"/anything
# else degrades to risk_level "medium", same safe default `routing.py`'s own
# docstring documents for an unrecognized value.
_HIGH_PRIORITY_VALUES = {"high", "urgent", "critical"}

# Phase 13 Task 3's own real `ModelCall.purpose` value -- distinct from
# every existing value (`model_call_logging.py`'s module docstring's
# breadcrumb list), per this task's "a new, honestly-distinct purpose value
# ... not one fake aggregate row" requirement. One value, reused across
# every real internal call a single diagnostic run makes -- each call still
# gets its OWN `ModelCall` row (see `log_adk_model_call`'s docstring); the
# `purpose` string does not need a per-call suffix for that to hold, since
# row identity already comes from the row's own id/created_at/trace_id.
_MODEL_CALL_PURPOSE = "diagnostic.specialist_turn"

# See ADR 0004's "max_llm_calls" section for the real justification: ADR
# 0003 Finding 5 found ONE trivial, zero-argument tool could take up to 36
# real turns (qwen3's confirmed real tendency to re-invoke an
# already-answered tool) under Task 1's spike, even with an explicit
# "do not call it again" instruction already present. This specialist has
# FOUR real tools instead of one, each a real, independent opportunity for
# the same pathology to recur -- a naive 4x compounding of Task 1's observed
# worst case would be 4 * 36 = 144. 60 is chosen as a middle ground: well
# above Task 1's own single-tool cap (comfortably covering a well-behaved
# run that calls every tool at least once, plus some retry headroom), while
# staying far below the theoretical 144-call compounding worst case, so a
# genuinely pathological run still terminates in bounded time rather than
# silently approaching the library's own 500-call default. This module's own
# real end-to-end verification run's ACTUAL observed call count is recorded
# in ADR 0004 and the Task 3 completion report -- amend this constant if a
# real run is ever observed to approach it, per this phase's own "report
# honestly" standing directive.
_MAX_LLM_CALLS = 60

_INSTRUCTION = (
    "You are an IT-operations diagnostic specialist helping a human analyst "
    "investigate a support ticket. You have four READ-ONLY tools: "
    "find_similar_tickets, get_employee_entitlements, retrieve_ticket_kb_evidence, "
    "and get_ticket_state_and_history. Use them to gather real evidence before "
    "answering a question about this ticket -- never guess or invent ticket "
    "history, employee entitlements, or knowledge-base content.\n\n"
    "IMPORTANT CONSTRAINT, not an oversight: you have NO tool that performs any "
    "mutating action (granting access, closing a ticket, changing an "
    "entitlement, or anything else that changes real data). You may only "
    "SUGGEST such an action in your final answer (for example: 'consider "
    "granting VPN access to this employee'); any such action must be reviewed "
    "and invoked by the human analyst themselves, through the existing "
    "human-in-the-loop tool-invocation approval endpoint (an HTTP POST to the "
    "/tools/ path for that specific tool's name, followed by /invoke). NEVER "
    "claim or imply that you have already performed, granted, revoked, or "
    "changed anything yourself -- you are strictly an advisor, not an actor.\n\n"
    "Call each tool AT MOST ONCE per distinct question; once a tool has "
    "returned a result, do not call it again with the same (or equivalent) "
    "arguments. If a tool returns an {'error': ...} result, that means the "
    "lookup was denied or the referenced record doesn't exist -- do not retry "
    "it, and do not fabricate a substitute answer; simply note the limitation. "
    "Once you have gathered sufficient evidence, give a concise final "
    "diagnostic summary: what you found, citing any [chunk:<id>] evidence "
    "returned by retrieve_ticket_kb_evidence, and, if relevant, a suggested "
    "next step for the analyst to action themselves."
)


@dataclass(frozen=True)
class DiagnosisResult:
    """Real return shape of `diagnose_ticket` -- a plain, ADK-free value
    (`apps/api`'s own type, not re-exported from `google.adk`) so a future
    caller never needs to import ADK to consume it. Intentionally minimal:
    Task 3's own scope is "make the real callable's signature/return shape
    clean and ready for Task 4," not building the full structured
    (summary/evidence/next-steps) UI contract the plan's Task 4 section
    describes -- that's real prompt-engineering/parsing work left for Task 4,
    not invented here ahead of schedule.
    """

    ticket_id: int
    summary: str
    tool_calls_made: list[str] = field(default_factory=list)
    turn_count: int = 0
    model_calls_logged: int = 0
    # Phase 13 Task 4 addition (not present when Task 3 wrote this module --
    # see this class's own docstring above for why Task 3 deliberately left
    # this kind of parsing to Task 4): real, VERIFIED `{"chunk_id": int,
    # "document_title": str}` evidence the specialist actually cited in
    # `summary`, using the exact same real `resolvegrid_retrieval.
    # citation_verification.verify_citations` deterministic check `/chat`'s
    # `verify_citations_node` already uses (graph.py) -- never a new,
    # parallel citation-trust mechanism. `summary` itself has already had
    # any FABRICATED `[chunk:<id>]` marker stripped (same "strip, don't
    # abstain" consequence `verify_citations_node`'s docstring documents and
    # justifies) by the time a caller reads this field -- see
    # `diagnose_ticket`'s own docstring for the real mechanics (which ADK
    # event this is parsed from, and why only `retrieve_ticket_kb_evidence`'s
    # real tool results are eligible evidence).
    cited_chunks: list[dict] = field(default_factory=list)
    # Real, Task 6 fresh-state-verification finding (the robustness gap ADR
    # 0005's "fourth follow-up" flagged but deliberately left unfixed at the
    # time): `True` when this run's own ADK `Runner.run_async` loop had to be
    # stopped early because the model hallucinated a call to a tool name that
    # was never registered (confirmed real and reproducible against a
    # genuinely empty, freshly-migrated database -- not a one-off; observed
    # names include both "diagnostic_summary", ADR 0005's original finding,
    # and "diagnostic_specialist", a NEW variant this task's own fresh-state
    # run newly observed, matching the agent's own `name` -- i.e. qwen3:14b
    # sometimes tries to "submit" its final answer as a fake tool call
    # instead of plain final text). `summary` reflects a real, earlier,
    # genuine final-response event's `final_text` if one was already set
    # before the hallucination; otherwise it falls back to the hallucinated
    # call's own "summary" argument (real model output recovered through the
    # wrong structural envelope -- confirmed safe since none of this
    # specialist's 4 real registered tools take a "summary" parameter), or
    # "" if even that is absent. `cited_chunks` is unaffected either way --
    # citation verification/stripping still runs on whatever `summary` ends
    # up being, so a hallucination-recovered summary can never carry an
    # unverified `[chunk:<id>]` marker either.
    hallucinated_tool_call: bool = False


class _ModelCallLoggingPlugin(BasePlugin):
    """Real, non-mocked per-internal-call `ModelCall` logging, driven by the
    real `BasePlugin.after_model_callback` hook ADR 0003 Finding 8 confirmed
    (Task 1's `_UsageCapturePlugin` only ever proved the hook fires with real
    usage data -- it never wrote a real `ModelCall` row; this is that real
    wiring). `before_model_callback` is also used, purely to bracket a real
    `time.monotonic()` pair so each logged row gets a real, measured
    `latency_ms` rather than a placeholder 0 (see `log_adk_model_call`'s
    docstring for why that function can't measure it itself).

    `self._start_times` is a plain list used as a stack: ADK's own internal
    tool-calling loop makes model calls strictly sequentially within one
    `run_async` invocation (confirmed by this module's own real end-to-end
    test -- `captured_usage` always grows one entry per `after_model_
    callback` firing, never concurrently), so a simple push/pop pair
    correctly associates each `after_model_callback` firing with the
    `before_model_callback` firing that immediately preceded it.
    """

    def __init__(
        self, session: Session, tracer: Tracer, *, provider: str, model: str, routing_reason: str
    ) -> None:
        super().__init__(name="diagnostic_model_call_logging")
        self._session = session
        self._tracer = tracer
        self._provider = provider
        self._model = model
        self._routing_reason = routing_reason
        self._start_times: list[float] = []
        self.logged_count = 0

    async def before_model_callback(self, *, callback_context, llm_request):
        self._start_times.append(time.monotonic())
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        start = self._start_times.pop() if self._start_times else None
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0
        # Code-review fix: this used to only call `log_adk_model_call` when
        # `usage_metadata is not None`, silently dropping any real internal
        # call that failed or returned a malformed response (no row, no
        # error signal -- a real gap vs. `log_completion`'s own error-path
        # convention). Now ALWAYS logs, every real `after_model_callback`
        # firing -- `log_adk_model_call` itself branches on `usage_metadata`
        # to write a success- or error-shaped row (see its docstring),
        # passing through `llm_response.error_message` (a real field on
        # `LlmResponse`, confirmed by reading its source) when present.
        log_adk_model_call(
            self._session,
            self._tracer,
            purpose=_MODEL_CALL_PURPOSE,
            provider=self._provider,
            model=self._model,
            usage_metadata=llm_response.usage_metadata,
            latency_ms=latency_ms,
            routing_reason=self._routing_reason,
            error_message=llm_response.error_message,
        )
        self.logged_count += 1
        return None


def _curry_tool(func, session: Session, principal: Principal):
    """Bind `session`/`principal` into `func` (one of Task 2's 4 diagnostic
    tool functions) via a closure, producing a callable ADK's real schema
    introspection sees as if `session`/`principal` never existed as
    parameters -- see this module's own docstring for the full real evidence
    behind this exact shape (closure + manual `__signature__`, not a bare
    `functools.partial`).

    Also degrades any `DiagnosticToolError` (the authz/not-found exceptions
    Task 2's tools raise) to the same `{"error": str(exc)}` shape ADK's own
    `FunctionTool.run_async` already returns for its own missing-argument
    case -- never a data-leaking payload (these exceptions' messages are
    plain strings like "not authorized to access ticket 123", never the
    denied record's own content), and never an unhandled exception that
    would otherwise crash the whole multi-turn run over one adversarial or
    mistaken tool argument.
    """
    original_signature = inspect.signature(func)
    remaining_params = [
        param
        for name, param in original_signature.parameters.items()
        if name not in ("session", "principal")
    ]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(session, principal, *args, **kwargs)
        except DiagnosticToolError as exc:
            return {"error": str(exc)}

    # functools.wraps copies func.__wrapped__/__name__/__doc__ etc, but NOT
    # a narrowed signature (it would otherwise copy func's FULL signature,
    # session/principal included) -- this explicit override is what ADK's
    # real schema-declaration AND invocation-time mandatory-arg/valid-params
    # checks both actually read (see this module's docstring for the
    # real, empirically-verified confirmation that both call sites use
    # `inspect.signature(self.func)`, honoring this override consistently).
    wrapper.__signature__ = original_signature.replace(parameters=remaining_params)
    return wrapper


def _risk_level_for_ticket(priority: str) -> str:
    return "high" if priority in _HIGH_PRIORITY_VALUES else "medium"


def _build_agent(session: Session, principal: Principal, *, litellm_model_name: str) -> Agent:
    tools = [
        _curry_tool(find_similar_tickets, session, principal),
        _curry_tool(get_employee_entitlements, session, principal),
        _curry_tool(retrieve_ticket_kb_evidence, session, principal),
        _curry_tool(get_ticket_state_and_history, session, principal),
    ]
    litellm_kwargs = {
        "model": f"openai/{litellm_model_name}",
        "api_base": llm_gateway.LITELLM_BASE_URL,
        "api_key": llm_gateway.LITELLM_MASTER_KEY,
        # Real, verified fix carried forward from Task 1's spike (ADR 0003
        # Finding 3) -- required for every real tool-calling call to a
        # litellm-proxied model, not optional tuning.
        "_skip_mcp_handler": True,
    }
    if litellm_model_name == llm_gateway.DEFAULT_MODEL:
        # ADR 0003 Finding 4: without this, local-qwen3 returns a bare
        # literal "{}" text reply instead of a real tool call -- confirmed
        # the hard way (first real run of this module returned exactly that
        # broken shape before this conditional existed; see ADR 0004's
        # amendment note). `think` is Ollama-specific and must NOT be sent
        # to cloud-primary/cloud-fallback at all (llm_gateway.py's own
        # precedent: sending it unconditionally breaks every cloud call with
        # a real 400), so this is only added to the kwargs for local-qwen3,
        # never passed as `think=True`/omitted-as-default for it.
        litellm_kwargs["think"] = False
    return Agent(
        name="diagnostic_specialist",
        model=LiteLlm(**litellm_kwargs),
        instruction=_INSTRUCTION,
        tools=tools,
    )


async def diagnose_ticket(
    session: Session, tracer: Tracer, principal: Principal, ticket_id: int
) -> DiagnosisResult:
    """Run one real, bounded diagnostic specialist session over `ticket_id`,
    scoped to `principal`'s own authorized department(s), and return a real
    `DiagnosisResult`.

    Fails fast (raises `TicketNotFoundError`/`TicketNotAuthorizedError`,
    same as every Task 2 tool) BEFORE building any real `Agent`/spending any
    real model call, by calling the real `get_ticket_state_and_history` tool
    directly once up front -- reusing Task 2's own authz check rather than
    duplicating it, and getting the real ticket `priority`/`subject` this
    function also needs for model routing and the opening prompt.

    `session`/`tracer` are expected to be a real, request-scoped
    `Depends(get_db)` session and the app's real `Tracer` (Task 4's future
    endpoint has both already) -- this function does NOT call `session.
    commit()` itself (same caller-owns-the-transaction-boundary convention
    `model_call_logging.py` already established); the caller must commit (or
    otherwise flush) after this returns, or every `ModelCall` row this run
    logged is silently lost.

    Citation evidence (Phase 13 Task 4): every real `retrieve_ticket_kb_
    evidence` tool call this run makes produces a real ADK `function_
    response` event alongside the `function_call` event the pre-existing
    `tool_calls_made` tracking already read -- this function now ALSO reads
    that event (`part.function_response.name ==
    "retrieve_ticket_kb_evidence"`, confirmed by reading the installed
    `google-adk==1.14.1` source: a dict-returning tool's raw return value is
    used as-is as `function_response.response`, never wrapped) to recover
    the real `{chunk_id, document_title}` pairs `retrieve_ticket_kb_
    evidence` actually returned. After the run, `summary`'s own
    `[chunk:<id>]` markers are verified against exactly those real chunk
    ids via `resolvegrid_retrieval.citation_verification.verify_citations`
    (the SAME deterministic check `/chat`'s `verify_citations_node` already
    applies -- never a second, parallel mechanism); any fabricated marker is
    stripped from `summary` before it's returned, and `DiagnosisResult.
    cited_chunks` carries only the verified subset. See that field's own
    docstring for the full rationale.
    """
    ticket_state = get_ticket_state_and_history(session, principal, ticket_id)
    ticket = ticket_state["ticket"]

    risk_level = _risk_level_for_ticket(ticket["priority"])
    litellm_model_name = select_model_for_risk_level(risk_level)
    provider = llm_gateway.provider_for_model(litellm_model_name)
    routing_reason = f"risk_level={risk_level}"

    agent = _build_agent(session, principal, litellm_model_name=litellm_model_name)
    plugin = _ModelCallLoggingPlugin(
        session, tracer, provider=provider, model=litellm_model_name, routing_reason=routing_reason
    )
    app = App(name="resolvegrid-diagnostic-specialist", root_agent=agent, plugins=[plugin])
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    adk_session = await session_service.create_session(
        app_name="resolvegrid-diagnostic-specialist", user_id=str(principal.employee_id)
    )

    tool_calls_made: list[str] = []
    turn_count = 0
    final_text: str | None = None
    # Phase 13 Task 4: real evidence recovered from `retrieve_ticket_kb_
    # evidence`'s own real `function_response` event(s) -- see this
    # function's docstring's "Citation evidence" section below. Keyed by
    # chunk_id (deduped across however many times the tool was actually
    # called this run, last-write-wins -- the underlying chunk row's own
    # title is stable across calls, so this is a dedup, not a
    # correctness-sensitive ordering choice).
    evidence_chunks: dict[int, dict] = {}

    prompt = (
        f"Diagnose ticket #{ticket['id']} (subject: {ticket['subject']!r}, "
        f"type: {ticket['type']}, priority: {ticket['priority']}, status: "
        f"{ticket['status']}). Use your tools to investigate and produce a "
        "diagnostic summary for the analyst."
    )

    hallucinated_tool_call = False
    # Real Task 6 finding (see the `except ValueError` branch below): when
    # qwen3:14b hallucinates a call to an unregistered tool name, the call's
    # own `args` have been observed to contain exactly the model's real
    # intended final answer, under a "summary" key (e.g.
    # `{"summary": "Ticket #43 (...) is an open incident..."}`) -- the model
    # tried to "submit" its answer via a fake tool call instead of plain
    # final text. None of this agent's 4 REAL registered tools take a
    # "summary" parameter (confirmed: `find_similar_tickets`/`get_employee_
    # entitlements`/`retrieve_ticket_kb_evidence`/`get_ticket_state_and_
    # history`'s real signatures, `diagnostic_tools.py`), so capturing this
    # is safe and specific to the hallucination case, never a generic
    # "any tool arg named summary is an answer" rule. Used ONLY as a
    # fallback (below, after the loop) when the run is caught as
    # hallucinated AND no real `final_text` was ever set by a genuine
    # final-response event -- this is real model output recovered through
    # the wrong structural envelope, not a fabrication.
    candidate_summary_text: str | None = None
    try:
        async for event in runner.run_async(
            user_id=str(principal.employee_id),
            session_id=adk_session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            run_config=RunConfig(max_llm_calls=_MAX_LLM_CALLS),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call is not None:
                        turn_count += 1
                        tool_calls_made.append(part.function_call.name)
                        call_args = part.function_call.args or {}
                        if isinstance(call_args.get("summary"), str):
                            candidate_summary_text = call_args["summary"]
                    # Real, empirically-confirmed shape (see this function's
                    # docstring): since `retrieve_ticket_kb_evidence` already
                    # returns a real `dict`, ADK's own `__build_response_event`
                    # (google.adk.flows.llm_flows.functions, read directly from
                    # the installed `google-adk==1.14.1` source) uses that dict
                    # AS-IS as `function_response.response` -- never wraps it
                    # under a `{"result": ...}` key the way it would for a
                    # non-dict return value. `.get(...)` throughout (never bare
                    # indexing): a `{"error": ...}` result (the authz-denial/
                    # not-found degrade `_curry_tool` produces) has no "chunks"
                    # key at all, and must not raise here.
                    response = part.function_response
                    if response is not None and response.name == _KB_EVIDENCE_TOOL_NAME:
                        chunks = (response.response or {}).get("chunks") or []
                        for chunk in chunks:
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id is not None:
                                evidence_chunks[chunk_id] = {
                                    "chunk_id": chunk_id,
                                    "document_title": chunk.get("document_title", ""),
                                }
            if event.is_final_response() and event.content and event.content.parts:
                text_parts = [part.text for part in event.content.parts if part.text]
                if text_parts:
                    final_text = "".join(text_parts)
    except ValueError as exc:
        # Real, previously-flagged-but-unfixed ADK robustness gap (ADR
        # 0005's "fourth follow-up"), now root-caused and fixed here: when
        # qwen3:14b hallucinates a call to a tool name that was never
        # registered (observed real names: "diagnostic_summary", ADR 0005's
        # original finding, and "diagnostic_specialist" -- matching this
        # agent's OWN `name`, i.e. the model sometimes tries to "submit" its
        # final answer as a fake tool call instead of plain final text),
        # ADK's own internal `_get_tool_and_context`
        # (google.adk.flows.llm_flows.functions) raises a bare,
        # undifferentiated `ValueError` ("Function <name> is not found in
        # the tools_dict.") that would otherwise propagate out of this
        # function, crash the whole diagnostic run, and surface as a real
        # 502 to the analyst -- reproduced for real against a genuinely
        # empty, freshly-migrated database during this phase's Task 6
        # fresh-state verification (not a one-off). Caught narrowly, by this
        # exact, stable ADK-internal message substring -- never a blanket
        # `except ValueError`, since `Runner.run_async` could in principle
        # raise an unrelated `ValueError` this must NOT silently swallow.
        # Degrades to whatever real state this run had already legitimately
        # gathered (`tool_calls_made`/`turn_count`/`evidence_chunks`, and any
        # real `final_text` already captured from an earlier, genuine
        # final-response event); if no genuine `final_text` was ever set,
        # falls back to `candidate_summary_text` (above) -- the hallucinated
        # call's OWN real "summary" argument, recovered because this phase's
        # own real, reproduced observation is that it genuinely is the
        # model's real intended answer, delivered through the wrong
        # structural envelope, not an unrelated/adversarial value. This is
        # real model output read correctly, not a fabrication.
        if "is not found in the tools_dict" not in str(exc):
            raise
        hallucinated_tool_call = True
        if final_text is None:
            final_text = candidate_summary_text

    # Citation verification/stripping (Phase 13 Task 4): the exact same
    # real, deterministic check `/chat`'s `verify_citations_node` applies
    # (`resolvegrid_agent_orchestration.graph`), reused directly rather than
    # a second, parallel citation-trust mechanism -- see `DiagnosisResult.
    # cited_chunks`'s docstring for the full rationale. `valid_chunk_ids` is
    # exactly the set of chunk ids `retrieve_ticket_kb_evidence` actually
    # returned THIS run (never the full KB), matching `verify_citations`'s
    # own documented "shown to the model, not exists anywhere" semantic.
    # `verify_citations("", ...)`/an answer with no `[chunk:<id>]` markers
    # at all is itself a real, unremarkable, vacuously-verified case (see
    # that function's own docstring) -- no special-casing needed here.
    summary_text = final_text or ""
    verification = verify_citations(summary_text, set(evidence_chunks.keys()))
    if not verification.all_verified:
        fabricated_spans = [(c.start, c.end) for c in verification.citations if not c.verified]
        summary_text = strip_fabricated_spans(summary_text, fabricated_spans)
    cited_chunks = [evidence_chunks[chunk_id] for chunk_id in verification.verified_chunk_ids]

    return DiagnosisResult(
        ticket_id=ticket_id,
        summary=summary_text,
        tool_calls_made=tool_calls_made,
        turn_count=turn_count,
        model_calls_logged=plugin.logged_count,
        cited_chunks=cited_chunks,
        hallucinated_tool_call=hallucinated_tool_call,
    )
