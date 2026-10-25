# ADR 0004: ADK Diagnostic Specialist Real Agent/Runner Wiring (Phase 13 Task 3)

## Status
Accepted

## Context
Task 1 (ADR 0003) proved the real ADK integration mechanism works against a
trivial, zero-argument tool. Task 2 (`apps/api/src/resolvegrid_api/
diagnostic_tools.py`) built 4 real, authz-scoped, multi-argument diagnostic
tool functions, but deliberately left two open questions for Task 3 to
resolve empirically, not by assumption: how to curry `session`/`principal`
out of each tool's model-visible schema, and which `SessionService`/
`max_llm_calls`/model-routing/`ModelCall`-logging choices the real
`DiagnosticSpecialist` `Agent` should make. This ADR records those real,
evidence-backed decisions.

## Finding 1: currying via a closure (not a bare `functools.partial`), `ToolContext` rejected as unsuitable

Read directly from the installed `google-adk==1.14.1` source
(`google.adk.tools.function_tool.FunctionTool`,
`google.adk.tools.tool_context.ToolContext`,
`google.adk.tools._automatic_function_calling_util.build_function_
declaration`):

- **`ToolContext` is ADK's own fixed, internal object** (invocation
  context, session state, memory, auth-credential helpers) injected via a
  hardcoded `self._ignore_params = ['tool_context', 'input_stream']` list in
  `FunctionTool.__init__` -- it is not a generic "pass any caller-supplied
  object without exposing it to the model" mechanism. There is no way to
  make ADK inject this codebase's own `Session`/`Principal` objects through
  it; `ToolContext` is the wrong tool for this problem, rejected after
  reading its real source, not assumed unsuitable.
- **A bare `functools.partial(tool_fn, session, principal)` almost works**:
  `inspect.signature()` on a `functools.partial` object (confirmed via a
  real, non-mocked call) automatically narrows to only the UN-bound
  remaining parameters -- exactly the schema ADK's `build_function_
  declaration` needs, no manual signature surgery required for that part.
  But a bare `functools.partial` instance has no `__name__` (confirmed:
  `hasattr(partial_obj, '__name__')` is `False`), so `FunctionTool.__init__`
  falls back to `func.__class__.__name__ == "partial"` for every one of the
  4 tools -- a real, confirmed NAME COLLISION across all 4 tools, which
  would make the model unable to distinguish them. `partial.__doc__` is
  also the generic `functools.partial` docstring, not the wrapped
  function's, discarding the real description text the model needs.
- **Decision**: a closure (`diagnostic_agent._curry_tool`), not a bare
  `functools.partial`. `functools.wraps(func)` restores `__name__`/
  `__doc__`/`__qualname__`; an explicit `wrapper.__signature__ = <func's
  signature, minus session/principal>` assignment is still required (a
  plain `def wrapper(*args, **kwargs)` closure's own signature is NOT
  auto-narrowed the way a `functools.partial`'s is -- confirmed
  empirically). This override is read consistently both by `FunctionTool.
  _get_declaration()` (schema) and by `FunctionTool.run_async()`'s
  `_get_mandatory_args()`/`valid_params` filtering (invocation-time
  argument validation) -- both call `inspect.signature(self.func)`,
  verified by reading `function_tool.py`'s real source.
- **Bonus, not incidental**: the closure shape also lets `_curry_tool` catch
  this module's own `DiagnosticToolError` subclasses (the authz/not-found
  exceptions Task 2's tools raise) and degrade them to the same
  `{"error": str(exc)}` shape `FunctionTool.run_async` already returns for
  its own missing-argument case, instead of letting an unhandled exception
  crash the entire multi-turn run over one adversarial/mistaken tool
  argument (e.g. a model-guessed out-of-scope `employee_id`). This is real,
  working, and verified against the actual installed package and the real
  `find_similar_tickets` function (see `apps/api/tests/test_diagnostic_
  agent.py`'s `test_curry_tool_*` tests) -- not just the toy functions used
  during initial exploration.

## Finding 2: `SessionService` choice -- `InMemorySessionService`

This is a bounded, single-shot diagnostic request (an analyst clicks
"Diagnose," gets one real answer), not a persistent multi-day conversation
an analyst resumes across sessions. `DatabaseSessionService` genuinely works
(ADR 0003 Finding 6) but carries two real, confirmed costs with no benefit
for this use case: it auto-creates `sessions`/`events`/`app_states`/
`user_states` tables outside Alembic's migration history, and
`app_states`/`user_states` rows are never cleaned up by any method ADK's
own public API exposes (`delete_session` issues exactly one `DELETE FROM
sessions`, relying on a DB-level `ON DELETE CASCADE` for `events`;
`app_states`/`user_states` have no cascade and no cleanup method at all).
Choosing `InMemorySessionService` instead means there is nothing to leak:
each diagnostic run builds a fresh `InMemorySessionService`, uses it for
exactly one session, and the whole thing is garbage-collected when the
request ends -- zero Alembic-bypass risk, zero orphaned-row risk.

## Finding 3: built per-call, not once at app startup

The plan doc explicitly leaves this as a real decision for Task 3 ("your
call, consistent with whether `DatabaseSessionService` genuinely needs
request-scoped vs. app-lifetime construction"). Since Task 2's tools
require `session`/`principal` to be curried in at construction time, and
those only exist for the lifetime of one request, the `Agent`'s `tools=
[...]` list cannot be built once at app startup the way `classify_complete_
fn`/`compose_complete_fn`/`retrieve_for_agent` are (those either open their
own short-lived session per call via `session_factory()`, or take the scope
as a plain dict argument -- `diagnose_ticket`'s tools instead need a LIVE
session object curried in, per Task 2's own docstring rationale). Combined
with choosing `InMemorySessionService` (no connection pool worth keeping
warm across calls, unlike `DatabaseSessionService`), there is no real
upside left to splitting "build the model" from "build the agent" into two
different lifetimes -- `diagnose_ticket` builds the whole `Agent`/`Runner`/
`InMemorySessionService` fresh per call, mirroring Task 1's own spike shape.

## Finding 4: model routing reuses `routing.select_model_for_risk_level`

A ticket's own `priority` (`high`/`urgent`/`critical` -> risk_level `"high"`,
everything else -> `"medium"`) is mapped to the EXISTING, already-reviewed
`routing.select_model_for_risk_level` policy, not a new, parallel routing
mechanism -- a high-priority ticket's diagnosis can route to `cloud-primary`
exactly like `compose_response`'s own real risk-aware routing already does,
while the common case stays on the zero-marginal-cost `local-qwen3` default.
This task's own real end-to-end verification run used a `medium`-priority
ticket (routes to `local-qwen3`, zero cost, per the plan's own preference
for repeated zero-cost test runs) -- the `cloud-primary` branch is real and
unit-tested (`_risk_level_for_ticket`), but was NOT live-exercised with a
real paid Anthropic call in this task, consistent with this phase's
"zero-marginal-cost by default" framing; a future task adding real evals
(Task 5) is the natural place to add a real, deliberately-incurred
high-priority/cloud-primary live run if one is judged worth the real cost.

## Finding 5: `max_llm_calls` = 60, not a re-copied 40

ADR 0003 Finding 5 found ONE trivial, zero-argument tool could take up to 36
real turns (qwen3's confirmed real tendency to re-invoke an already-answered
tool) even with an explicit "do not call it again" instruction already
present. This specialist has FOUR real tools, each an independent
opportunity for the same pathology -- a naive compounding of Task 1's
observed worst case would be `4 * 36 = 144`. 60 was chosen as a deliberate
middle ground: comfortably above Task 1's own single-tool cap (room for a
well-behaved run to call every tool at least once plus some retry
headroom), while staying well below the theoretical 144-call compounding
worst case, so a genuinely pathological run still terminates in bounded time
rather than silently approaching `RunConfig`'s own 500-call default. This
task's real end-to-end verification run's actual observed call/turn count
is recorded in the Task 3 completion report; amend this constant if a real
run is ever observed to approach it.

## Finding 6: per-internal-call `ModelCall` logging via `after_model_callback`

`model_call_logging.log_adk_model_call` (new) is invoked once per real
`after_model_callback` firing (`diagnostic_agent._ModelCallLoggingPlugin`),
writing one real `ModelCall` row per real internal model call -- never one
aggregate row for the whole multi-turn run. `purpose="diagnostic.
specialist_turn"` is a new, honestly-distinct value (`model_call_logging.
py`'s module docstring's `<domain>.<action>` convention), not reused from
any existing call site. `latency_ms` is measured by the plugin itself (a
real `time.monotonic()` pair bracketing `before_model_callback`/`after_
model_callback`), since -- unlike `log_completion` -- no real completion
call happens inside `log_adk_model_call` for it to time.

## Amendment: a real sign-flip bug, caught by this task's own end-to-end run

The first real draft of `_build_agent` computed `think=(litellm_model_name
== llm_gateway.DEFAULT_MODEL)` -- inverted: this evaluates to `True` for
`local-qwen3`, i.e. passes `think=True`, the exact opposite of ADR 0003
Finding 4's real, confirmed requirement (`think=False` for `local-qwen3`
specifically). This task's own real, non-mocked end-to-end test caught it
immediately and concretely: the specialist's real response was the literal
text `"{}"`, zero tool calls, exactly Finding 4's documented broken shape
reproduced via this bug, not a new one. Fixed by building the `LiteLlm`
kwargs conditionally -- `think=False` is only added to the kwargs dict at
all when `litellm_model_name == llm_gateway.DEFAULT_MODEL`, never sent (not
even as `True`) for any other model, matching `llm_gateway.py`'s own
real-confirmed "cloud providers reject this field outright" precedent.
Recorded here per this phase's "report honestly" standing directive, not
silently amended.

## Flagged, deferred decision: sync tool calls block the event loop

ADK's real `FunctionTool._invoke_callable` (confirmed by reading its
source) calls a sync tool function directly (`target(**args_to_call)`),
with no thread-pool offload -- so each of this specialist's 4 curried tool
calls (all plain, synchronous, DB-querying functions) blocks `diagnose_
ticket`'s own event loop for the real duration of its DB round-trip. For
TODAY's scope -- one bounded, single-shot diagnostic request -- this is not
disqualifying: the blocking window is one real (fast, local Postgres)
query at a time, and nothing else in this process needs that event loop
during a single synchronous test/request. It is explicitly flagged here,
not fixed now, because it is a genuinely different real risk under Task
4's actual deployment shape: a FastAPI process serving several concurrent
analyst diagnose requests at once would have each one's tool calls blocking
the SAME shared event loop other requests' coroutines are also scheduled
on, degrading concurrent throughput. Task 4/5 should revisit this
deliberately once real concurrent-usage numbers exist (e.g. wrapping each
curried tool body in `asyncio.to_thread(...)`, or confirming the real
request volume never makes it matter) rather than this ADR silently
assuming single-request behavior generalizes.

## Consequences
- `apps/api/src/resolvegrid_api/diagnostic_agent.py` is the new real
  `Agent`/`Runner` wiring; `model_call_logging.py` gains `log_adk_model_
  call`.
- Task 4's endpoint handler should call `diagnose_ticket(session, tracer,
  principal, ticket_id)` directly (it already has a real `Depends(get_db)`
  session and a real resolved `Principal`) and must `session.commit()`
  afterward, or every `ModelCall` row this run logged is silently lost
  (same convention as every other `model_call_logging.py` function).
- `services/agent-orchestration` gains a minimal `DiagnosisOutcome`/
  `DiagnosisFn` type (primitive-typed, no `Principal`/ADK import, mirroring
  `RequestApprovalFn`'s plain-`dict`-payload precedent) so Task 4 has a
  ready contract to wire a real graph node against -- `apps/api`'s own
  `diagnose_ticket(session, tracer, principal, ticket_id)` signature is NOT
  that contract verbatim (it needs a live `Session`/`Tracer`/`Principal`,
  none of which can cross into `services/agent-orchestration`); Task 4 must
  write the same kind of session-opening/committing closure wrapper
  `model_call_logging.make_logging_complete_fn` already demonstrates for
  `CompleteFn`.

## Task 4 addendum: no LangGraph node/graph wraps `diagnose_ticket`

The plan doc's "Architecture decision" section explicitly left open,
as a real Task 4 decision, whether this specialist reaches the app via a
new node in the existing chat graph, a small separate compiled graph
(mirroring Phase 9's `build_tool_invocation_graph`), or "another
mechanism" -- and further invited a THIN one-node wrapper as a middle
ground if a full graph felt like pure ceremony. Having now built the real
endpoint against `diagnose_ticket`'s actual shape, the decision is: **no
LangGraph artifact at all** -- `POST /tickets/{id}/diagnose`
(`apps/api/src/resolvegrid_api/routers/tickets.py`) calls `diagnostic_
agent.diagnose_ticket(session, tracer, principal, ticket_id)` directly,
the same way `summarize_ticket` calls `log_completion` directly.

Reasoning, real and specific to what was actually built (not a generic
"graphs are overhead" preference):

1. **No pause/resume boundary exists for a graph to own.** Every other
   LangGraph artifact in this codebase earns its real checkpointing either
   because a run spans multiple HTTP calls across a human-approval
   `interrupt()` (`build_tool_invocation_graph`, genuinely needs durable
   pause/resume across process restarts -- Phase 9 Task 8's own exit
   criterion) or because per-node state needs to accumulate across a
   multi-stage pipeline with real conditional branching (`build_graph`'s
   classify_intent -> retrieve -> compose_response -> verify_citations ->
   finalize chain). `diagnose_ticket` has neither: ADK's own `Runner`
   already owns its ENTIRE multi-turn tool-calling loop as one opaque,
   synchronous-from-the-caller's-perspective unit and returns a finished
   `DiagnosisResult` in one `await`. There is nothing for a checkpointer to
   persist that `diagnose_ticket`'s own return value doesn't already
   carry, and nothing to resume.
2. **The real signature doesn't fit `AgentState` without inventing new
   plumbing for zero benefit.** `AgentState` (services/agent-orchestration/
   state.py) can never hold a live `Session`/`Tracer`/`Principal` (its own
   module docstring's hard rule -- it's checkpointed to Postgres after
   every superstep). A graph node wrapping `diagnose_ticket` would
   therefore need the SAME closure-over-apps/api-internals shape
   `model_call_logging.make_logging_complete_fn` already uses for
   `CompleteFn` -- except unlike `CompleteFn` (a reusable, built-once-at
   -startup closure), `diagnose_ticket` needs a FRESH, request-scoped
   `Session`/`Principal`/`ticket_id` on every call, so the closure would
   have to be built per-request anyway, inside the endpoint handler,
   immediately before `ainvoke()`-ing a graph that does nothing but call
   it once and hand the result straight back out. That is strictly more
   code than calling it directly, for a StateGraph/checkpointer wrapper
   that adds no real capability on top.
3. **The already-added `DiagnosisFn`/`DiagnosisOutcome` types remain
   real, but unused by this task's actual implementation.** They were
   added by Task 3 as a primitive-typed, ready contract IN CASE a future
   graph node needed one (see Task 3's own docstring on those types,
   `graph.py`) -- not a mandate that Task 4 must produce such a node. They
   are left in place (removing public, documented types from a published
   workspace package for a hypothetical future consumer is a worse trade
   than a few unused lines), with their existing docstrings already
   stating this explicitly ("NOT yet wired into `build_graph`"). If a
   future phase genuinely needs diagnosis reachable FROM the general chat
   graph (e.g. `classify_intent` learning to route a ticket-diagnosis-
   shaped question there), these types are exactly the ready seam for
   that -- this addendum is not asking anyone to remove them, only
   recording why Task 4 itself did not consume them.

Real, honest Span-shape consequence of this decision (see `routers/
tickets.py`'s `diagnose_ticket_endpoint` docstring for the full detail):
without a LangGraph node's own `time.monotonic()` instrumentation around
each of ADK's internal turns, there is no real per-tool-call latency
breakdown to report -- `/chat`'s `node_latencies_ms` mechanism has no
equivalent here. Rather than fabricate one, this endpoint writes exactly
ONE `Span` row per diagnostic run (`stage_name="diagnose_ticket"`,
covering the whole real measured call), with `Span.detail_json` carrying
the real `tool_calls_made`/`turn_count`/`model_calls_logged` structured
counts `DiagnosisResult` already reports -- real, honest detail at the
granularity this architecture can actually measure, not a second
fabricated timing breakdown.

## Task 4 addendum: citation evidence (`DiagnosisResult.cited_chunks`)

Task 3 deliberately left `DiagnosisResult` minimal (`summary`/`tool_calls_
made`/`turn_count`/`model_calls_logged` only -- see that class's own
docstring: "real prompt-engineering/parsing work left for Task 4"). Task 4
needed structured, UI-renderable citations (the plan's own requirement:
"real `[chunk:<id>]`-shaped citations, same rendering contract `/chat`
already uses") -- `summary`'s raw text alone isn't that; a caller would
need to re-parse `[chunk:<id>]` markers out of prose and would have no
real `document_title` to show alongside each id.

`diagnose_ticket` (`apps/api/src/resolvegrid_api/diagnostic_agent.py`) now
also inspects each real ADK `function_response` event (not just the
`function_call` events the pre-existing `tool_calls_made` tracking already
read) for `name == "retrieve_ticket_kb_evidence"`, recovering that real
tool call's own `{"chunks": [...], ...}` return value -- confirmed by
reading the installed `google-adk==1.14.1` source
(`google.adk.flows.llm_flows.functions.__build_response_event`): a
dict-returning tool's raw return value is used AS-IS as `function_
response.response`, never wrapped under a `{"result": ...}` key. The
real `[chunk:<id>]` markers in `summary` are then verified against
exactly those real chunk ids via `resolvegrid_retrieval.citation_
verification.verify_citations` -- the SAME deterministic function `/chat`'s
`verify_citations_node` already uses (`services/agent-orchestration/.../
graph.py`), reused directly rather than inventing a second, parallel
citation-trust mechanism. Any fabricated marker is stripped from `summary`
before it is returned (same "strip, don't abstain" consequence that
node's own docstring documents and justifies), and `DiagnosisResult.
cited_chunks` carries only the verified `{chunk_id, document_title}`
pairs -- the exact shape `/chat`'s own `citations` response field already
uses, so `apps/web`'s existing citation-rendering markup needed zero
changes to also render a diagnosis's citations.

**Code-review amendment**: the first real draft of this task duplicated
the span-stripping algorithm as a small, private `_strip_fabricated_
citation_spans` in `diagnostic_agent.py`, reasoning that `graph.py`'s
`_strip_spans` was module-private and therefore off-limits to reach into
from `apps/api`. Code review correctly flagged this as understating a
real drift risk: nothing enforced the two copies staying identical, so a
future real edge case found in one (overlapping-span dedup, unicode
offset handling) would silently not reach the other. Fixed by promoting
the algorithm to a real, PUBLIC function --
`resolvegrid_retrieval.citation_verification.strip_fabricated_spans` --
since both `apps/api` and `services/agent-orchestration` already depend
on that package for `verify_citations` itself; this removes the
cross-package-private-access objection entirely (it's public API now) and
needed no new shared package. Both real call sites (`graph.py`'s
`verify_citations_node` and `diagnostic_agent.py`'s `diagnose_ticket`) now
import and call the same shared function; the two private copies were
deleted, not left behind as dead code.
