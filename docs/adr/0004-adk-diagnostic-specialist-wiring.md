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
