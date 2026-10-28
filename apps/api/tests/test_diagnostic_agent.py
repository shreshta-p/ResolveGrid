"""Tests for `resolvegrid_api.diagnostic_agent` (Phase 13 Task 3).

Three kinds of real evidence, matching this phase's own "nothing mocked"
standing directive:

1. Real currying proof against the ACTUAL Task 2 tool functions (not toy
   stand-ins) -- `test_curry_tool_*` builds a real `google.adk.tools.
   function_tool.FunctionTool` from `_curry_tool`'s output and inspects its
   real declaration/invocation behavior. This is the committed version of
   the empirical investigation ADR 0004's "Finding 1" cites.
2. `test_risk_level_for_ticket_mapping`: a pure, DB-free unit test of the
   model-routing mapping.
3. `test_diagnose_ticket_real_end_to_end_against_live_stack`: the real,
   non-mocked, end-to-end proof this task's own completion criteria
   require -- a real seeded ticket/employee/entitlement/similar-tickets
   fixture, a real `Agent`/`Runner` run against the live LiteLLM/Ollama
   stack, asserting the specialist genuinely calls at least 2 of its 4 real
   tools and that real per-internal-call `ModelCall` rows land in Postgres.
   Mirrors `test_adk_integration_spike.py`'s "no pytest-asyncio dependency --
   drive asyncio.run() directly" convention.
"""

import asyncio
import re

import pytest
from google.adk.events.event import Event
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from sqlalchemy import func, select

from resolvegrid_api.diagnostic_agent import (
    _MODEL_CALL_PURPOSE,
    _curry_tool,
    _risk_level_for_ticket,
    diagnose_ticket,
)
from resolvegrid_api.diagnostic_tools import (
    EmployeeNotFoundError,
    find_similar_tickets,
    get_employee_entitlements,
)
from resolvegrid_api.models.org import Department, Employee, Location
from resolvegrid_api.models.telemetry import ModelCall
from resolvegrid_api.models.ticketing import Queue, Ticket
from resolvegrid_api.operational_adapters.entitlements import grant_vpn_access
from resolvegrid_authz import Principal, RoleGrant
from resolvegrid_telemetry import init_tracing


class _FakeToolContext:
    tool_confirmation = None


# ---------------------------------------------------------------------------
# 1. Real currying proof against the actual Task 2 tool functions
# ---------------------------------------------------------------------------


def test_curry_tool_narrows_schema_and_preserves_name_and_doc():
    """`_curry_tool(find_similar_tickets, session, principal)` must produce
    a callable whose real ADK tool declaration shows ONLY `ticket_id`/
    `limit` (never `session`/`principal`), under the tool's REAL name and
    docstring -- not `"partial"`/the generic `functools.partial` docstring a
    bare `functools.partial` would yield (ADR 0004 Finding 1).
    """
    bound = _curry_tool(find_similar_tickets, "FAKE_SESSION", "FAKE_PRINCIPAL")

    assert bound.__name__ == "find_similar_tickets"
    assert bound.__doc__ == find_similar_tickets.__doc__

    declaration = FunctionTool(bound)._get_declaration()
    assert declaration.name == "find_similar_tickets"
    assert set(declaration.parameters.properties.keys()) == {"ticket_id", "limit"}


def test_curry_tool_threads_curried_args_through_on_real_invocation():
    """A real `FunctionTool.run_async()` call against the curried callable
    must thread the curried `session`/`principal` through to the real
    underlying function unchanged -- proven against `get_employee_
    entitlements`'s real signature `(session, principal, employee_id)`.
    """
    calls = []

    def _spy(session, principal, employee_id):
        calls.append((session, principal, employee_id))
        return []

    _spy.__name__ = "get_employee_entitlements"
    _spy.__doc__ = get_employee_entitlements.__doc__
    bound = _curry_tool(_spy, "SESSION_X", "PRINCIPAL_Y")
    tool = FunctionTool(bound)

    result = asyncio.run(tool.run_async(args={"employee_id": 42}, tool_context=_FakeToolContext()))

    assert result == []
    assert calls == [("SESSION_X", "PRINCIPAL_Y", 42)]


def test_curry_tool_degrades_diagnostic_tool_error_to_error_dict_without_raising(db_session):
    """A real `DiagnosticToolError` (here: `EmployeeNotFoundError`, from a
    real call against a real session with a deliberately nonexistent
    `employee_id`) must come back as a plain `{"error": ...}` dict, never
    propagate as an unhandled exception that would crash the whole
    multi-turn run -- and the message must not leak any denied record's
    content (there is none to leak here; the record doesn't exist at all).
    """
    principal = Principal(employee_id=1, roles=(RoleGrant(role="admin", scope="global"),))
    bound = _curry_tool(get_employee_entitlements, db_session, principal)
    tool = FunctionTool(bound)

    result = asyncio.run(
        tool.run_async(args={"employee_id": -1}, tool_context=_FakeToolContext())
    )

    assert result == {"error": "employee -1 not found"}

    # Sanity check: calling the real, uncurried function directly with the
    # same arguments really does raise (proving the curry's try/except is
    # doing real work, not masking a path that never raised in the first
    # place).
    with pytest.raises(EmployeeNotFoundError):
        get_employee_entitlements(db_session, principal, -1)


# ---------------------------------------------------------------------------
# 2. Model-routing mapping (pure, no DB/network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "priority,expected",
    [("high", "high"), ("urgent", "high"), ("critical", "high"), ("medium", "medium"),
     ("low", "medium"), ("unrecognized", "medium")],
)
def test_risk_level_for_ticket_mapping(priority, expected):
    assert _risk_level_for_ticket(priority) == expected


# ---------------------------------------------------------------------------
# 3. Real end-to-end diagnostic run against the live stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tracer():
    # Same lazy-module-scoped-fixture rationale as test_model_call_logging.py's
    # own `tracer` fixture (see that file for the full collection-order
    # explanation) -- init_tracing() is idempotent, deferred to test-run time.
    return init_tracing("test-diagnostic-agent")


_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def test_diagnose_ticket_real_end_to_end_against_live_stack(db_session, tracer):
    """The real Task 3 completion proof: a real seeded ticket/employee/
    entitlement/similar-tickets fixture, a real `Agent`/`Runner` run against
    the live LiteLLM proxy + Ollama `local-qwen3`, asserting:
    - the specialist genuinely calls at least 2 of its 4 real tools,
    - it produces a real, non-empty, tool-informed summary,
    - real per-internal-call `ModelCall` rows land in Postgres with the new
      `"diagnostic.specialist_turn"` purpose, a real non-zero trace id, and
      real non-zero token counts.
    """
    location = Location(name="Diag Agent HQ", region="US", timezone="America/Chicago")
    department = Department(name="Diag Agent Eng")
    db_session.add_all([location, department])
    db_session.flush()

    queue = Queue(name="Diag Agent Queue", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    requester = Employee(
        display_name="Diag Agent Requester",
        email="diag.agent.requester@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    # A real, active entitlement grant -- so get_employee_entitlements has
    # real, non-empty data to return if the specialist calls it.
    grant_vpn_access(db_session, requester.id, "pre-existing access, for diagnostic context")

    anchor = Ticket(
        subject="VPN connection fails after password reset",
        category="network_access",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(anchor)
    db_session.flush()

    # A real past ticket with genuine subject-text overlap, so
    # find_similar_tickets has a real, non-empty candidate to surface.
    similar = Ticket(
        subject="VPN connection fails for remote employee",
        category="network_access",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(similar)
    db_session.flush()

    principal = Principal(employee_id=requester.id, roles=(RoleGrant(role="admin", scope="global"),))

    # Phase 13 Task 4 fix (found via this codebase's own real full-suite
    # run, not assumed): this test's `ModelCall` query below used to filter
    # by `purpose == _MODEL_CALL_PURPOSE` ALONE, which was correct only
    # while this test's own `db_session` (rollback-based, so THIS test's
    # own rows never persist) was the sole real caller ever writing rows
    # under that purpose. Task 4's `POST /tickets/{id}/diagnose` is a
    # SECOND real caller that writes real, permanently COMMITTED rows under
    # the identical purpose string (via a genuinely different connection --
    # Postgres' read-committed default means those committed rows are
    # visible here too) -- a real end-to-end run of that endpoint (or a
    # repeat full-suite run) inflates this query's result, exactly as this
    # task's own real test run demonstrated (`assert 12 == 3`, not a
    # mocked/predicted failure). A watermark -- the max existing matching
    # `ModelCall.id` taken BEFORE this call -- scopes the later assertion to
    # rows this run itself created, mirroring `test_chat_api.py`'s own
    # established watermark-based cleanup precedent for the identical
    # "shared real DB, can't just filter by a global column" problem.
    watermark_id = db_session.scalar(
        select(func.max(ModelCall.id)).where(ModelCall.purpose == _MODEL_CALL_PURPOSE)
    ) or 0

    result = asyncio.run(diagnose_ticket(db_session, tracer, principal, anchor.id))

    assert result.ticket_id == anchor.id
    # Real, documented exceptions to "always non-empty" (Phase 13 Task 6,
    # found via this exact test during real fresh-state/CI runs): qwen3:14b
    # sometimes hallucinates its final answer as a fake, unregistered tool
    # call (observed real name: "diagnostic_specialist", matching this
    # agent's own `name`) instead of plain final text -- `diagnose_ticket`
    # recovers real text from that call's own arguments when possible, but
    # an honestly empty summary is the correct, non-fabricated outcome on
    # the rare real run where even that recovery finds nothing usable.
    # Separately, under slow-hardware conditions (confirmed for real in
    # CI), qwen3:14b's documented repeated-tool-re-invocation pathology can
    # genuinely exhaust `_MAX_LLM_CALLS=60` within one run, leaving no
    # final answer at all (`exceeded_call_limit`). A NORMAL (neither
    # hallucinated nor call-limit-exhausted) run must still produce real,
    # non-empty content -- this assertion still enforces that.
    assert result.summary.strip() or result.hallucinated_tool_call or result.exceeded_call_limit, (
        f"empty summary on an otherwise-normal run is a real bug: {result!r}"
    )
    # The hallucinated pseudo-tool-call name (if any) is intentionally
    # EXCLUDED from "real tools called" below -- it was never a real tool
    # invocation (ADK's own internals rejected it; see `diagnose_ticket`'s
    # `except ValueError` branch), so counting it toward "distinct real
    # tools used" would overstate what this run actually, successfully did.
    distinct_tools_called = set(result.tool_calls_made) - {"diagnostic_specialist"}
    assert len(distinct_tools_called) >= 2, (
        f"expected at least 2 distinct real tools called, got: {result.tool_calls_made!r}"
    )
    assert distinct_tools_called <= {
        "find_similar_tickets",
        "get_employee_entitlements",
        "retrieve_ticket_kb_evidence",
        "get_ticket_state_and_history",
    }
    assert result.model_calls_logged > 0

    # Phase 13 Task 4: `cited_chunks` is real, verified evidence (never a
    # fabrication) -- if the specialist actually called
    # retrieve_ticket_kb_evidence AND cited something from it, every
    # resulting entry must be a real {chunk_id, document_title} dict whose
    # id is drawn from this run's own real tool results, and `summary` must
    # never contain a `[chunk:<id>]` marker for an id NOT in `cited_chunks`
    # (a fabricated/unverified citation would have been stripped).
    import re as _re

    cited_ids = {entry["chunk_id"] for entry in result.cited_chunks}
    for entry in result.cited_chunks:
        assert isinstance(entry["chunk_id"], int)
        assert isinstance(entry["document_title"], str)
    summary_marker_ids = {int(m) for m in _re.findall(r"\[chunk:(\d+)\]", result.summary)}
    assert summary_marker_ids <= cited_ids

    # Real DB proof: one ModelCall row per real internal call this run
    # logged, under the new, honestly-distinct purpose -- not one fake
    # aggregate row. Uses the SAME session (autoflush sees the pending
    # ModelCall rows diagnose_ticket added without needing a commit).
    rows = db_session.scalars(
        select(ModelCall).where(ModelCall.purpose == _MODEL_CALL_PURPOSE, ModelCall.id > watermark_id)
    ).all()
    assert len(rows) == result.model_calls_logged
    for row in rows:
        assert row.provider == "ollama"
        assert row.model == "local-qwen3"
        assert row.status == "success"
        assert row.routing_reason == "risk_level=medium"
        assert row.input_tokens > 0
        # estimated_cost_usd is a real, honest $0 -- the real seeded
        # ollama/local-qwen3 PricingVersion row is a genuine $0 rate, not a
        # placeholder this test is masking a missing lookup with.
        assert row.estimated_cost_usd == 0.0
        assert _TRACE_ID_RE.match(row.trace_id)


# ---------------------------------------------------------------------------
# 4. Phase 13 Task 6 fix: a hallucinated tool-call name must degrade the run,
#    never crash it
# ---------------------------------------------------------------------------


def test_diagnose_ticket_degrades_gracefully_on_hallucinated_tool_call(db_session, tracer, monkeypatch):
    """Real, reproducible finding from this phase's own Task 6 fresh-state
    verification (see `docs/adr/0005-diagnostic-eval-integration-depth.md`'s
    "fourth follow-up" closing note): ADK's own internals
    (`google.adk.flows.llm_flows.functions._get_tool_and_context`) raise a
    bare, undifferentiated `ValueError` ("Function <name> is not found in
    the tools_dict.") when the model hallucinates a call to a tool name
    that was never registered -- observed for real against this exact live
    stack, not a hypothetical. This test does not re-run the real model
    (that nondeterministic behavior is already proven reproducible by the
    real end-to-end tests above and in `test_diagnostic_eval.py`); it
    directly, deterministically proves `diagnose_ticket`'s own catch
    behaves correctly by monkeypatching ADK's `Runner.run_async` to raise
    the exact real exception string observed, confirming `diagnose_ticket`
    degrades (returns a real `DiagnosisResult` with
    `hallucinated_tool_call=True`) rather than letting it propagate and
    crash the caller (previously surfacing as a real 502 at the endpoint).
    """
    from google.adk.runners import Runner

    location = Location(name="Diag Hallucination HQ", region="US", timezone="America/Chicago")
    department = Department(name="Diag Hallucination Eng")
    db_session.add_all([location, department])
    db_session.flush()

    queue = Queue(name="Diag Hallucination Queue", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    requester = Employee(
        display_name="Diag Hallucination Requester",
        email="diag.hallucination.requester@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    ticket = Ticket(
        subject="Printer firmware reports a paper jam that isn't real",
        category="hardware",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    principal = Principal(employee_id=requester.id, roles=(RoleGrant(role="admin", scope="global"),))

    async def _raise_hallucinated_tool_call(self, *args, **kwargs):
        raise ValueError("Function diagnostic_specialist is not found in the tools_dict.")
        yield  # pragma: no cover -- makes this an async generator function

    monkeypatch.setattr(Runner, "run_async", _raise_hallucinated_tool_call)

    result = asyncio.run(diagnose_ticket(db_session, tracer, principal, ticket.id))

    assert result.hallucinated_tool_call is True
    assert result.ticket_id == ticket.id
    # No real final-response event was ever observed before the
    # hallucination -- an honest empty summary, never a fabricated one.
    assert result.summary == ""
    assert result.cited_chunks == []


def test_diagnose_ticket_does_not_swallow_an_unrelated_value_error(db_session, tracer, monkeypatch):
    """The catch added for the hallucinated-tool-call gap above is narrowly
    scoped to that exact, stable ADK-internal message substring -- a
    genuinely unrelated `ValueError` from `Runner.run_async` (e.g. a real
    ADK/LiteLLM wiring bug unrelated to tool-call hallucination) must still
    propagate, not be silently absorbed into a falsely-successful
    `DiagnosisResult`.
    """
    from google.adk.runners import Runner

    location = Location(name="Diag Unrelated HQ", region="US", timezone="America/Chicago")
    department = Department(name="Diag Unrelated Eng")
    db_session.add_all([location, department])
    db_session.flush()

    queue = Queue(name="Diag Unrelated Queue", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    requester = Employee(
        display_name="Diag Unrelated Requester",
        email="diag.unrelated.requester@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    ticket = Ticket(
        subject="Unrelated ValueError probe",
        category="hardware",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    principal = Principal(employee_id=requester.id, roles=(RoleGrant(role="admin", scope="global"),))

    async def _raise_unrelated_value_error(self, *args, **kwargs):
        raise ValueError("some genuinely unrelated ADK/LiteLLM wiring error")
        yield  # pragma: no cover -- makes this an async generator function

    monkeypatch.setattr(Runner, "run_async", _raise_unrelated_value_error)

    with pytest.raises(ValueError, match="some genuinely unrelated ADK/LiteLLM wiring error"):
        asyncio.run(diagnose_ticket(db_session, tracer, principal, ticket.id))


def test_diagnose_ticket_degrades_gracefully_when_llm_call_limit_is_exceeded(db_session, tracer, monkeypatch):
    """Real, reproduced finding from this phase's own first-ever CI push
    (run 40668281723): `test_diagnose_ticket_real_end_to_end_against_live_
    stack` took 33m39s in CI before FAILING -- consistent with qwen3:14b's
    already-documented (ADR 0003 Finding 5) tendency to repeatedly
    re-invoke an already-answered tool genuinely reaching this
    specialist's full `_MAX_LLM_CALLS=60` hard cap under CI's much slower
    (CPU-only) per-call speed, something fast local GPU inference never
    surfaced. ADK's own `LlmCallsLimitExceededError`
    (`google.adk.agents.invocation_context`) is a plain `Exception`
    subclass, NOT a `ValueError` -- confirmed by reading its real source --
    so it was never caught by the hallucination fix's `except ValueError`
    above, and previously crashed the whole run exactly like the
    hallucination case did before THAT was fixed. This test proves
    `diagnose_ticket` now degrades instead: a real `DiagnosisResult` with
    `exceeded_call_limit=True`, never a propagated crash.
    """
    from google.adk.runners import Runner

    location = Location(name="Diag Call Limit HQ", region="US", timezone="America/Chicago")
    department = Department(name="Diag Call Limit Eng")
    db_session.add_all([location, department])
    db_session.flush()

    queue = Queue(name="Diag Call Limit Queue", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    requester = Employee(
        display_name="Diag Call Limit Requester",
        email="diag.call.limit.requester@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    ticket = Ticket(
        subject="Call limit exhaustion probe",
        category="hardware",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    principal = Principal(employee_id=requester.id, roles=(RoleGrant(role="admin", scope="global"),))

    async def _raise_call_limit_exceeded(self, *args, **kwargs):
        from google.adk.agents.invocation_context import LlmCallsLimitExceededError

        raise LlmCallsLimitExceededError("Max number of llm calls limit of 60 exceeded")
        yield  # pragma: no cover -- makes this an async generator function

    monkeypatch.setattr(Runner, "run_async", _raise_call_limit_exceeded)

    result = asyncio.run(diagnose_ticket(db_session, tracer, principal, ticket.id))

    assert result.exceeded_call_limit is True
    assert result.hallucinated_tool_call is False
    assert result.ticket_id == ticket.id
    assert result.summary == ""
    assert result.cited_chunks == []


def test_diagnose_ticket_recovers_real_text_from_hallucinated_summary_argument(
    db_session, tracer, monkeypatch
):
    """Real, reproduced finding (this phase's own Task 6 fresh-state run): the
    hallucinated tool call's own `args` have been observed to carry the
    model's genuine intended final answer under a "summary" key -- the model
    tried to "submit" its real answer via a fake tool call shaped like a
    function call instead of plain final text. None of this specialist's 4
    real registered tools take a "summary" parameter, so recovering it in
    this exact, narrow circumstance is reading real model output through the
    wrong envelope, not fabricating one. This test proves that recovery
    deterministically: a real ADK `Event` carrying the hallucinated
    function_call is yielded first (exercising the SAME code path that reads
    `part.function_call.args` for every real tool call), then the real
    ValueError is raised on the next iteration, mirroring the exact sequence
    observed live.
    """
    from google.adk.runners import Runner

    location = Location(name="Diag Recovery HQ", region="US", timezone="America/Chicago")
    department = Department(name="Diag Recovery Eng")
    db_session.add_all([location, department])
    db_session.flush()

    queue = Queue(name="Diag Recovery Queue", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    requester = Employee(
        display_name="Diag Recovery Requester",
        email="diag.recovery.requester@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    ticket = Ticket(
        subject="Recovery probe ticket",
        category="hardware",
        type="incident",
        priority="medium",
        queue_id=queue.id,
        requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    principal = Principal(employee_id=requester.id, roles=(RoleGrant(role="admin", scope="global"),))

    recovered_text = "Ticket #999 is a real recovered diagnostic answer."

    async def _yield_hallucinated_call_then_raise(self, *args, **kwargs):
        yield Event(
            author="diagnostic_specialist",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="diagnostic_specialist",
                            args={"summary": recovered_text},
                        )
                    )
                ],
            ),
        )
        raise ValueError("Function diagnostic_specialist is not found in the tools_dict.")

    monkeypatch.setattr(Runner, "run_async", _yield_hallucinated_call_then_raise)

    result = asyncio.run(diagnose_ticket(db_session, tracer, principal, ticket.id))

    assert result.hallucinated_tool_call is True
    assert result.summary == recovered_text
    assert result.tool_calls_made == ["diagnostic_specialist"]
