"""Real, authz-scoped diagnostic tool functions for the Phase 13 ADK
diagnostic specialist (Task 2 -- see
`docs/superpowers/plans/2026-10-21-phase13-adk-diagnostic-specialist.md`'s
"Task 2" section).

This module does NOT build or wire up the ADK `Agent`/`Runner` itself --
that is Task 3's job. It defines the plain Python functions Task 3 will
register as ADK tools (each takes a real docstring, per ADK's own
documented convention, confirmed in Task 1's spike --
`test_adk_integration_spike.py`'s `get_current_server_time`). Each function
below also takes an explicit `session: Session` and `principal: Principal`
as its first two parameters -- these are NOT meant to ever be part of the
JSON schema a model sees; Task 3 is expected to bind them via
`functools.partial` (or an equivalent closure) before passing the resulting
callable into `Agent(tools=[...])`, exactly the same "factory over an
injected plain callable" shape this codebase already uses for
`CompleteFn`/`RetrieveFn` (see
`services/agent-orchestration/.../graph.py`'s module docstring). Taking
`session`/`principal` as explicit parameters here (rather than this module
opening its own session via `resolvegrid_api.db.session_factory()`, the way
`agent_retrieval.retrieve_for_agent` does) is deliberate: unlike
`retrieve_for_agent` -- called from inside a LangGraph node whose
`AgentState` is checkpointed to Postgres and can never hold a live
`Session` -- these diagnostic tools are expected to be invoked from a
single, synchronous, request-scoped `POST /tickets/{id}/diagnose` handler
(Task 4), which already has both a real `Depends(get_db)` session and a
real resolved `Principal` in scope for the whole request. Matches this
codebase's `operational_adapters/entitlements.py` precedent of taking a
`Session` directly rather than opening its own.

AUTHZ-SCOPING PRINCIPLE (the one this whole module exists to prove):
every function below calls `resolvegrid_authz.authorize()` with an
EXISTING, already-reviewed action string (`"ticket.list"`, `"ticket.view"`,
`"directory.view_employee"`, `"knowledge.retrieve"` via
`build_authz_filter`) -- never a new, parallel authz mechanism -- and
applies the resulting `Decision` as a real SQL-level filter (never a
post-hoc Python-side filter of an already-fetched row set), mirroring
`routers/tickets.py`'s `_apply_ticket_scope`/`_ticket_in_scope` and
`routers/directory.py`'s `_apply_scope`/`_in_scope`. A principal whose
`Decision` doesn't cover a requested resource gets a raised exception (see
below), never an empty/degraded result that could be mistaken for "nothing
found" -- the caller (eventually, an ADK `Agent`) must be able to
distinguish "this resource doesn't exist for you" from "this resource
exists but has no data", and Task 3/4 can decide how each exception surfaces
to the model.

Exceptions (plain, HTTP-framework-free -- these tool functions are not
FastAPI route handlers, so raising `HTTPException` here would be the wrong
layer; mirrors `tool_execution.py`'s own plain-`Exception`-subclass
precedent for the same reason) distinguish "doesn't exist" from "exists but
out of scope", unlike `tool_execution.ToolNotAllowedError`'s deliberate
one-exception-for-both convention -- that convention exists specifically to
stop a caller from probing the TOOL REGISTRY itself; here the equivalent
precedent is `routers/tickets.py`/`routers/directory.py`'s existing,
already-reviewed 404-vs-403 distinction for a single RESOURCE, which this
module mirrors exactly rather than inventing a third convention.

Code-review follow-up: the actual `Decision`-interpretation rules (what
`department_ids`/`employee_id` mean for a query filter / an in-scope check)
now live in `resolvegrid_api.authz_scope`, shared with `routers/tickets.py`
and `routers/directory.py` -- this module used to keep its own independent
copy of the same ticket-scoping logic `routers/tickets.py` already had,
which would have made it the THIRD such copy codebase-wide (the other being
`routers/directory.py`'s Employee-scoped equivalent). See `authz_scope.py`'s
module docstring for the full reasoning.

Note for Task 3 (ADK Agent/Runner wiring): Task 1's spike
(`test_adk_integration_spike.py`) only ever exercised a ZERO-argument tool
function (`get_current_server_time`) -- it never proved ADK's real schema
-introspection (`google.adk.tools.function_tool`'s `build_function_declaration`)
behaves correctly for a `functools.partial`-bound multi-argument tool like
the four defined here (where `session`/`principal` must be curried away
before the model ever sees a schema). Task 3 must empirically verify this
against the real installed package before assuming currying is sufficient,
and should also check whether ADK has a native `ToolContext`-style
injected-parameter mechanism (a real, first-class way to pass
caller-supplied context into a tool without it appearing in the model
-visible schema) before committing to `functools.partial` as the only
approach -- don't assume either way without checking the real source, per
this phase's own established "verify against the installed package, not
secondhand research" precedent (ADR 0003).
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolvegrid_api.agent_retrieval import retrieve_for_agent
from resolvegrid_api.authz_scope import apply_decision_scope, resource_in_scope
from resolvegrid_api.models.org import Employee
from resolvegrid_api.models.ticketing import AuditLog, Queue, Ticket, TicketMessage
from resolvegrid_api.operational_adapters.entitlements import lookup_employee_entitlements
from resolvegrid_api.retrieval_authz import build_authz_filter
from resolvegrid_authz import Decision, Principal, authorize


class DiagnosticToolError(Exception):
    """Base class for every error this module's diagnostic tools raise."""


class TicketNotFoundError(DiagnosticToolError):
    """Raised when a referenced `Ticket.id` does not exist at all."""


class TicketNotAuthorizedError(DiagnosticToolError):
    """Raised when a referenced ticket exists but falls outside the
    requesting principal's authorized scope (or the principal has no
    `ticket.list`/`ticket.view` access at all)."""


class EmployeeNotFoundError(DiagnosticToolError):
    """Raised when a referenced `Employee.id` does not exist at all."""


class EmployeeNotAuthorizedError(DiagnosticToolError):
    """Raised when a referenced employee exists but falls outside the
    requesting principal's authorized scope (or the principal has no
    `directory.view_employee` access at all)."""


# ---------------------------------------------------------------------------
# Ticket-scoping helpers specific to this module's fetch-then-check flow.
# The actual Decision-interpretation logic (what department_ids/employee_id
# mean for a query filter / an in-scope check) lives in
# `resolvegrid_api.authz_scope` (see that module's docstring) -- shared with
# `routers/tickets.py` and `routers/directory.py` rather than each keeping
# an independent copy of the same two rules.
# ---------------------------------------------------------------------------


def _ticket_department_id(session: Session, ticket: Ticket) -> int | None:
    queue = session.get(Queue, ticket.queue_id)
    return queue.department_id if queue else None


def _apply_ticket_scope(query, decision: Decision):
    if decision.department_ids is not None:
        query = query.join(Queue, Ticket.queue_id == Queue.id)
    return apply_decision_scope(
        query, decision, department_id_column=Queue.department_id, owner_id_column=Ticket.requester_id
    )


def _ticket_in_scope(decision: Decision, ticket: Ticket, department_id: int | None) -> bool:
    return resource_in_scope(decision, department_id=department_id, owner_id=ticket.requester_id)


def _require_ticket_in_scope(session: Session, decision: Decision, ticket_id: int) -> Ticket:
    """Fetch `ticket_id` and verify it is within `decision`'s allowed set,
    raising the appropriate not-found/not-authorized error otherwise.
    """
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise TicketNotFoundError(f"ticket {ticket_id} not found")
    department_id = _ticket_department_id(session, ticket)
    if not _ticket_in_scope(decision, ticket, department_id):
        raise TicketNotAuthorizedError(f"not authorized to access ticket {ticket_id}")
    return ticket


def _employee_in_scope(decision: Decision, employee: Employee) -> bool:
    return resource_in_scope(decision, department_id=employee.department_id, owner_id=employee.id)


# ---------------------------------------------------------------------------
# Tool 1: find similar/related past tickets
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Code-review follow-up (Minor): short ticket subjects are dominated by a
# handful of common function words ("for", "with", "to") that inflate
# Jaccard overlap between otherwise-unrelated tickets sharing the same
# category/type, without the specialist (or an analyst reading its output)
# having any way to tell a real content match from this noise. A small,
# fixed stopword set -- not an NLP dependency, just the highest-frequency
# English function words most likely to appear in a short ticket subject --
# is removed before scoring. Deliberately conservative (no stemming/lemma,
# no domain-specific exclusions): this is a precision improvement over the
# plan doc's explicitly-allowed simple heuristic, not a new modeling
# pipeline.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "for", "with", "to", "of", "in", "on", "at", "is",
        "are", "and", "or", "my", "it", "this", "that", "be", "has", "have",
    }
)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower())) - _STOPWORDS


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def find_similar_tickets(
    session: Session, principal: Principal, ticket_id: int, limit: int = 5
) -> list[dict]:
    """Find other real tickets with a similar issue to `ticket_id`, scoped to
    the requesting analyst's own authorized department(s).

    "Similar" heuristic (deliberate choice, not a placeholder): tickets have
    no existing embedding/vector-similarity infrastructure (unlike
    `Chunk`/`Document`, Phase 7) -- adding one purely for this tool would be
    new schema this phase's own "no new data model unless the existing
    schema proves insufficient" scope decision says to avoid. Instead this
    reuses the real `Ticket.category`/`Ticket.type` columns as a coarse
    candidate filter (same `category` if the anchor has one, else the same
    `type`), then ranks survivors by real `subject`-text word overlap
    (Jaccard similarity over lowercased, alphanumeric-tokenized word sets) --
    a fully deterministic, real, tested computation over real ticket rows,
    not a mocked/placeholder score.

    Authz: reuses `ticket.list`'s existing policy
    (`packages/authz/src/resolvegrid_authz/policy.py`) exactly, not a
    parallel mechanism. The anchor ticket itself must be in `principal`'s
    authorized scope (raises `TicketNotFoundError`/`TicketNotAuthorizedError`
    otherwise, mirroring `routers/tickets.py`'s `get_ticket`), and the
    candidate-pool query is scoped by the SAME `Decision` via
    `Queue.department_id` (mirrors `routers/tickets.py`'s
    `_apply_ticket_scope`) -- an out-of-scope ticket is excluded by the SQL
    query itself and never becomes a candidate row, let alone a returned
    result, regardless of how textually similar its subject is.

    Returns up to `limit` dicts (ticket fields + `similarity_score`), ordered
    by score descending, ties broken by ticket id ascending for
    deterministic output.
    """
    decision = authorize(principal, "ticket.list")
    if not decision.allowed:
        raise TicketNotAuthorizedError(decision.reason)

    anchor = _require_ticket_in_scope(session, decision, ticket_id)

    query = select(Ticket).where(Ticket.id != anchor.id)
    if anchor.category is not None:
        query = query.where(Ticket.category == anchor.category)
    else:
        query = query.where(Ticket.type == anchor.type)
    query = _apply_ticket_scope(query, decision)

    candidates = session.scalars(query).all()

    anchor_tokens = _tokenize(anchor.subject)
    scored = [
        (ticket, _jaccard_similarity(anchor_tokens, _tokenize(ticket.subject))) for ticket in candidates
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0].id))

    return [
        {
            "id": ticket.id,
            "subject": ticket.subject,
            "type": ticket.type,
            "category": ticket.category,
            "status": ticket.status,
            "priority": ticket.priority,
            "similarity_score": score,
        }
        for ticket, score in scored[:limit]
    ]


# ---------------------------------------------------------------------------
# Tool 2: look up an employee's entitlements
# ---------------------------------------------------------------------------


def get_employee_entitlements(session: Session, principal: Principal, employee_id: int) -> list[dict]:
    """Look up `employee_id`'s active (non-revoked) entitlements, scoped to
    the requesting analyst's own authorized department(s).

    Thin wrapper around Phase 9's existing, real, read-only
    `operational_adapters.entitlements.lookup_employee_entitlements` -- that
    function's own query/join logic is reused directly and unchanged, never
    reimplemented, per this task's explicit instruction.

    Authz: reuses `directory.view_employee`'s existing policy (the same
    action `routers/directory.py`'s `get_employee` already enforces for
    viewing an employee's own directory record) rather than inventing a new
    action -- an employee's entitlements are at least as sensitive as their
    directory record, and the existing department-scoping is exactly the
    boundary needed here too. Raises `EmployeeNotFoundError`/
    `EmployeeNotAuthorizedError` (mirroring `routers/directory.py`'s 404/403
    distinction) if `employee_id` doesn't exist or falls outside
    `principal`'s authorized scope -- entitlements for an out-of-scope
    employee are never returned, not even an empty list standing in for a
    silent denial.
    """
    decision = authorize(principal, "directory.view_employee")
    if not decision.allowed:
        raise EmployeeNotAuthorizedError(decision.reason)

    employee = session.get(Employee, employee_id)
    if employee is None:
        raise EmployeeNotFoundError(f"employee {employee_id} not found")
    if not _employee_in_scope(decision, employee):
        raise EmployeeNotAuthorizedError(f"not authorized to view employee {employee_id}")

    summaries = lookup_employee_entitlements(session, employee_id)
    return [
        {
            "entitlement_id": summary.entitlement_id,
            "entitlement_name": summary.entitlement_name,
            "access_group_name": summary.access_group_name,
            "granted_at": summary.granted_at.isoformat() if summary.granted_at else None,
        }
        for summary in summaries
    ]


# ---------------------------------------------------------------------------
# Tool 3: retrieve relevant knowledge-base evidence for a ticket's issue
# ---------------------------------------------------------------------------


def retrieve_ticket_kb_evidence(
    session: Session, principal: Principal, ticket_id: int, query_text: str | None = None
) -> dict:
    """Retrieve knowledge-base evidence relevant to `ticket_id`'s described
    issue, via the existing Phase 7/8 hybrid retrieval pipeline
    (`agent_retrieval.retrieve_for_agent`) -- reused directly, not a new
    search implementation, so the specialist cites real `[chunk:<id>]`
    evidence exactly like `/chat` already does.

    `query_text` defaults to the ticket's own subject plus its earliest
    message body (real ticket content) when not given explicitly -- a
    caller may override with a more targeted query while still defaulting
    to "the ticket's actual described issue", matching this tool's stated
    purpose.

    Authz, two independent layers (defense in depth -- neither layer
    substitutes for the other):
      1. The ticket itself must be in `principal`'s authorized scope
         (`ticket.view`, mirroring `routers/tickets.py`'s `get_ticket`
         exactly) -- a ticket's own content must never be usable as a
         retrieval query by an analyst who isn't authorized to view that
         ticket in the first place, independent of what the KB-side filter
         below would separately return.
      2. The actual KB chunks returned are independently authz-filtered by
         `retrieve_for_agent`'s own `scope` dict, built here via
         `build_authz_filter` into the exact same
         `{"unrestricted": ..., "allowed_tags": [...]}` shape
         `routers/chat.py` already builds for `/chat` -- applied inside
         `vector_search`/`lexical_search`'s SQL `WHERE` clause, so an
         unauthorized chunk is excluded by the query itself.

    Returns `retrieve_for_agent`'s own dict shape unchanged:
    `{"chunks": [...], "sufficient": bool, "context_block": str}`.
    """
    decision = authorize(principal, "ticket.view")
    if not decision.allowed:
        raise TicketNotAuthorizedError(decision.reason)

    ticket = _require_ticket_in_scope(session, decision, ticket_id)

    if query_text is None:
        first_message = session.scalars(
            select(TicketMessage)
            .where(TicketMessage.ticket_id == ticket.id)
            .order_by(TicketMessage.created_at)
        ).first()
        body = first_message.body if first_message is not None else ""
        query_text = f"{ticket.subject}\n{body}".strip()

    authz_filter = build_authz_filter(principal, session)
    scope = {"unrestricted": authz_filter.unrestricted, "allowed_tags": sorted(authz_filter.allowed_tags)}
    return retrieve_for_agent(query_text, scope)


# ---------------------------------------------------------------------------
# Tool 4: get a ticket's own current state/history
# ---------------------------------------------------------------------------


def get_ticket_state_and_history(session: Session, principal: Principal, ticket_id: int) -> dict:
    """Return `ticket_id`'s own current row plus its real `AuditLog` entries,
    authz-checked against the requesting analyst's department scope
    (mirrors `routers/tickets.py`'s `get_ticket` authz check exactly).

    Raises `TicketNotFoundError`/`TicketNotAuthorizedError` if the ticket
    doesn't exist or falls outside `principal`'s authorized scope -- the
    audit history of an out-of-scope ticket is never returned: `AuditLog.
    before_json`/`after_json` routinely contains the same ticket fields the
    ticket-scope check exists to protect in the first place, so this cannot
    be treated as a separately-authorized, lower-sensitivity resource.
    """
    decision = authorize(principal, "ticket.view")
    if not decision.allowed:
        raise TicketNotAuthorizedError(decision.reason)

    ticket = _require_ticket_in_scope(session, decision, ticket_id)

    audit_rows = session.scalars(
        select(AuditLog)
        .where(AuditLog.entity_type == "ticket", AuditLog.entity_id == ticket.id)
        .order_by(AuditLog.created_at)
    ).all()

    return {
        "ticket": {
            "id": ticket.id,
            "subject": ticket.subject,
            "type": ticket.type,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "queue_id": ticket.queue_id,
            "requester_id": ticket.requester_id,
            "assignee_id": ticket.assignee_id,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
        },
        "audit_log": [
            {
                "id": row.id,
                "action": row.action,
                "actor_type": row.actor_type,
                "actor_id": row.actor_id,
                "before_json": row.before_json,
                "after_json": row.after_json,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in audit_rows
        ],
    }
