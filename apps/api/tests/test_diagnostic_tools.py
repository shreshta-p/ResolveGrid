"""Tests for `resolvegrid_api.diagnostic_tools` (Phase 13 Task 2).

Each tool gets a real, adversarial authz-boundary test engineered so the
out-of-scope data is the OBJECTIVELY BEST possible match/candidate for the
request (same rigor bar as `test_retrieval.py`'s
`test_adversarial_authz_filter_blocks_cross_scope_leakage`) -- a sanity
check proves the leakage candidate really would win if unscoped, so the
later assertion that it's absent/denied for the scoped principal is proof
the authz filter is doing real work, not an accident of ordering or an
empty result set.

`find_similar_tickets`/`get_employee_entitlements`/`get_ticket_state_and_
history` all take `session` directly and run their query through that same
session, so the standard rollback-based `db_session` fixture is sufficient.
`retrieve_ticket_kb_evidence` is different: it delegates to `agent_
retrieval.retrieve_for_agent`, which opens its OWN session via
`resolvegrid_api.db.session_factory()` (a brand-new connection) rather than
reusing whatever session is passed to `retrieve_ticket_kb_evidence` -- so
test data for that tool must be committed on a real, separate session
(`raw_db_session`) for the separate connection to see it, with explicit
cleanup afterwards (mirrors `test_tickets_api.py`'s `ticket_fixtures`
precedent for the same "shared commit-based fixture, manual cleanup"
shape).
"""

import uuid

from sqlalchemy import delete, select

from resolvegrid_api.diagnostic_tools import (
    EmployeeNotAuthorizedError,
    EmployeeNotFoundError,
    TicketNotAuthorizedError,
    TicketNotFoundError,
    find_similar_tickets,
    get_employee_entitlements,
    get_ticket_state_and_history,
    retrieve_ticket_kb_evidence,
)
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding
from resolvegrid_api.models.org import (
    AccessGroup,
    Department,
    Employee,
    EmployeeEntitlement,
    Entitlement,
    Location,
)
from resolvegrid_api.models.ticketing import AuditLog, Queue, Ticket
from resolvegrid_api.operational_adapters.entitlements import grant_vpn_access
from resolvegrid_authz import Principal, RoleGrant
from resolvegrid_retrieval.chunker import ChunkRecord
from resolvegrid_retrieval.embedder import DEFAULT_EMBEDDING_MODEL, embed_texts

EMBEDDING_DIM = 768


def _make_location_and_departments(session, suffix: str):
    location = Location(name=f"Diag Test HQ {suffix}", region="US", timezone="America/Chicago")
    dept_eng = Department(name=f"Diag Eng {suffix}")
    dept_hr = Department(name=f"Diag HR {suffix}")
    session.add_all([location, dept_eng, dept_hr])
    session.flush()
    return location, dept_eng, dept_hr


def _make_queue(session, name: str, department_id: int) -> Queue:
    queue = Queue(name=name, department_id=department_id)
    session.add(queue)
    session.flush()
    return queue


def _make_employee(session, suffix: str, *, location, department) -> Employee:
    employee = Employee(
        display_name=f"Diag Employee {suffix}",
        email=f"diag.{suffix}.{uuid.uuid4().hex[:8]}@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone=location.timezone,
        location_id=location.id,
        department_id=department.id,
    )
    session.add(employee)
    session.flush()
    return employee


def _make_ticket(session, *, subject, category, type_, queue_id, requester_id) -> Ticket:
    ticket = Ticket(
        subject=subject, category=category, type=type_, queue_id=queue_id, requester_id=requester_id
    )
    session.add(ticket)
    session.flush()
    return ticket


def _analyst_principal(employee_id: int, department_id: int) -> Principal:
    return Principal(
        employee_id=employee_id,
        roles=(RoleGrant(role="analyst", scope="department", scope_id=department_id),),
    )


def _admin_principal(employee_id: int) -> Principal:
    return Principal(employee_id=employee_id, roles=(RoleGrant(role="admin", scope="global"),))


# ---------------------------------------------------------------------------
# Tool 1: find_similar_tickets
# ---------------------------------------------------------------------------


def test_find_similar_tickets_adversarial_blocks_cross_department_leakage(db_session):
    """An HR ticket engineered to be a PERFECT textual match (identical
    subject, same category) for the Engineering-scoped analyst's anchor
    ticket must never appear in that analyst's results -- even though an
    unrestricted view would rank it #1.
    """
    location, dept_eng, dept_hr = _make_location_and_departments(db_session, "find-similar")
    queue_eng = _make_queue(db_session, "Diag Eng Queue find-similar", dept_eng.id)
    queue_hr = _make_queue(db_session, "Diag HR Queue find-similar", dept_hr.id)
    requester = _make_employee(db_session, "requester", location=location, department=dept_eng)

    anchor = _make_ticket(
        db_session,
        subject="VPN connection drops frequently for remote worker",
        category="network",
        type_="incident",
        queue_id=queue_eng.id,
        requester_id=requester.id,
    )
    # Weak match, correct department: should be a legitimate low-ranked result.
    eng_candidate = _make_ticket(
        db_session,
        subject="Printer on third floor is out of toner",
        category="network",
        type_="incident",
        queue_id=queue_eng.id,
        requester_id=requester.id,
    )
    # Adversarial: identical subject (Jaccard similarity 1.0 -- the best
    # possible score), same category, but in the DISJOINT HR department.
    hr_candidate = _make_ticket(
        db_session,
        subject="VPN connection drops frequently for remote worker",
        category="network",
        type_="incident",
        queue_id=queue_hr.id,
        requester_id=requester.id,
    )

    admin_principal = _admin_principal(requester.id)
    unrestricted_results = find_similar_tickets(db_session, admin_principal, anchor.id)
    # Sanity check: unfiltered, the HR ticket really does win outright.
    assert unrestricted_results[0]["id"] == hr_candidate.id
    assert unrestricted_results[0]["similarity_score"] == 1.0

    eng_principal = _analyst_principal(requester.id, dept_eng.id)
    scoped_results = find_similar_tickets(db_session, eng_principal, anchor.id)

    result_ids = {r["id"] for r in scoped_results}
    assert hr_candidate.id not in result_ids
    assert eng_candidate.id in result_ids


def test_find_similar_tickets_raises_for_out_of_scope_anchor_ticket(db_session):
    location, dept_eng, dept_hr = _make_location_and_departments(db_session, "anchor-scope")
    queue_hr = _make_queue(db_session, "Diag HR Queue anchor-scope", dept_hr.id)
    requester = _make_employee(db_session, "requester", location=location, department=dept_hr)

    hr_ticket = _make_ticket(
        db_session,
        subject="Payroll question about benefits enrollment",
        category="hr",
        type_="request",
        queue_id=queue_hr.id,
        requester_id=requester.id,
    )

    eng_principal = _analyst_principal(requester.id, dept_eng.id)
    try:
        find_similar_tickets(db_session, eng_principal, hr_ticket.id)
        assert False, "expected TicketNotAuthorizedError"
    except TicketNotAuthorizedError:
        pass


def test_find_similar_tickets_raises_for_nonexistent_anchor_ticket(db_session):
    location, dept_eng, _dept_hr = _make_location_and_departments(db_session, "missing-anchor")
    requester = _make_employee(db_session, "requester", location=location, department=dept_eng)
    eng_principal = _analyst_principal(requester.id, dept_eng.id)

    try:
        find_similar_tickets(db_session, eng_principal, 999_999_999)
        assert False, "expected TicketNotFoundError"
    except TicketNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Tool 2: get_employee_entitlements
# ---------------------------------------------------------------------------


def test_get_employee_entitlements_blocks_cross_department_lookup(db_session):
    location, dept_eng, dept_hr = _make_location_and_departments(db_session, "entitlements")
    eng_employee = _make_employee(db_session, "eng", location=location, department=dept_eng)
    hr_employee = _make_employee(db_session, "hr", location=location, department=dept_hr)

    # Grant the HR employee a real entitlement -- the thing a leak would expose.
    grant_vpn_access(db_session, hr_employee.id, justification="hr onboarding")
    db_session.flush()

    # Sanity check: admin really does see it.
    admin_principal = _admin_principal(eng_employee.id)
    admin_view = get_employee_entitlements(db_session, admin_principal, hr_employee.id)
    assert len(admin_view) == 1
    assert admin_view[0]["entitlement_name"] == "VPN Access"

    eng_principal = _analyst_principal(eng_employee.id, dept_eng.id)
    try:
        get_employee_entitlements(db_session, eng_principal, hr_employee.id)
        assert False, "expected EmployeeNotAuthorizedError"
    except EmployeeNotAuthorizedError:
        pass

    # Same-department lookup must still work.
    own_dept_view = get_employee_entitlements(db_session, eng_principal, eng_employee.id)
    assert own_dept_view == []


def test_get_employee_entitlements_raises_for_nonexistent_employee(db_session):
    location, dept_eng, _dept_hr = _make_location_and_departments(db_session, "missing-employee")
    requester = _make_employee(db_session, "requester", location=location, department=dept_eng)
    eng_principal = _analyst_principal(requester.id, dept_eng.id)

    try:
        get_employee_entitlements(db_session, eng_principal, 999_999_999)
        assert False, "expected EmployeeNotFoundError"
    except EmployeeNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Tool 3: retrieve_ticket_kb_evidence
# ---------------------------------------------------------------------------


def test_retrieve_ticket_kb_evidence_blocks_cross_department_ticket_access(db_session):
    """An eng-scoped analyst must not be able to use an out-of-scope HR
    ticket's content as a retrieval query at all (layer 1 -- the ticket
    -level authz check), regardless of what the KB-side filter would
    separately do.
    """
    location, dept_eng, dept_hr = _make_location_and_departments(db_session, "kb-ticket-scope")
    queue_hr = _make_queue(db_session, "Diag HR Queue kb-ticket-scope", dept_hr.id)
    requester = _make_employee(db_session, "requester", location=location, department=dept_hr)

    hr_ticket = _make_ticket(
        db_session,
        subject="Benefits enrollment question",
        category="hr",
        type_="request",
        queue_id=queue_hr.id,
        requester_id=requester.id,
    )

    eng_principal = _analyst_principal(requester.id, dept_eng.id)
    try:
        retrieve_ticket_kb_evidence(db_session, eng_principal, hr_ticket.id)
        assert False, "expected TicketNotAuthorizedError"
    except TicketNotAuthorizedError:
        pass


def test_retrieve_ticket_kb_evidence_adversarial_blocks_cross_department_chunk_leakage(raw_db_session):
    """Real, non-mocked proof of layer 2: even for a ticket the analyst IS
    authorized to view, the KB chunk evidence returned must never include a
    chunk belonging to a document scoped to a DIFFERENT department -- even
    one engineered to be the single best possible vector AND lexical match
    for the real query text actually derived from the ticket.

    Uses `raw_db_session` (a real, committing session), not the rollback
    -based `db_session` fixture: `retrieve_ticket_kb_evidence` delegates to
    `agent_retrieval.retrieve_for_agent`, which opens its own, separate DB
    connection via `session_factory()` -- it would never see uncommitted
    rows written only on a different, still-open transaction. Real cleanup
    happens in the `finally` block, matching `test_tickets_api.py`'s
    `ticket_fixtures`/`test_adk_integration_spike.py`'s manual-cleanup
    precedent for real, committed test data in a shared dev database.
    """
    suffix = uuid.uuid4().hex[:8]
    session = raw_db_session

    location, dept_eng, dept_hr = _make_location_and_departments(session, f"kb-chunk-{suffix}")
    queue_eng = _make_queue(session, f"Diag Eng Queue kb-chunk-{suffix}", dept_eng.id)
    requester = _make_employee(session, f"kb-requester-{suffix}", location=location, department=dept_eng)

    anchor_subject = f"quarterly headcount budget planning figures {suffix}"
    anchor = _make_ticket(
        session,
        subject=anchor_subject,
        category="finance",
        type_="request",
        queue_id=queue_eng.id,
        requester_id=requester.id,
    )
    session.commit()

    eng_doc = Document(
        source_type="synthetic_private",
        title=f"Engineering KB doc {suffix}",
        checksum=f"diag-eng-checksum-{suffix}",
        access_scope_tags=[f"diag_eng_{suffix}"],
    )
    hr_doc = Document(
        source_type="synthetic_private",
        title=f"HR KB doc {suffix}",
        checksum=f"diag-hr-checksum-{suffix}",
        access_scope_tags=[f"diag_hr_{suffix}"],
    )
    session.add_all([eng_doc, hr_doc])
    session.flush()

    eng_version = DocumentVersion(
        document_id=eng_doc.id, parser_version="v1", chunking_version="v1", content_hash=f"eng-hash-{suffix}"
    )
    hr_version = DocumentVersion(
        document_id=hr_doc.id, parser_version="v1", chunking_version="v1", content_hash=f"hr-hash-{suffix}"
    )
    session.add_all([eng_version, hr_version])
    session.flush()

    # Real embedding of the real query text this tool will actually derive
    # from the ticket (subject, no messages) -- a real Ollama call, not a
    # hand-waved vector, so the adversarial chunk's vector below is a
    # genuine, real cosine-distance-0 match for the pipeline's real query
    # embedding, not merely assumed to be.
    real_query_vector = embed_texts([anchor_subject], model=DEFAULT_EMBEDDING_MODEL)[0]
    worst_vector = [-x for x in real_query_vector]

    eng_chunk = Chunk(
        document_version_id=eng_version.id,
        ordinal=0,
        text="Unrelated deployment runbook for the API gateway service.",
        token_count=8,
    )
    hr_chunk = Chunk(
        document_version_id=hr_version.id,
        ordinal=0,
        text=anchor_subject,  # guaranteed best possible lexical match
        token_count=len(anchor_subject.split()),
    )
    session.add_all([eng_chunk, hr_chunk])
    session.flush()

    session.add_all(
        [
            Embedding(
                chunk_id=eng_chunk.id, embedding_model="test-model", embedding_version="v1", vector=worst_vector
            ),
            Embedding(
                chunk_id=hr_chunk.id,
                embedding_model="test-model",
                embedding_version="v1",
                vector=real_query_vector,  # guaranteed best possible vector match
            ),
        ]
    )
    session.commit()

    try:
        # Sanity check: an admin (unrestricted) really does get the HR chunk
        # back, proving it genuinely is the best possible match -- not that
        # the test accidentally can't find it at all.
        admin_principal = _admin_principal(requester.id)
        admin_result = retrieve_ticket_kb_evidence(session, admin_principal, anchor.id)
        admin_chunk_ids = {c["chunk_id"] for c in admin_result["chunks"]}
        assert hr_chunk.id in admin_chunk_ids, "adversarial HR chunk did not win unfiltered -- test setup is broken"

        eng_principal = _analyst_principal(requester.id, dept_eng.id)
        eng_result = retrieve_ticket_kb_evidence(session, eng_principal, anchor.id)
        eng_chunk_ids = {c["chunk_id"] for c in eng_result["chunks"]}
        assert hr_chunk.id not in eng_chunk_ids
    finally:
        session.execute(delete(Embedding).where(Embedding.chunk_id.in_([eng_chunk.id, hr_chunk.id])))
        session.execute(delete(Chunk).where(Chunk.id.in_([eng_chunk.id, hr_chunk.id])))
        session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_([eng_version.id, hr_version.id])))
        session.execute(delete(Document).where(Document.id.in_([eng_doc.id, hr_doc.id])))
        session.execute(delete(AuditLog).where(AuditLog.entity_type == "ticket", AuditLog.entity_id == anchor.id))
        session.execute(delete(Ticket).where(Ticket.id == anchor.id))
        session.execute(delete(Queue).where(Queue.id == queue_eng.id))
        session.execute(
            delete(Employee).where(Employee.id == requester.id)
        )
        session.execute(delete(Department).where(Department.id.in_([dept_eng.id, dept_hr.id])))
        session.execute(delete(Location).where(Location.id == location.id))
        session.commit()


# ---------------------------------------------------------------------------
# Tool 4: get_ticket_state_and_history
# ---------------------------------------------------------------------------


def test_get_ticket_state_and_history_blocks_cross_department_access(db_session):
    location, dept_eng, dept_hr = _make_location_and_departments(db_session, "history")
    queue_hr = _make_queue(db_session, "Diag HR Queue history", dept_hr.id)
    requester = _make_employee(db_session, "requester", location=location, department=dept_hr)

    hr_ticket = _make_ticket(
        db_session,
        subject="Sensitive HR grievance ticket",
        category="hr",
        type_="request",
        queue_id=queue_hr.id,
        requester_id=requester.id,
    )
    db_session.add(
        AuditLog(
            actor_type="employee",
            actor_id=requester.id,
            action="ticket.create",
            entity_type="ticket",
            entity_id=hr_ticket.id,
            after_json='{"subject": "Sensitive HR grievance ticket"}',
            record_hash=f"diag-hist-hash-{uuid.uuid4().hex[:8]}",
        )
    )
    db_session.flush()

    # Sanity check: admin really does see the ticket + its history.
    admin_principal = _admin_principal(requester.id)
    admin_view = get_ticket_state_and_history(db_session, admin_principal, hr_ticket.id)
    assert admin_view["ticket"]["id"] == hr_ticket.id
    assert len(admin_view["audit_log"]) == 1

    eng_principal = _analyst_principal(requester.id, dept_eng.id)
    try:
        get_ticket_state_and_history(db_session, eng_principal, hr_ticket.id)
        assert False, "expected TicketNotAuthorizedError"
    except TicketNotAuthorizedError:
        pass


def test_get_ticket_state_and_history_raises_for_nonexistent_ticket(db_session):
    location, dept_eng, _dept_hr = _make_location_and_departments(db_session, "missing-history")
    requester = _make_employee(db_session, "requester", location=location, department=dept_eng)
    eng_principal = _analyst_principal(requester.id, dept_eng.id)

    try:
        get_ticket_state_and_history(db_session, eng_principal, 999_999_999)
        assert False, "expected TicketNotFoundError"
    except TicketNotFoundError:
        pass
