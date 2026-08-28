from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion
from resolvegrid_api.llm_gateway import CompletionResult
from resolvegrid_api.main import app
from resolvegrid_api.models import AgentRun, Department, Employee, Location, Span
from resolvegrid_api.models.knowledge import Chunk, Document, DocumentVersion, Embedding, IngestionRun
from resolvegrid_api.seed_corpus import load_seed_corpus

_SEED_EMAIL = "chat.requester@example.test"
_SEED_DEPARTMENT_NAME = "Chat Test Dept"
_SEED_LOCATION_NAME = "Chat Test HQ"

# `/chat`'s `complete_fn` is injected as a lambda built ONCE inside main.py's
# lifespan (`complete_fn = lambda prompt: llm_gateway.complete(prompt).text`),
# and chat.py itself never imports `llm_gateway` as a name at all. Patching
# `resolvegrid_api.routers.chat.llm_gateway.complete` (this file's sibling
# test_ticket_summarize.py's pattern for tickets.py) is therefore a no-op
# here -- there is no such attribute to patch. The lambda's body does
# `llm_gateway.complete(prompt)`, which is a fresh attribute lookup on the
# `llm_gateway` *module object* at CALL time, so patching the module
# attribute itself -- `resolvegrid_api.llm_gateway.complete` -- is what every
# call actually sees, regardless of whether the patch was applied before or
# after `with TestClient(app) as client:` ran the lifespan and built the
# lambda. Empirically verified via a throwaway probe test before writing this
# file: patching `resolvegrid_api.llm_gateway.complete` and then entering
# `with TestClient(app) as client:` produced a 200 response and
# `mock_complete.call_count == 2` (classify_intent + compose_response both
# reached the mock) -- confirming the mechanism works.
_COMPLETE_PATCH_TARGET = "resolvegrid_api.llm_gateway.complete"


# A module-scoped client: `with TestClient(app) as client:` re-runs main.py's
# lifespan (init_tracing, AsyncPostgresSaver.setup(), build_graph(...)) every
# time it's entered. setup() is documented+verified idempotent, so re-running
# it per-test would have been safe too, but there's no need to pay that cost
# on every single test in this file -- one client, entered once for the
# whole module, is enough since each test's mocking is independent (done via
# `patch(...)` per test, not tied to client construction). Measured: this
# file's 4 tests + lifespan startup complete in ~2-3s total (see PR
# description / test run output), so this was a minor optimization, not a
# necessity -- per-test scoping would also have been fine.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _cleanup_agent_runs(raw_db_session):
    # Every test in this file creates a real AgentRun row with a real FK to
    # the get-or-create "chat.requester@example.test" Employee row (see
    # chat_fixtures below) -- unlike test_tickets_api.py's/test_ticket_
    # summarize.py's fixture employees, which are protected forever via a
    # real AuditLog.actor_id row that apps/api/src/resolvegrid_api/seed.py's
    # generate_org() already knows to check, seed.py's protected_employee_ids
    # set has no knowledge of agent_run.principal_employee_id -- that FK
    # (Phase 6) never had a test-created row pointing through it until this
    # file existed. Left uncleaned, a later test_seed.py run would hit
    # `ForeignKeyViolation: ... agent_run_principal_employee_id_fkey` trying
    # to wipe/reseed the Employee table (empirically confirmed while writing
    # this file). Fixing that is a seed.py production-code change, which is
    # out of this task's scope (adding a test file only) -- so instead this
    # fixture deletes the Span/AgentRun rows this file creates after each
    # test, keeping the whole blast radius inside this file.
    yield
    requester = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAIL))
    if requester is None:
        return
    run_ids = raw_db_session.scalars(
        select(AgentRun.id).where(AgentRun.principal_employee_id == requester.id)
    ).all()
    if run_ids:
        raw_db_session.execute(delete(Span).where(Span.agent_run_id.in_(run_ids)))
        raw_db_session.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        raw_db_session.commit()


@pytest.fixture
def chat_fixtures(raw_db_session):
    # Mirrors test_ticket_summarize.py's get-or-create-by-natural-key pattern:
    # AgentRun.principal_employee_id is a real FK to employee.id, so a
    # seeded Employee row must exist and must never be deleted out from
    # under earlier AgentRun rows that reference it.
    loc = raw_db_session.scalar(select(Location).where(Location.name == _SEED_LOCATION_NAME))
    if loc is None:
        loc = Location(name=_SEED_LOCATION_NAME, region="US", timezone="America/Chicago")
        raw_db_session.add(loc)
        raw_db_session.flush()

    dept = raw_db_session.scalar(select(Department).where(Department.name == _SEED_DEPARTMENT_NAME))
    if dept is None:
        dept = Department(name=_SEED_DEPARTMENT_NAME)
        raw_db_session.add(dept)
        raw_db_session.flush()

    requester = raw_db_session.scalar(select(Employee).where(Employee.email == _SEED_EMAIL))
    if requester is None:
        requester = Employee(
            display_name="Chat Requester", email=_SEED_EMAIL, title="Engineer",
            hire_date="2024-01-01T00:00:00", timezone=loc.timezone, location_id=loc.id, department_id=dept.id,
        )
        raw_db_session.add(requester)
        raw_db_session.flush()
    raw_db_session.commit()
    return requester


@pytest.fixture
def seed_corpus_ingested(raw_db_session):
    """Ingest `resolvegrid_api.seed_corpus.SEED_CORPUS` for a retrieval
    -focused chat test, then delete every row this fixture itself created,
    afterward -- mirrors `test_ingestion_worker.py`'s
    `test_arq_worker_processes_ingest_seed_corpus_task_via_real_redis`
    cleanup pattern for the same reason documented there:
    `apps/api/tests/test_retrieval.py`'s fixed-content lexical/vector
    -search tests assert *exact* result lists (e.g. `results ==
    [(chunk_dense_id, 1), (chunk_sparse_id, 2)]`) against this same live,
    shared dev database -- leaving seed-corpus rows behind permanently
    would risk an unrelated corpus chunk sneaking into one of those top-N
    result lists and breaking that equality assertion in a different test
    file entirely. Function-scoped (not module-scoped) so it cleans up
    after every test that uses it, not just once for the whole file.

    Cleanup is tracked by id-watermark (rows with id greater than what
    existed before this fixture ran), NOT by matching the seed corpus's
    titles -- title-matching deletes rows this fixture didn't create
    whenever the dev DB already has the corpus permanently ingested
    (Phase 7 Task 10's deliberate final-state choice, see
    docs/PROGRESS.md's Phase 7 row): `ingest_document`'s idempotency means
    re-ingesting an already-present corpus creates zero new rows (a title
    match resolves to the existing Document/DocumentVersion), so a
    title-based cleanup would silently destroy that deliberately
    -persisted corpus as a side effect of merely running this test (caught
    for real during Task 10's own fresh-state verification).
    """
    before_max_run_id = raw_db_session.scalar(select(func.max(IngestionRun.id))) or 0
    before_max_document_id = raw_db_session.scalar(select(func.max(Document.id))) or 0
    before_max_version_id = raw_db_session.scalar(select(func.max(DocumentVersion.id))) or 0
    before_max_chunk_id = raw_db_session.scalar(select(func.max(Chunk.id))) or 0
    before_max_embedding_id = raw_db_session.scalar(select(func.max(Embedding.id))) or 0
    run_seed_corpus_ingestion(raw_db_session)
    raw_db_session.commit()
    try:
        yield
    finally:
        embedding_ids = raw_db_session.scalars(
            select(Embedding.id).where(Embedding.id > before_max_embedding_id)
        ).all()
        chunk_ids = raw_db_session.scalars(select(Chunk.id).where(Chunk.id > before_max_chunk_id)).all()
        version_ids = raw_db_session.scalars(
            select(DocumentVersion.id).where(DocumentVersion.id > before_max_version_id)
        ).all()
        document_ids = raw_db_session.scalars(
            select(Document.id).where(Document.id > before_max_document_id)
        ).all()
        raw_db_session.execute(delete(Embedding).where(Embedding.id.in_(embedding_ids)))
        raw_db_session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))
        raw_db_session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_(version_ids)))
        raw_db_session.execute(delete(Document).where(Document.id.in_(document_ids)))
        raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id > before_max_run_id))
        raw_db_session.commit()


def _classification_result(intent: str = "general_question", risk_level: str = "low") -> CompletionResult:
    # graph.py's classify_intent node does `json.loads(complete_fn(prompt))`
    # then validates the shape with IntentClassification -- this must be
    # exactly that JSON shape, no markdown fences, no extra text.
    return CompletionResult(
        text=f'{{"intent": "{intent}", "risk_level": "{risk_level}"}}',
        input_tokens=10, output_tokens=5, latency_ms=5,
        provider="ollama", model="local-qwen3",
    )


def _answer_result(text: str) -> CompletionResult:
    return CompletionResult(
        text=text, input_tokens=20, output_tokens=15, latency_ms=8,
        provider="ollama", model="local-qwen3",
    )


def test_chat_success_writes_agent_run_and_four_success_spans(chat_fixtures, raw_db_session, client):
    requester = chat_fixtures
    message = "What is a VPN?"

    # Two-call pattern matching the real graph: classify_intent calls
    # complete_fn once (expects JSON), compose_response calls it again
    # (expects plain answer text). Exercising the "well-formed classification"
    # path here (not the classify-degrades-to-unclear/low path) since that's
    # the more informative one to assert an intent/risk_level round-trip on;
    # the malformed-JSON degrade path is graph.py's own unit-tested behavior,
    # not this endpoint's contract to re-prove.
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("A VPN is a Virtual Private Network.")],
    ) as mock_complete:
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A VPN is a Virtual Private Network."
    assert isinstance(body["thread_id"], str) and body["thread_id"]
    assert mock_complete.call_count == 2

    # chat_fixtures's requester belongs to "Chat Test Dept" -- not part of
    # the real seed corpus, and this test doesn't ingest it (see
    # `seed_corpus_ingested` fixture) -- so retrieval finds nothing to
    # cite here, and the response degrades to the general-knowledge
    # caption/empty citations, per chat.py's documented branch condition.
    assert body["citations"] == []
    assert "General-knowledge answer" in body["caption"]

    run = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body["thread_id"]))
    assert run is not None
    assert run.status == "completed"
    assert run.input_text == message
    assert run.output_text == "A VPN is a Virtual Private Network."
    assert run.principal_employee_id == requester.id
    assert run.error_message is None
    assert run.completed_at is not None

    spans = raw_db_session.scalars(
        select(Span).where(Span.agent_run_id == run.id).order_by(Span.id)
    ).all()
    assert [s.stage_name for s in spans] == ["classify_intent", "retrieve", "compose_response", "finalize"]
    assert len(spans) == 4
    assert all(s.status == "success" for s in spans)


def test_chat_gateway_error_returns_502_and_records_error(chat_fixtures, raw_db_session, client):
    requester = chat_fixtures
    message = "trigger a completion failure please"

    # complete_fn's exception surfaces from inside classify_intent: that
    # node's try/except only catches (json.JSONDecodeError, ValidationError,
    # TypeError) -- a RuntimeError from complete_fn is NOT one of those, so
    # it propagates out of the node, out of ainvoke(), and into chat.py's
    # broad `except Exception` handler, which is exactly the 502 contract
    # under test here (not an LLMGatewayError-specific path -- chat.py's
    # docstring is explicit that it catches failures from any layer).
    with patch(_COMPLETE_PATCH_TARGET, side_effect=RuntimeError("boom")) as mock_complete:
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 502
    assert "boom" in response.json()["detail"]
    mock_complete.assert_called_once()

    run = raw_db_session.scalar(
        select(AgentRun)
        .where(AgentRun.input_text == message, AgentRun.principal_employee_id == requester.id)
        .order_by(AgentRun.id.desc())
    )
    assert run is not None
    assert run.status == "error"
    assert run.error_message == "boom"
    assert run.output_text is None
    assert run.completed_at is None

    # No Span rows on the error path -- chat.py only writes the 4 stage
    # Spans after a successful ainvoke() call.
    spans = raw_db_session.scalars(select(Span).where(Span.agent_run_id == run.id)).all()
    assert spans == []


def test_chat_two_calls_get_different_thread_ids(chat_fixtures, client):
    requester = chat_fixtures

    def _call(message: str) -> str:
        with patch(
            _COMPLETE_PATCH_TARGET,
            side_effect=[_classification_result(), _answer_result(f"Answer to: {message}")],
        ):
            response = client.post(
                "/chat",
                json={"message": message},
                headers={"X-Debug-Employee-Id": str(requester.id)},
            )
        assert response.status_code == 200
        return response.json()["thread_id"]

    thread_id_1 = _call("first message")
    thread_id_2 = _call("second message")
    assert thread_id_1 != thread_id_2


# ---------------------------------------------------------------------------
# Retrieval-focused chat tests (Phase 7 Task 7)
# ---------------------------------------------------------------------------


def test_chat_retrieves_and_cites_public_seed_corpus_chunk(chat_fixtures, seed_corpus_ingested, client):
    """End-to-end: with the real seed corpus ingested, a question that
    matches the corpus's public "What Is a VPN" reference doc should come
    back with a real citation to it -- proving retrieve_for_agent's real
    vector+lexical+RRF pipeline, not just the graph's plumbing, actually
    finds and threads through real ingested chunks.

    Uses `chat_fixtures`'s requester, whose home department ("Chat Test
    Dept") is NOT one of the seed corpus's real department tags -- this
    deliberately proves the PUBLIC document is retrievable regardless of
    department (per `retrieval_authz.py`'s "empty access_scope_tags =
    visible to everyone" convention), without needing a department-scoped
    fixture employee.
    """
    requester = chat_fixtures
    message = "What is a VPN?"

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[
            _classification_result(),
            _answer_result("A VPN creates an encrypted tunnel for remote access [chunk:1]."),
        ],
    ) as mock_complete:
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    assert mock_complete.call_count == 2
    body = response.json()

    assert body["citations"], "expected at least one citation from the ingested seed corpus"
    titles = {c["document_title"] for c in body["citations"]}
    assert "What Is a VPN (Public Reference)" in titles
    assert all("chunk_id" in c and "document_title" in c for c in body["citations"])
    assert "knowledge base" in body["caption"].lower()
    assert "citation" in body["caption"].lower() or "citations" in body["caption"].lower()

    # The 2nd complete_fn call (compose_response's prompt) must have
    # actually carried the citation context, proving the graph's
    # `retrieve` -> `compose_response` wiring, not just chat.py's response
    # shaping, is what's under test here.
    compose_prompt = mock_complete.call_args_list[1].args[0]
    assert "[chunk:" in compose_prompt
    assert "What Is a VPN (Public Reference)" in compose_prompt


def test_chat_authz_filters_out_department_scoped_seed_corpus_chunk(
    chat_fixtures, seed_corpus_ingested, client
):
    """Adversarial-style leakage check: `chat_fixtures`'s requester belongs
    to "Chat Test Dept", which is NOT the seed corpus's
    "platform_engineering" tag -- the platform-engineering-scoped on-call
    escalation policy must never appear in this requester's citations,
    even though it exists in the now-ingested corpus and the query below
    is worded specifically to match its content. The response must still
    degrade sensibly (200, some answer, no crash), not fail outright.
    """
    requester = chat_fixtures
    message = "What is the Kestrel Platform Engineering on-call escalation policy?"

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("I don't have that information.")],
    ):
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    body = response.json()
    titles = {c["document_title"] for c in body["citations"]}
    assert "Kestrel Platform Engineering On-Call Escalation Policy" not in titles
    assert body["answer"] == "I don't have that information."
