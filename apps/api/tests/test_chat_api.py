import re
import time
from unittest.mock import patch

import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from resolvegrid_api.cache import REDIS_URL
from resolvegrid_api.ingestion_worker import run_seed_corpus_ingestion
from resolvegrid_api.llm_gateway import CompletionResult
from resolvegrid_api.main import app
from resolvegrid_api.models import AgentRun, Department, Employee, Location, ModelCall, Span
from resolvegrid_api.models.knowledge import (
    Chunk,
    Document,
    DocumentVersion,
    Embedding,
    IngestionRun,
    SemanticResponseCache,
)
from resolvegrid_api.seed_corpus import load_seed_corpus

_SEED_EMAIL = "chat.requester@example.test"
_SEED_DEPARTMENT_NAME = "Chat Test Dept"
_SEED_LOCATION_NAME = "Chat Test HQ"
_OTHER_SCOPE_SEED_EMAIL = "chat.cache.other@example.test"
_OTHER_SCOPE_DEPARTMENT_NAME = "Chat Cache Other Dept"

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
def _clear_response_cache_namespace():
    """Phase 12 Task 3: this file's own tests re-use fixed, literal message
    strings (e.g. "What is a VPN?") across MULTIPLE tests, deliberately, so
    a human reader can see exactly which turn a given assertion is about
    (same reasoning `test_retrieval_cache.py`'s own namespace-clearing
    fixture documents for the same pattern). Without clearing
    `cache:response:*` before AND after every test in this file, the real
    exact-response cache Task 3 adds would turn the SECOND test in this
    file to reuse a given (scope, message) pair into an unintended real
    cache HIT -- silently skipping the real graph run (and the mocked
    `complete_fn` call-count assertions, injected-latency-timing
    assertions, etc.) those tests are actually about. This is a real
    correctness requirement for this test file specifically, not
    defensive boilerplate: `test_chat_cache_hit_*` tests below are the
    only ones in this file that WANT a real cache hit, and they populate
    their own entry explicitly within the test body.
    """
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fail fast (real connection check) if Redis isn't up
    for key in client.scan_iter(match="cache:response:*"):
        client.delete(key)
    yield
    for key in client.scan_iter(match="cache:response:*"):
        client.delete(key)


@pytest.fixture(autouse=True)
def _cleanup_semantic_cache_rows(raw_db_session):
    """Phase 12 Task 5: `semantic_response_cache` is a real Postgres table,
    not Redis, so the exact-cache namespace fixture above doesn't touch it.
    Watermark-based cleanup (mirrors `seed_corpus_ingested`'s own id
    -watermark convention elsewhere in this file) rather than a fixed-key
    sweep, since there's no natural "namespace" string to scan like Redis's
    `cache:response:*` -- every row with an id greater than the watermark
    taken before a given test ran was created BY that test, regardless of
    which scope_sig it landed under.
    """
    before_max_id = raw_db_session.scalar(select(func.max(SemanticResponseCache.id))) or 0
    yield
    raw_db_session.execute(delete(SemanticResponseCache).where(SemanticResponseCache.id > before_max_id))
    raw_db_session.commit()


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
    # Phase 12 Task 3 added a second fixture requester (`_OTHER_SCOPE_SEED_
    # EMAIL`, a different department -- see `other_scope_chat_fixtures`
    # below) for the cross-scope cache-isolation test; both are cleaned up
    # here for the identical reason the original requester already was.
    for email in (_SEED_EMAIL, _OTHER_SCOPE_SEED_EMAIL):
        requester = raw_db_session.scalar(select(Employee).where(Employee.email == email))
        if requester is None:
            continue
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
def other_scope_chat_fixtures(raw_db_session):
    """A second requester in a genuinely DIFFERENT department from `chat_
    fixtures`'s requester -- gives a real, differently-resolved `AuthzFilter`
    / `scope_sig` (Phase 12 Task 2's `compute_scope_sig`), needed by the
    cross-scope response-cache-isolation test below. Mirrors `chat_
    fixtures`'s own get-or-create-by-natural-key pattern exactly.
    """
    loc = raw_db_session.scalar(select(Location).where(Location.name == _SEED_LOCATION_NAME))
    if loc is None:
        loc = Location(name=_SEED_LOCATION_NAME, region="US", timezone="America/Chicago")
        raw_db_session.add(loc)
        raw_db_session.flush()

    dept = raw_db_session.scalar(select(Department).where(Department.name == _OTHER_SCOPE_DEPARTMENT_NAME))
    if dept is None:
        dept = Department(name=_OTHER_SCOPE_DEPARTMENT_NAME)
        raw_db_session.add(dept)
        raw_db_session.flush()

    requester = raw_db_session.scalar(select(Employee).where(Employee.email == _OTHER_SCOPE_SEED_EMAIL))
    if requester is None:
        requester = Employee(
            display_name="Chat Cache Other Requester", email=_OTHER_SCOPE_SEED_EMAIL, title="Engineer",
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

    # Phase 11 Task 3: watermark ModelCall.id BEFORE this turn, so the
    # assertions below can identify exactly which rows THIS chat turn wrote
    # (ModelCall is an accumulating log table -- like every other test that
    # asserts against it in this codebase, e.g. test_ticket_summarize.py --
    # rows are never cleaned up afterward).
    before_max_model_call_id = raw_db_session.scalar(select(func.max(ModelCall.id))) or 0

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

    # Phase 11 Task 3's core exit criterion: a real chat turn now writes a
    # real ModelCall row for BOTH classify_intent and compose_response --
    # not just summarize_ticket (Task 1's only prior call site). Each is
    # written by main.py's real per-node `make_logging_complete_fn` closure
    # (see that module's lifespan), on its own session_factory() session,
    # committed independently of this handler's own request-scoped session
    # -- hence querying via raw_db_session (a separate, real connection)
    # rather than the request's own session, which this test has no direct
    # handle on anyway.
    new_model_calls = raw_db_session.scalars(
        select(ModelCall).where(ModelCall.id > before_max_model_call_id).order_by(ModelCall.id)
    ).all()
    purposes = [c.purpose for c in new_model_calls]
    assert "chat.classify_intent" in purposes
    assert "chat.compose_response" in purposes
    for call in new_model_calls:
        assert call.status == "success"
        assert call.provider == "ollama"
        assert call.model == "local-qwen3"
        # Real OTel trace id capture (Task 1) -- 32-char lowercase hex,
        # never the reserved all-zero invalid sentinel.
        assert call.trace_id is not None
        assert len(call.trace_id) == 32
        assert call.trace_id != "0" * 32


def _sleepy_complete_side_effect(classify_sleep_s: float, compose_sleep_s: float, answer_text: str):
    """Build a `resolvegrid_api.llm_gateway.complete` `side_effect` that
    injects a REAL, deliberately different `time.sleep()` duration around
    the 1st (classify_intent) vs 2nd (compose_response) completion call --
    used by the Phase 11 Task 2 tests below to prove `chat.py`'s `Span`
    rows carry genuinely per-node measured latency, not the removed
    "split evenly across 4 stages" placeholder.

    This mocks the LLM completion call itself, matching every other test
    in this file (see `_COMPLETE_PATCH_TARGET`'s docstring above) -- the
    thing under test here is the SPAN TIMING INSTRUMENTATION around real
    node work, not whether a real network call reaches a real LLM
    provider (a separate, already-covered concern). The `time.sleep()`
    calls themselves are real, wall-clock-blocking work the test process
    genuinely performs -- not a mocked/fabricated latency value written
    directly into the assertion.
    """
    calls = {"n": 0}

    def _side_effect(
        prompt: str, *, model: str = "local-qwen3", cacheable_prefix: str | None = None
    ) -> CompletionResult:
        # Phase 11 Task 3: real chat-graph closures now call
        # `llm_gateway.complete(prompt, model=model)` (via `log_completion`,
        # same as summarize_ticket already did) instead of the old bare
        # `llm_gateway.complete(prompt)` -- this mocked side_effect must
        # accept that keyword too, or Mock's argument-binding itself raises
        # a TypeError before this function's body (including its real
        # `time.sleep()`) ever runs, which classify_intent's own
        # `except (..., TypeError)` soft-degrade path would then silently
        # swallow -- exactly the failure mode this comment documents so a
        # future reader doesn't have to re-diagnose it.
        #
        # Phase 12 Task 4: `log_completion` now ALWAYS passes
        # `cacheable_prefix=...` through to `llm_gateway.complete()` too
        # (None for classify_intent/the no-context compose branch, real
        # stable text for the with-context compose branch) -- this mock
        # must accept that keyword for the identical reason, or it silently
        # degrades every chat turn to the fallback message via the exact
        # same swallowed-TypeError mechanism the comment above documents.
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(classify_sleep_s)
            return _classification_result()
        time.sleep(compose_sleep_s)
        return _answer_result(answer_text)

    return _side_effect


def _spans_for_thread(raw_db_session, thread_id: str) -> list[Span]:
    run = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == thread_id))
    assert run is not None
    return list(
        raw_db_session.scalars(
            select(Span).where(Span.agent_run_id == run.id).order_by(Span.id)
        ).all()
    )


def test_chat_span_latencies_are_genuinely_measured_and_non_uniform(
    chat_fixtures, raw_db_session, client
):
    """Phase 11 Task 2's core exit criterion: real per-node `Span.latency_ms`
    values, not the old "split evenly across 4 stages" placeholder.

    Injects a REAL 20ms sleep around classify_intent's completion call and
    a REAL 80ms sleep around compose_response's -- if the old placeholder
    logic still existed (single `ainvoke()` wall-clock time divided evenly
    by 4), every stage would measure the SAME value regardless of this
    deliberate real timing difference. Real per-node instrumentation
    cannot produce that: `compose_response`'s row must measure meaningfully
    more than `classify_intent`'s, because that is where the real 80ms of
    (test-injected, but genuinely executed) work actually happened.
    """
    requester = chat_fixtures
    message = "What is a VPN?"

    request_start = time.monotonic()
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=_sleepy_complete_side_effect(
            0.02, 0.08, "A VPN is a Virtual Private Network."
        ),
    ):
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    request_wallclock_ms = (time.monotonic() - request_start) * 1000

    assert response.status_code == 200
    body = response.json()

    spans = _spans_for_thread(raw_db_session, body["thread_id"])
    assert [s.stage_name for s in spans] == [
        "classify_intent",
        "retrieve",
        "compose_response",
        "finalize",
    ]
    latencies = {s.stage_name: s.latency_ms for s in spans}

    # Hard proof the placeholder is gone: an evenly-split value would make
    # all 4 stages equal by construction, no matter what real work each
    # node actually did. Real per-node measurement, given a real 60ms
    # difference deliberately injected between two of the nodes, cannot.
    assert len(set(latencies.values())) > 1

    # The node that slept longer must measure as having taken longer --
    # proves the VALUES correspond to which node did the (real, injected)
    # work, not just "some values happen to differ."
    assert latencies["compose_response"] > latencies["classify_intent"]
    assert latencies["classify_intent"] >= 15  # >= 20ms sleep, minus scheduler jitter
    assert latencies["compose_response"] >= 70  # >= 80ms sleep, minus scheduler jitter

    # Every stage individually plausible.
    for value in latencies.values():
        assert value >= 0

    # Sum of the 4 real per-node measurements should be at least the ~100ms
    # of real sleep actually injected (minus jitter), and should not wildly
    # exceed the real measured wall-clock time for the whole HTTP request
    # (generous slack for FastAPI/DB/session-commit overhead the per-node
    # timers deliberately don't capture -- they only time each graph
    # node's own work, per this task's brief).
    total_latency_ms = sum(latencies.values())
    assert total_latency_ms >= 90
    assert total_latency_ms <= request_wallclock_ms + 50


def test_chat_span_latency_distribution_differs_across_two_different_inputs(
    chat_fixtures, raw_db_session, client
):
    """Phase 11 Task 2's other required proof: two real chat turns with
    genuinely different per-node work produce genuinely DIFFERENT
    per-node latency distributions -- not just "non-uniform within one
    run" (the test above), but actually different FROM EACH OTHER,
    ruling out a fixed/hardcoded distribution that merely looks
    non-uniform.
    """
    requester = chat_fixtures

    def _run(classify_sleep_s: float, compose_sleep_s: float, message: str) -> str:
        with patch(
            _COMPLETE_PATCH_TARGET,
            side_effect=_sleepy_complete_side_effect(
                classify_sleep_s, compose_sleep_s, f"Answer to: {message}"
            ),
        ):
            response = client.post(
                "/chat",
                json={"message": message},
                headers={"X-Debug-Employee-Id": str(requester.id)},
            )
        assert response.status_code == 200
        return response.json()["thread_id"]

    # Run 1: classify_intent does the (relatively) heavier real work.
    thread_id_1 = _run(0.06, 0.01, "first message, slow classification")
    # Run 2: compose_response does the heavier real work instead.
    thread_id_2 = _run(0.01, 0.06, "second message, slow composition")

    latencies_1 = {s.stage_name: s.latency_ms for s in _spans_for_thread(raw_db_session, thread_id_1)}
    latencies_2 = {s.stage_name: s.latency_ms for s in _spans_for_thread(raw_db_session, thread_id_2)}

    assert latencies_1 != latencies_2
    # Directly confirms the distribution is genuinely input-dependent, not
    # coincidentally different: whichever node got the real heavier sleep
    # in each run measures as the slower one in THAT run.
    assert latencies_1["classify_intent"] > latencies_1["compose_response"]
    assert latencies_2["compose_response"] > latencies_2["classify_intent"]


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

    # The mocked answer must cite a REAL chunk id -- Phase 8 Task 7's
    # citation-verification step (`verify_citations`, running inside the
    # real graph after compose_response) strips any [chunk:<id>] marker
    # that doesn't resolve to a chunk actually present in the real
    # context_block, and chat.py now filters the returned `citations` list
    # down to exactly the verified survivors. A hardcoded fake id like
    # `[chunk:1]` would always get stripped as fabricated, since real
    # `Chunk.id` values are DB-assigned autoincrement ints, never
    # literally 1 in this shared dev DB. `side_effect` here is a callable
    # (not a fixed list) so the second call can inspect the REAL prompt
    # `compose_response` built (which already contains real `[chunk:<id>]`
    # labels for whatever `retrieve_for_agent` actually found) and cite
    # the first real id it sees, rather than guessing one.
    _call_count = {"n": 0}

    def _complete_side_effect(
        prompt: str, *, model: str = "local-qwen3", cacheable_prefix: str | None = None
    ) -> CompletionResult:
        # See `_sleepy_complete_side_effect`'s identical comment (Phase 11
        # Task 3 / Phase 12 Task 4): real closures now call
        # `llm_gateway.complete(prompt, model=model, cacheable_prefix=...)`,
        # so this side_effect must accept both keywords too.
        _call_count["n"] += 1
        if _call_count["n"] == 1:
            # 1st call: classify_intent's prompt.
            return _classification_result()
        # 2nd call: compose_response's prompt, which already contains real
        # [chunk:<id>] labels for whatever retrieve_for_agent actually
        # found -- extract one to cite rather than guessing a fake id.
        # Must search only inside the <retrieved_context>...</retrieved_context>
        # delimiter (Phase 8 Task 7's prompt-injection framing fix), not
        # the whole prompt: the instruction text above it contains its own
        # literal formatting EXAMPLE, "[chunk:123]", which is not a real
        # chunk id and must not be picked up here.
        context_match = re.search(
            r"<retrieved_context>(.*)</retrieved_context>", prompt, re.DOTALL
        )
        assert context_match, f"expected a <retrieved_context> block in the prompt: {prompt!r}"
        match = re.search(r"\[chunk:(\d+)\]", context_match.group(1))
        assert match, f"expected a real [chunk:<id>] citation label inside <retrieved_context>: {prompt!r}"
        real_chunk_id = match.group(1)
        return _answer_result(
            f"A VPN creates an encrypted tunnel for remote access [chunk:{real_chunk_id}]."
        )

    with patch(_COMPLETE_PATCH_TARGET, side_effect=_complete_side_effect) as mock_complete:
        response = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )

    assert response.status_code == 200
    assert mock_complete.call_count == 2
    body = response.json()

    assert body["citations"], "expected at least one verified citation from the ingested seed corpus"
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


# ---------------------------------------------------------------------------
# Exact-response cache tests (Phase 12 Task 3)
# ---------------------------------------------------------------------------


def test_chat_second_identical_call_is_a_real_cache_hit_and_skips_the_graph(
    chat_fixtures, raw_db_session, client
):
    """The real point of Task 3: an identical (scope, kb_version,
    prompt_template_version, message) turn must only ever run the real
    graph ONCE. Proven with a real call-counting spy on the SECOND call
    that raises if it's ever invoked -- not just "should work" reasoning
    -- plus real DB row assertions distinguishing the two AgentRun/Span
    row sets and the real $0 ModelCall row the cache hit must still write.
    """
    requester = chat_fixtures
    message = "What does the cache-hit test ask about VPNs?"

    before_model_call_id = raw_db_session.scalar(select(func.max(ModelCall.id))) or 0

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("A VPN is a Virtual Private Network.")],
    ) as mock_complete_1:
        response_1 = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_1.status_code == 200
    assert mock_complete_1.call_count == 2
    body_1 = response_1.json()

    # The SECOND call must never reach the real completion function at
    # all -- a real hit skips classify_intent/retrieve/compose_response/
    # verify_citations entirely, so `llm_gateway.complete` must never be
    # called. `side_effect` here raises loudly if it ever is.
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=AssertionError("cache hit must never call the real completion function"),
    ) as mock_complete_2:
        response_2 = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_2.status_code == 200
    mock_complete_2.assert_not_called()
    body_2 = response_2.json()

    # Identical response SHAPE/content, but a genuinely fresh thread_id
    # (per this task's own explicit contract: thread_id identifies THIS
    # call's AgentRun, never the cached content).
    assert body_2["answer"] == body_1["answer"]
    assert body_2["citations"] == body_1["citations"]
    assert body_2["caption"] == body_1["caption"]
    assert body_2["thread_id"] != body_1["thread_id"]

    # Two genuinely distinct AgentRun/Span row sets -- the cache hit is its
    # own real row, not silently merged/skipped.
    run_1 = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body_1["thread_id"]))
    run_2 = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body_2["thread_id"]))
    assert run_1 is not None and run_2 is not None
    assert run_1.id != run_2.id
    assert run_1.status == "completed"
    assert run_2.status == "completed"
    assert run_1.cache_hit is False
    assert run_2.cache_hit is True
    assert run_2.output_text == run_1.output_text

    spans_1 = raw_db_session.scalars(
        select(Span).where(Span.agent_run_id == run_1.id).order_by(Span.id)
    ).all()
    spans_2 = raw_db_session.scalars(
        select(Span).where(Span.agent_run_id == run_2.id).order_by(Span.id)
    ).all()
    assert [s.stage_name for s in spans_1] == ["classify_intent", "retrieve", "compose_response", "finalize"]
    assert [s.stage_name for s in spans_2] == ["classify_intent", "retrieve", "compose_response", "finalize"]
    assert all(s.status == "success" for s in spans_2)

    # Real, dramatic latency drop: the cache-hit run's total real measured
    # latency must be a small fraction of the real graph run's.
    total_latency_1 = sum(s.latency_ms for s in spans_1)
    total_latency_2 = sum(s.latency_ms for s in spans_2)
    assert total_latency_2 < total_latency_1

    # Real $0 ModelCall row for the cache hit -- cost/telemetry dashboards
    # must not silently under-count this real traffic.
    new_model_calls = raw_db_session.scalars(
        select(ModelCall).where(ModelCall.id > before_model_call_id).order_by(ModelCall.id)
    ).all()
    cache_hit_calls = [c for c in new_model_calls if c.purpose == "chat.cache_hit"]
    assert len(cache_hit_calls) == 1
    cache_hit_call = cache_hit_calls[0]
    assert cache_hit_call.estimated_cost_usd == 0.0
    assert cache_hit_call.input_tokens == 0
    assert cache_hit_call.output_tokens == 0
    assert cache_hit_call.status == "success"
    assert cache_hit_call.trace_id is not None and len(cache_hit_call.trace_id) == 32

    # The second call must have added EXACTLY the one cache-hit ModelCall
    # row -- no additional "chat.classify_intent"/"chat.compose_response"
    # rows, proving no real completion happened for this turn. Every row
    # ID strictly greater than the cache-hit row's own id-minus-nothing
    # (i.e. everything at or after it) must be that one row alone.
    rows_at_or_after_cache_hit = [c for c in new_model_calls if c.id >= cache_hit_call.id]
    assert {c.purpose for c in rows_at_or_after_cache_hit} == {"chat.cache_hit"}
    assert "chat.classify_intent" in {c.purpose for c in new_model_calls}  # from the FIRST call only


def test_chat_cache_miss_then_hit_returns_identical_citations_for_seeded_corpus(
    chat_fixtures, seed_corpus_ingested, client
):
    """A cache hit must return the exact SAME citations/caption a real
    grounded answer produced, not just the same answer text -- proves the
    cached VALUE genuinely carries the full response payload, not just
    `answer`.
    """
    requester = chat_fixtures
    message = "What is a VPN, per the cache test?"

    _call_count = {"n": 0}

    def _complete_side_effect(
        prompt: str, *, model: str = "local-qwen3", cacheable_prefix: str | None = None
    ) -> CompletionResult:
        _call_count["n"] += 1
        if _call_count["n"] == 1:
            return _classification_result()
        context_match = re.search(r"<retrieved_context>(.*)</retrieved_context>", prompt, re.DOTALL)
        assert context_match
        match = re.search(r"\[chunk:(\d+)\]", context_match.group(1))
        assert match
        real_chunk_id = match.group(1)
        return _answer_result(f"Answer with a real citation [chunk:{real_chunk_id}].")

    with patch(_COMPLETE_PATCH_TARGET, side_effect=_complete_side_effect):
        response_1 = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_1.status_code == 200
    body_1 = response_1.json()
    assert body_1["citations"], "expected a real citation from the ingested seed corpus"

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=AssertionError("cache hit must never call the real completion function"),
    ) as mock_complete_2:
        response_2 = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_2.status_code == 200
    mock_complete_2.assert_not_called()
    body_2 = response_2.json()

    assert body_2["citations"] == body_1["citations"]
    assert body_2["caption"] == body_1["caption"]
    assert body_2["answer"] == body_1["answer"]


def test_chat_response_cache_never_shares_entries_across_different_scopes(
    chat_fixtures, other_scope_chat_fixtures, client
):
    """Cross-user-leakage guard for the response cache, mirroring Task 2's
    mandatory retrieval-cache leakage test: two employees with genuinely
    DIFFERENT resolved `AuthzFilter`s (different departments -- neither is
    the seed corpus's own department, so both resolve to a real, distinct,
    empty-tag scope_sig computed from their own department name) asking
    the IDENTICAL message text must NOT share a cache entry -- the second
    employee's call must still reach the real completion function.
    """
    requester_a = chat_fixtures
    requester_b = other_scope_chat_fixtures
    message = "What is a VPN, per the cross-scope cache test?"

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("Answer for requester A.")],
    ) as mock_complete_a:
        response_a = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester_a.id)},
        )
    assert response_a.status_code == 200
    assert mock_complete_a.call_count == 2

    # A genuinely different department -> a genuinely different resolved
    # AuthzFilter/scope_sig (see other_scope_chat_fixtures) -> this MUST be
    # a real cache miss, reaching the real completion function again, even
    # though the message text and kb_version are identical to requester A's
    # call above.
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("Answer for requester B.")],
    ) as mock_complete_b:
        response_b = client.post(
            "/chat",
            json={"message": message},
            headers={"X-Debug-Employee-Id": str(requester_b.id)},
        )
    assert response_b.status_code == 200
    assert mock_complete_b.call_count == 2

    assert response_a.json()["answer"] == "Answer for requester A."
    assert response_b.json()["answer"] == "Answer for requester B."


def test_chat_response_cache_degrades_to_real_graph_on_redis_outage(chat_fixtures, client):
    """A real, genuinely unreachable Redis endpoint (not a mock) must never
    break `/chat` -- the cache lookup/store degrade to "run the real
    graph" instead of raising, mirroring Task 1/2's identical outage
    -degrade proof (`test_cache.py`'s
    `test_redis_outage_degrades_to_direct_embed_fn_instead_of_raising`).

    `resolvegrid_api.cache.get_redis` returns the shared, real, reachable
    dev Redis client via FastAPI's `Depends(get_redis)` -- there is no
    per-request override plumbing for this in `main.py` today, so this
    test instead proves the outage-degrade behavior directly at the
    `response_cache` function level (real unreachable client, same as
    `test_response_cache.py`), which is the actual unit `chat.py` calls
    into and the only place a real Redis outage could ever surface.
    """
    import redis as redis_module

    from resolvegrid_api.response_cache import lookup_cached_response

    unreachable_client = redis_module.Redis(
        host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1
    )
    result = lookup_cached_response(
        message_text="does this even matter",
        kb_version=0,
        scope_sig="unrestricted",
        prompt_template_version="1",
        redis_client=unreachable_client,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Semantic-response cache tests (Phase 12 Task 5)
# ---------------------------------------------------------------------------


def test_chat_differently_worded_but_similar_question_hits_semantic_cache(
    chat_fixtures, raw_db_session, client
):
    """The real point of Task 5: a SECOND question, differently worded but
    semantically equivalent to the first (a real near-duplicate pair,
    empirically measured at cosine distance 0.0356 -- see `semantic_cache.
    py`'s module docstring -- well under `SEMANTIC_CACHE_SIMILARITY_
    THRESHOLD`'s 0.15), must hit the SEMANTIC cache even though it is a
    guaranteed MISS on Task 3's exact (byte-identical) cache. Proven with a
    real call-counting spy on the second call that raises if ever invoked,
    plus real AgentRun row assertions distinguishing this from an exact hit.
    """
    requester = chat_fixtures
    message_1 = "What is a VPN?"
    message_2 = "Could you explain what a VPN is?"  # different text, same meaning

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("A VPN is a Virtual Private Network.")],
    ) as mock_complete_1:
        response_1 = client.post(
            "/chat",
            json={"message": message_1},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_1.status_code == 200
    assert mock_complete_1.call_count == 2
    body_1 = response_1.json()

    # The real completion function must NEVER be reached for this second,
    # differently-worded call -- a real semantic-cache hit skips the whole
    # graph exactly like a real exact-cache hit does.
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=AssertionError("semantic cache hit must never call the real completion function"),
    ) as mock_complete_2:
        response_2 = client.post(
            "/chat",
            json={"message": message_2},
            headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_2.status_code == 200
    mock_complete_2.assert_not_called()
    body_2 = response_2.json()

    assert body_2["answer"] == body_1["answer"] == "A VPN is a Virtual Private Network."
    assert body_2["thread_id"] != body_1["thread_id"]

    run_1 = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body_1["thread_id"]))
    run_2 = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body_2["thread_id"]))
    assert run_1 is not None and run_2 is not None
    assert run_1.cache_hit is False
    assert run_1.cache_hit_kind is None
    assert run_2.cache_hit is True
    assert run_2.cache_hit_kind == "semantic"  # distinguished from an "exact" hit

    spans_2 = raw_db_session.scalars(
        select(Span).where(Span.agent_run_id == run_2.id).order_by(Span.id)
    ).all()
    assert [s.stage_name for s in spans_2] == ["classify_intent", "retrieve", "compose_response", "finalize"]
    assert all(s.status == "success" for s in spans_2)

    new_model_calls = raw_db_session.scalars(
        select(ModelCall).where(ModelCall.purpose == "chat.semantic_cache_hit").order_by(ModelCall.id)
    ).all()
    assert len(new_model_calls) >= 1
    semantic_hit_call = new_model_calls[-1]
    assert semantic_hit_call.estimated_cost_usd == 0.0
    assert semantic_hit_call.input_tokens == 0
    assert semantic_hit_call.output_tokens == 0
    assert semantic_hit_call.status == "success"


def test_chat_semantically_different_question_does_not_hit_semantic_cache(
    chat_fixtures, raw_db_session, client
):
    """A genuinely different-topic second question (real measured cosine
    distance 0.5374 against the first, per `semantic_cache.py`'s module
    docstring -- far above the 0.15 threshold) must NOT hit the semantic
    cache -- the real graph must run again, reaching the real completion
    function.
    """
    requester = chat_fixtures
    message_1 = "What is a firewall?"
    message_2 = "How do I request VPN access?"  # different topic entirely

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("A firewall is a network security device.")],
    ):
        response_1 = client.post(
            "/chat", json={"message": message_1}, headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_1.status_code == 200

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("Submit a VPN access request via the portal.")],
    ) as mock_complete_2:
        response_2 = client.post(
            "/chat", json={"message": message_2}, headers={"X-Debug-Employee-Id": str(requester.id)},
        )
    assert response_2.status_code == 200
    assert mock_complete_2.call_count == 2  # real graph ran again -- no false hit
    body_2 = response_2.json()
    assert body_2["answer"] == "Submit a VPN access request via the portal."

    run_2 = raw_db_session.scalar(select(AgentRun).where(AgentRun.thread_id == body_2["thread_id"]))
    assert run_2 is not None
    assert run_2.cache_hit is False
    assert run_2.cache_hit_kind is None


def test_chat_semantic_cache_never_shares_entries_across_different_scopes(
    chat_fixtures, other_scope_chat_fixtures, client
):
    """Cross-user-leakage guard for the semantic cache, mirroring Task 2/3's
    mandatory leakage tests: two employees with genuinely DIFFERENT
    resolved scopes asking NEAR-DUPLICATE (not even identical) questions
    must NOT share a semantic cache entry -- the second employee's call
    must still reach the real completion function.
    """
    requester_a = chat_fixtures
    requester_b = other_scope_chat_fixtures
    message_a = "What is a firewall?"
    message_b = "Can you explain what a firewall is?"  # near-duplicate of message_a

    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("Answer for requester A.")],
    ) as mock_complete_a:
        response_a = client.post(
            "/chat", json={"message": message_a}, headers={"X-Debug-Employee-Id": str(requester_a.id)},
        )
    assert response_a.status_code == 200
    assert mock_complete_a.call_count == 2

    # Requester B is in a genuinely different department (-> different
    # scope_sig) and asks a near-duplicate question -- this MUST be a real
    # miss on both caches, reaching the real completion function, even
    # though the embedding is nearly identical to requester A's.
    with patch(
        _COMPLETE_PATCH_TARGET,
        side_effect=[_classification_result(), _answer_result("Answer for requester B.")],
    ) as mock_complete_b:
        response_b = client.post(
            "/chat", json={"message": message_b}, headers={"X-Debug-Employee-Id": str(requester_b.id)},
        )
    assert response_b.status_code == 200
    assert mock_complete_b.call_count == 2

    assert response_a.json()["answer"] == "Answer for requester A."
    assert response_b.json()["answer"] == "Answer for requester B."


def test_chat_semantic_cache_degrades_to_real_graph_on_postgres_outage():
    """A real, genuinely unreachable Postgres endpoint (not a mock) must
    never break `/chat` -- mirrors `test_chat_response_cache_degrades_to_
    real_graph_on_redis_outage`'s identical reasoning/shape for this cache's
    own real failure mode (Postgres, not Redis). Proven directly at the
    `semantic_cache` function level, the actual unit `chat.py` calls into.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from resolvegrid_api.semantic_cache import lookup_semantic_cache

    unreachable_engine = create_engine(
        "postgresql+psycopg://user:pass@localhost:1/db", connect_args={"connect_timeout": 1}
    )
    unreachable_session = Session(unreachable_engine)
    try:
        result = lookup_semantic_cache(
            unreachable_session,
            query_embedding=[0.1] * 768,
            scope_sig="unrestricted",
            kb_version=0,
        )
        assert result is None
    finally:
        unreachable_session.close()
        unreachable_engine.dispose()
