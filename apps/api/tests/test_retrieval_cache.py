"""Tests for `resolvegrid_api.retrieval_cache` (Phase 12 Task 2: retrieval
-result cache + mandatory cross-user leakage test).

Real Redis and real Postgres throughout, matching this codebase's
established philosophy (`test_cache.py`'s module docstring, Task 1) --
nothing here mocks Redis or the DB away. `scope_sig` values used in the
mandatory security tests below are derived from REAL, resolved
`AuthzFilter`s (via `build_authz_filter` against real seeded `Employee`/
`Department` rows), not hand-typed strings, so a bug in `compute_scope_sig`
or in `build_authz_filter` itself would still be caught here, exactly the
same rigor bar `test_retrieval.py`'s
`test_adversarial_authz_filter_blocks_cross_scope_leakage` established for
the SQL-authz layer.
"""

import hashlib

import pytest
import redis
from sqlalchemy import delete, func, select

from resolvegrid_api.cache import REDIS_URL
from resolvegrid_api.models.knowledge import (
    Chunk,
    Document,
    DocumentVersion,
    Embedding,
    IngestionRun,
)
from resolvegrid_api.models.org import Department, Employee
from resolvegrid_api.retrieval_authz import AuthzFilter, build_authz_filter
from resolvegrid_api.retrieval_cache import (
    RETRIEVAL_CACHE_TTL_SECONDS,
    compute_scope_sig,
    get_kb_version,
    get_or_compute_retrieval,
)
from resolvegrid_authz import Principal

# Task 2's retrieval cache TTL must be strictly shorter than Task 1's
# embedding cache TTL, per the plan doc's explicit "hours, not days" bound
# on this cache specifically -- imported here (rather than hardcoded) so
# this assertion tracks either constant if it's ever tuned.
from resolvegrid_api.cache import EMBEDDING_CACHE_TTL_SECONDS


def _clear_retrieval_cache_namespace(client: redis.Redis) -> None:
    """Delete every `cache:retrieval:*` key.

    Cleans up BEFORE and AFTER each test, not just after: several tests
    below use fixed, literal `kb_version`/`query_text` values (deliberately
    -- so a human reader can see exactly which key a given assertion is
    about) precisely reproduce the same cache key across repeated test
    runs. Without a real TTL having elapsed (`RETRIEVAL_CACHE_TTL_SECONDS`
    is 2 hours), a key a PRIOR run wrote would otherwise still be a live,
    real cache hit the next run finds -- silently turning a "does this
    genuinely recompute" assertion into a false pass (or, worse, a false
    fail if a prior run's broken-code experiment left a wrong value
    cached). A broad `cache:retrieval:*` sweep is safe here: this is a
    dedicated dev Redis instance for this test suite's own cache
    namespace, and deleting a cache entry never loses real data -- the
    next real caller simply recomputes it, exactly the outage-degrade path
    this cache already has to support.
    """
    for key in client.scan_iter(match="cache:retrieval:*"):
        client.delete(key)


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fail fast (real connection check) if Redis isn't up
    _clear_retrieval_cache_namespace(client)
    yield client
    _clear_retrieval_cache_namespace(client)


# ---------------------------------------------------------------------------
# Cache-plumbing tests (key shape, hit/miss, TTL, isolation-by-key,
# real-outage degrade) -- mirrors test_cache.py's structure for Task 1.
# ---------------------------------------------------------------------------


def test_cache_key_shape(redis_client):
    query_text = "what is a firewall?"
    kb_version = 999_001  # namespaced "test*" prefix via scope_sig below
    scope_sig = "test-keyshape:unrestricted"

    get_or_compute_retrieval(
        query_text=query_text,
        kb_version=kb_version,
        scope_sig=scope_sig,
        redis_client=redis_client,
        compute_fn=lambda: {"chunks": [], "sufficient": True, "context_block": "x"},
    )

    digest = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    expected_key = f"cache:retrieval:{kb_version}:{scope_sig}:{digest}"
    assert redis_client.get(expected_key) is not None


def test_second_call_same_kb_version_scope_and_query_is_a_hit_and_does_not_recompute(redis_client):
    calls = []

    def compute():
        calls.append(1)
        return {"chunks": [{"chunk_id": 1}], "sufficient": True, "context_block": "real result"}

    kwargs = dict(
        query_text="what is a VPN?",
        kb_version=999_002,
        scope_sig="test-hit:unrestricted",
        redis_client=redis_client,
    )
    first = get_or_compute_retrieval(**kwargs, compute_fn=compute)
    second = get_or_compute_retrieval(**kwargs, compute_fn=compute)

    assert first == second == {"chunks": [{"chunk_id": 1}], "sufficient": True, "context_block": "real result"}
    assert len(calls) == 1  # only ONE real pipeline run across both calls


def test_ttl_is_set_and_shorter_than_embedding_cache_ttl(redis_client):
    query_text = "some stable ttl-checking retrieval query"
    kb_version = 999_003
    scope_sig = "test-ttl:unrestricted"

    get_or_compute_retrieval(
        query_text=query_text,
        kb_version=kb_version,
        scope_sig=scope_sig,
        redis_client=redis_client,
        compute_fn=lambda: {"chunks": [], "sufficient": False, "context_block": ""},
    )

    digest = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    key = f"cache:retrieval:{kb_version}:{scope_sig}:{digest}"
    ttl = redis_client.ttl(key)
    assert 0 < ttl <= RETRIEVAL_CACHE_TTL_SECONDS
    assert RETRIEVAL_CACHE_TTL_SECONDS < EMBEDDING_CACHE_TTL_SECONDS


def test_different_scope_sig_gets_an_independent_cache_entry(redis_client):
    """Same kb_version, same query text, different scope_sig -- must be
    treated as two entirely separate cache entries (this is the actual
    mechanism the mandatory security test below relies on).
    """
    calls_a, calls_b = [], []
    query_text = "identical query text, different scopes"
    kb_version = 999_004

    result_a = get_or_compute_retrieval(
        query_text=query_text,
        kb_version=kb_version,
        scope_sig="test-scopeA:tags:engineering",
        redis_client=redis_client,
        compute_fn=lambda: (calls_a.append(1), {"chunks": ["A"], "sufficient": True, "context_block": "A"})[1],
    )
    result_b = get_or_compute_retrieval(
        query_text=query_text,
        kb_version=kb_version,
        scope_sig="test-scopeB:tags:hr",
        redis_client=redis_client,
        compute_fn=lambda: (calls_b.append(1), {"chunks": ["B"], "sufficient": True, "context_block": "B"})[1],
    )

    assert result_a != result_b
    assert len(calls_a) == 1
    assert len(calls_b) == 1


def test_different_kb_version_gets_an_independent_cache_entry(redis_client):
    """A kb_version bump (real corpus re-ingestion) must not serve a stale
    entry from the prior version -- proven directly here without needing a
    real ingestion run.
    """
    calls = []
    query_text = "identical query, different kb_version"
    scope_sig = "test-kbversion:unrestricted"

    def compute():
        calls.append(1)
        return {"chunks": [f"result-{len(calls)}"], "sufficient": True, "context_block": "x"}

    result_v1 = get_or_compute_retrieval(
        query_text=query_text, kb_version=999_005, scope_sig=scope_sig,
        redis_client=redis_client, compute_fn=compute,
    )
    result_v2 = get_or_compute_retrieval(
        query_text=query_text, kb_version=999_006, scope_sig=scope_sig,
        redis_client=redis_client, compute_fn=compute,
    )

    assert result_v1 != result_v2
    assert len(calls) == 2  # both versions actually recomputed, no cross-version hit


def test_redis_outage_degrades_to_direct_compute_fn_instead_of_raising():
    """A real, genuinely unreachable Redis endpoint (port 1, refuses
    connections immediately) -- not a mock -- proves this cache degrades
    to the real computation rather than propagating and breaking `/chat`.
    """
    unreachable_client = redis.Redis(host="localhost", port=1, socket_connect_timeout=1, socket_timeout=1)
    calls = []

    def compute():
        calls.append(1)
        return {"chunks": ["real"], "sufficient": True, "context_block": "real"}

    result = get_or_compute_retrieval(
        query_text="outage test query",
        kb_version=1,
        scope_sig="unrestricted",
        redis_client=unreachable_client,
        compute_fn=compute,
    )

    assert result == {"chunks": ["real"], "sufficient": True, "context_block": "real"}
    assert len(calls) == 1


def test_redis_write_back_failure_still_returns_computed_result():
    real_client = redis.Redis.from_url(REDIS_URL)
    real_client.ping()

    class _BrokenPipelineClient:
        def get(self, key):
            return real_client.get(key)

        def set(self, *args, **kwargs):
            raise redis.exceptions.ConnectionError("simulated write-back failure")

    calls = []

    def compute():
        calls.append(1)
        return {"chunks": ["computed"], "sufficient": True, "context_block": "computed"}

    result = get_or_compute_retrieval(
        query_text="write-back-failure test query, never cached before",
        kb_version=1,
        scope_sig="unrestricted",
        redis_client=_BrokenPipelineClient(),
        compute_fn=compute,
    )

    assert result == {"chunks": ["computed"], "sufficient": True, "context_block": "computed"}
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# compute_scope_sig / get_kb_version -- the two real inputs to the cache key
# ---------------------------------------------------------------------------


def test_compute_scope_sig_unrestricted():
    assert compute_scope_sig(AuthzFilter(unrestricted=True)) == "unrestricted"


def test_compute_scope_sig_sorts_tags_deterministically():
    """Two AuthzFilters built from the same logical tag set, differing only
    in the frozenset's internal iteration order, must produce byte
    -identical signatures -- otherwise two same-scoped callers could
    accidentally miss each other's cache entry purely by hash-ordering
    luck, defeating the sharing the master plan asks for.
    """
    filter_a = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"zeta", "alpha", "mu"}))
    filter_b = AuthzFilter(unrestricted=False, allowed_tags=frozenset({"mu", "zeta", "alpha"}))

    assert compute_scope_sig(filter_a) == compute_scope_sig(filter_b) == "tags:alpha,mu,zeta"


def test_get_kb_version_reflects_real_max_completed_ingestion_run(raw_db_session):
    before_watermark = raw_db_session.scalar(select(func.max(IngestionRun.id))) or 0

    running_run = IngestionRun(
        parser_version="test-v1", chunking_version="test-v1", embedding_version="test-v1", status="running",
    )
    raw_db_session.add(running_run)
    raw_db_session.flush()
    completed_run = IngestionRun(
        parser_version="test-v1", chunking_version="test-v1", embedding_version="test-v1", status="completed",
    )
    raw_db_session.add(completed_run)
    raw_db_session.commit()

    try:
        version = get_kb_version(raw_db_session)
        # The just-inserted completed run is the newest completed row, so
        # it must be the reported kb_version (a real query result, not an
        # assumed literal) -- the running-status row must NOT win even
        # though its id is also > before_watermark.
        assert version == completed_run.id
        assert version > before_watermark
        assert version != running_run.id
    finally:
        raw_db_session.execute(delete(IngestionRun).where(IngestionRun.id > before_watermark))
        raw_db_session.commit()


# ---------------------------------------------------------------------------
# Mandatory security test (Phase 12 Task 2's most important test): real
# cross-user cache-leakage proof + the positive same-scope-sharing proof.
# Both `scope_sig`s below are derived from REAL, resolved `AuthzFilter`s via
# `build_authz_filter` against two real, disjointly-departmented `Employee`
# rows -- not hand-typed strings -- so a bug anywhere in the scope
# -resolution chain (not just inside `compute_scope_sig` itself) would
# still be caught here.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_disjoint_departments_and_employees(raw_db_session):
    """Two real employees, each in a real, disjoint department -- mirrors
    `test_retrieval.py`'s adversarial fixture's use of real
    engineering/hr-scoped data, adapted to the cache layer.
    """
    eng_dept = raw_db_session.scalar(select(Department).where(Department.name == "Cache Test Engineering"))
    if eng_dept is None:
        eng_dept = Department(name="Cache Test Engineering")
        raw_db_session.add(eng_dept)
        raw_db_session.flush()
    hr_dept = raw_db_session.scalar(select(Department).where(Department.name == "Cache Test HR"))
    if hr_dept is None:
        hr_dept = Department(name="Cache Test HR")
        raw_db_session.add(hr_dept)
        raw_db_session.flush()

    employee_a = Employee(
        display_name="Cache Test Employee A (Engineering)",
        email="cache.test.employee.a@example.test",
        title="Engineer",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
        department_id=eng_dept.id,
    )
    employee_b = Employee(
        display_name="Cache Test Employee B (HR)",
        email="cache.test.employee.b@example.test",
        title="HR Specialist",
        hire_date="2024-01-01T00:00:00",
        timezone="America/Chicago",
        department_id=hr_dept.id,
    )
    raw_db_session.add_all([employee_a, employee_b])
    raw_db_session.commit()

    try:
        yield employee_a, employee_b
    finally:
        raw_db_session.execute(delete(Employee).where(Employee.id.in_([employee_a.id, employee_b.id])))
        raw_db_session.execute(delete(Department).where(Department.id.in_([eng_dept.id, hr_dept.id])))
        raw_db_session.commit()


def test_cross_user_leakage_disjoint_scopes_never_share_a_cache_entry(
    raw_db_session, redis_client, two_disjoint_departments_and_employees
):
    employee_a, employee_b = two_disjoint_departments_and_employees

    filter_a = build_authz_filter(Principal(employee_id=employee_a.id, roles=()), raw_db_session)
    filter_b = build_authz_filter(Principal(employee_id=employee_b.id, roles=()), raw_db_session)
    scope_sig_a = compute_scope_sig(filter_a)
    scope_sig_b = compute_scope_sig(filter_b)
    # Sanity: the two real resolved scopes really are different -- a bug
    # that made them collide would make the rest of this test meaningless.
    assert scope_sig_a != scope_sig_b

    kb_version = get_kb_version(raw_db_session)
    identical_query_text = "quarterly headcount budget planning figures"

    calls_a, calls_b = [], []

    def compute_engineering_secret():
        calls_a.append(1)
        return {
            "chunks": [{"chunk_id": 1, "document_title": "Engineering Secret", "text": "ENG-SECRET-CONTENT"}],
            "sufficient": True,
            "context_block": "ENG-SECRET-CONTENT",
        }

    def compute_hr_secret():
        calls_b.append(1)
        return {
            "chunks": [{"chunk_id": 2, "document_title": "HR Secret", "text": "HR-SECRET-CONTENT"}],
            "sufficient": True,
            "context_block": "HR-SECRET-CONTENT",
        }

    # Employee A (engineering scope) issues the query first -- this
    # actually populates the cache under scope_sig_a, not a no-op.
    result_a = get_or_compute_retrieval(
        query_text=identical_query_text, kb_version=kb_version, scope_sig=scope_sig_a,
        redis_client=redis_client, compute_fn=compute_engineering_secret,
    )
    # Employee B (hr scope), the IDENTICAL query text -- must never receive
    # employee A's cached engineering-scoped result.
    result_b = get_or_compute_retrieval(
        query_text=identical_query_text, kb_version=kb_version, scope_sig=scope_sig_b,
        redis_client=redis_client, compute_fn=compute_hr_secret,
    )

    # Both real computations actually ran (real cache population happened
    # for both scopes -- a bug that skipped scope_sig entirely and always
    # hit/missed the same way would still be caught below, but this proves
    # the test itself isn't vacuous: neither call was skipped).
    assert len(calls_a) == 1
    assert len(calls_b) == 1

    assert result_a["context_block"] == "ENG-SECRET-CONTENT"
    assert result_b["context_block"] == "HR-SECRET-CONTENT"
    assert "ENG-SECRET-CONTENT" not in result_b["context_block"]
    assert "HR-SECRET-CONTENT" not in result_a["context_block"]
    assert result_a != result_b

    # Symmetric re-check: employee A querying again must get ITS OWN cached
    # entry back (a real hit -- compute_engineering_secret must not run
    # again) and must still never see employee B's HR content, even though
    # employee B's call happened in between.
    result_a_again = get_or_compute_retrieval(
        query_text=identical_query_text, kb_version=kb_version, scope_sig=scope_sig_a,
        redis_client=redis_client, compute_fn=compute_engineering_secret,
    )
    assert len(calls_a) == 1  # still 1 -- served from cache, not recomputed
    assert result_a_again["context_block"] == "ENG-SECRET-CONTENT"


def test_same_resolved_scope_two_employees_share_a_real_cache_hit(raw_db_session, redis_client):
    """The positive case the master plan explicitly calls for
    ("role-scoped, department-scoped sharing"): two DIFFERENT employees
    with the IDENTICAL resolved scope must hit the SAME cache entry, so the
    expensive retrieval pipeline only actually runs once for both of them
    combined -- proven with a real call-counting spy, not "should share"
    reasoning.
    """
    dept = raw_db_session.scalar(select(Department).where(Department.name == "Cache Test Shared Dept"))
    if dept is None:
        dept = Department(name="Cache Test Shared Dept")
        raw_db_session.add(dept)
        raw_db_session.flush()

    employee_c = Employee(
        display_name="Cache Test Employee C (Shared Dept)", email="cache.test.employee.c@example.test",
        title="Analyst", hire_date="2024-01-01T00:00:00", timezone="America/Chicago", department_id=dept.id,
    )
    employee_d = Employee(
        display_name="Cache Test Employee D (Shared Dept)", email="cache.test.employee.d@example.test",
        title="Analyst", hire_date="2024-01-01T00:00:00", timezone="America/Chicago", department_id=dept.id,
    )
    raw_db_session.add_all([employee_c, employee_d])
    raw_db_session.commit()

    try:
        filter_c = build_authz_filter(Principal(employee_id=employee_c.id, roles=()), raw_db_session)
        filter_d = build_authz_filter(Principal(employee_id=employee_d.id, roles=()), raw_db_session)
        scope_sig_c = compute_scope_sig(filter_c)
        scope_sig_d = compute_scope_sig(filter_d)
        # Sanity: two different employees, same department, must really
        # resolve to the identical scope signature -- otherwise the rest
        # of this test would be proving nothing.
        assert scope_sig_c == scope_sig_d

        kb_version = get_kb_version(raw_db_session)
        identical_query_text = "shared department onboarding checklist question"
        calls = []

        def compute_shared_department_result():
            calls.append(1)
            return {
                "chunks": [{"chunk_id": 3, "document_title": "Shared Dept Doc", "text": "SHARED-CONTENT"}],
                "sufficient": True,
                "context_block": "SHARED-CONTENT",
            }

        result_from_c = get_or_compute_retrieval(
            query_text=identical_query_text, kb_version=kb_version, scope_sig=scope_sig_c,
            redis_client=redis_client, compute_fn=compute_shared_department_result,
        )
        result_from_d = get_or_compute_retrieval(
            query_text=identical_query_text, kb_version=kb_version, scope_sig=scope_sig_d,
            redis_client=redis_client, compute_fn=compute_shared_department_result,
        )

        assert result_from_c == result_from_d == {
            "chunks": [{"chunk_id": 3, "document_title": "Shared Dept Doc", "text": "SHARED-CONTENT"}],
            "sufficient": True,
            "context_block": "SHARED-CONTENT",
        }
        # The real, expensive pipeline (represented by this spy) only ran
        # ONCE for both employees combined -- employee D's call was a real
        # cache hit off employee C's earlier real computation.
        assert len(calls) == 1
    finally:
        raw_db_session.execute(delete(Employee).where(Employee.id.in_([employee_c.id, employee_d.id])))
        raw_db_session.execute(delete(Department).where(Department.id == dept.id))
        raw_db_session.commit()
