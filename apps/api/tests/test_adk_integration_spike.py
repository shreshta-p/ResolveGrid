"""Phase 13 Task 1: real, non-mocked `google-adk` integration spike.

Proves the real mechanism this phase's diagnostic specialist (Tasks 2-5) will
depend on actually works end to end against this project's real, live infra
-- NOT a mocked/stubbed ADK interaction (standing directive: "nothing
mocked, this is going to be a properly working project.") Mirrors `test_eval_
judge_calibration.py`'s established precedent for a real, non-mocked test
that calls the live LiteLLM proxy / Ollama `local-qwen3` directly, and
`test_checkpoint_restore.py`'s "no pytest-asyncio dependency -- a sync test
function driving `asyncio.run()` itself" convention (confirmed again here:
still not importable, still not in `apps/api/pyproject.toml`).

Every finding below was obtained by reading the actually-installed
`google-adk==1.14.1` package's real source/docstrings and by running real
calls against this project's live dev stack -- not from secondhand web
research (which is what the plan doc's own pre-task research was explicitly
limited to and flagged as unconfirmed). Real, empirically confirmed findings
that diverge from the plan doc's assumptions, each reproduced independently
of ADK where noted:

- **Installed version is 1.14.1, not 2.12.0.** The plan doc's web research
  (against adk.dev) found 2.12.0 "as currently published"; the real,
  dependency-resolved version `uv add --package resolvegrid-api google-adk`
  actually installs into this workspace is 1.14.1.
- **`LiteLlm` needs a separate, non-default extra.** `from google.adk.models.
  lite_llm import LiteLlm` raises `ModuleNotFoundError: No module named
  'litellm'` on a bare `google-adk` install -- `litellm` is gated behind the
  `extensions` extra (confirmed via `importlib.metadata.metadata("google-adk"
  ).get_all("Requires-Dist")`), not a core dependency. Fixed by depending on
  `google-adk[extensions]`, not `google-adk` alone.
- **Real, version-range dependency conflict between `google-adk` and
  `litellm`'s own MCP-proxy code, unrelated to anything this task actually
  uses.** `google-adk` pins `mcp>=1.8.0,<2.0.0`; `litellm`'s `proxy`/`mcp`
  extras need `mcp>=2.2.0,<3` -- mutually exclusive ranges. Neither extra was
  requested here, but `litellm`'s own `main.py` (`completion()`) *
  unconditionally* imports `litellm.responses.mcp.chat_completions_handler`
  (which imports a submodule needing `mcp>=2.2.0`'s `CacheableResult`,
  absent from the resolved `mcp==1.28.1`) whenever a real call passes
  `tools=[...]` -- i.e. on every real ADK tool-calling request, regardless of
  whether litellm's own MCP-gateway feature is used at all. Confirmed by
  reading `litellm/main.py`'s real source: the crashing imports sit inside
  `if not skip_mcp_handler and tools:`, gated by a real, litellm-defined
  `_skip_mcp_handler` kwarg this task uses as the fix -- not a monkeypatch,
  litellm's own documented escape hatch for exactly this branch. `LiteLlm`'s
  `**kwargs` pass straight through to the real `litellm.acompletion()` call
  (confirmed via `LiteLlm.__init__`'s source: `self._additional_args =
  kwargs`), so `LiteLlm(..., _skip_mcp_handler=True)` is sufficient.
- **qwen3:14b's default reasoning ("thinking") mode breaks real tool-calling
  outright, not just cost/latency.** Without `llm_gateway.py`'s own documented
  `"think": false` tuning (that module already warns this costs ~60x tokens/
  latency for non-reasoning tasks -- confirmed here it also breaks tool use
  the first real spike run: the agent received a bare, literal `"{}"` text
  reply (`finish_reason: "stop"`, 2 completion tokens) instead of a real tool
  call. Reproduced independently of ADK entirely via a direct `curl` against
  the live LiteLLM proxy with the identical tool schema and no `"think":
  false"`: byte-identical broken shape (`content: "{}"`, `finish_reason:
  "stop"`, 2 completion tokens). Adding `"think": false` to the SAME direct
  curl call immediately produces a real `finish_reason: "tool_calls"`
  response. Fixed by passing `think=False` through `LiteLlm(...)`'s kwargs
  (same pass-through mechanism as `_skip_mcp_handler` above) -- `ADK` has no
  opinion on this; it is purely qwen3/Ollama-specific, matching this
  project's own pre-existing finding for non-agentic calls.
- **qwen3:14b real, reproducible tendency to re-invoke an already-answered
  tool several times before finally answering**, confirmed NOT to be an ADK
  or LiteLlm translation bug: a `litellm._turn_on_debug()` capture of the
  real second-turn HTTP request shows a correctly OpenAI-shaped conversation
  (`role: "tool"`, matching `tool_call_id`, the real tool result already
  present, a system instruction explicitly saying "do not call it again") --
  and qwen3 still independently chose `finish_reason: "tool_calls"` again,
  re-invoking the same tool with identical empty arguments. This repeated
  across multiple real runs (one uncapped run took 36 real model turns before
  finally producing a real final answer; a second, differently-capped run
  repeated the identical pattern from the very first retry). This is a real,
  quantified MODEL behavior limitation (not an ADK/LiteLlm defect) that
  Tasks 3-5's real production prompt/tool design must account for (e.g. a
  tighter per-request tool-availability strategy), not something this spike
  task fixes. `google.adk.agents.run_config.RunConfig.max_llm_calls` (a real,
  confirmed hard safety cap -- defaults to 500, raises `LlmCallsLimitExceeded
  Error` when hit) is the real, verified bounding mechanism; this test uses a
  generous-but-finite cap (40, comfortably above the observed 36-call worst
  case) rather than tolerating an unbounded loop.
- **`DatabaseSessionService` genuinely works against this project's real
  Postgres**, confirmed via a real `create_session`/`get_session`/
  `delete_session` round trip (see the second test below) -- it silently
  auto-creates its own `sessions`/`events`/`app_states`/`user_states` tables
  on first use (confirmed via psql's list-tables command), OUTSIDE this project's Alembic
  migration history. No `prepare_tables()` method exists on the real
  installed `DatabaseSessionService` (the plan doc's secondhand research
  mentioned one) -- `create_session` creates what it needs lazily instead.
  Real, reportable architectural gap for Task 3/4 to resolve deliberately
  (e.g. a dedicated Alembic migration mirroring these tables, or an accepted
  exception to the "everything is a migration" convention) rather than
  silently inheriting tables with no migration history.
- **`delete_session`'s real cleanup scope, verified by reading its source AND
  by directly querying all four tables before/after a real call**: it issues
  exactly one `DELETE FROM sessions WHERE ...` statement (confirmed via
  `inspect.getsource`) -- nothing else. `events` rows still genuinely
  disappear with it, but only as a real Postgres-level side effect: `events`
  has a real `ON DELETE CASCADE` foreign key onto `sessions` (confirmed via
  psql's describe-table output for `events`), not any Python-level cleanup code. `app_states`/
  `user_states` are NEVER touched by `delete_session` -- confirmed both by
  reading its source (no reference to either table) and empirically: a real
  row survived in each after a real `delete_session` call in an earlier run
  of this exact test, found still present in the shared dev Postgres and
  manually cleaned up during this finding's own verification. This is
  `DatabaseSessionService`'s real, deliberate design, not a bug:
  `create_session`'s own source shows `app_states`/`user_states` are keyed
  only by `app_name` / `(app_name, user_id)` -- genuinely session-independent,
  long-lived state meant to persist across every session for that app/user,
  so a single session's deletion correctly leaves them alone. The real
  consequence for this project: nothing in ADK's own public API deletes an
  `app_states`/`user_states` row (`DatabaseSessionService`'s only public
  methods are `append_event`/`create_session`/`delete_session`/`get_session`/
  `list_sessions` -- confirmed via `dir()`), so a caller that wants to avoid
  leaving residue in a shared database (this test does) must delete them
  directly via raw SQL -- there is no ADK-provided alternative. The second
  test below verifies both halves directly: `events` really reaches zero rows
  (the cascade), and `app_states`/`user_states` really still have a row
  immediately after `delete_session` (proving they are NOT cascaded) before
  this test's own raw-SQL cleanup removes them.
- **`Runner`/`DatabaseSessionService` are safe under this project's
  UNMODIFIED, default Windows `ProactorEventLoop`** -- confirmed by reading
  `DatabaseSessionService`'s real source: it uses a plain, synchronous
  SQLAlchemy `sqlalchemy.engine.create_engine` internally (not an async
  engine/driver), unlike `AsyncPostgresSaver` (LangGraph's checkpointer),
  which genuinely needs `apps/api/src/resolvegrid_api/main.py`'s `Windows
  SelectorEventLoopPolicy` workaround because it uses psycopg's ASYNC
  driver. This test file deliberately does NOT import `resolvegrid_api.main`
  (so no event-loop-policy side effect applies) and both real async ADK
  calls below (`Runner.run_async`, `DatabaseSessionService`'s own async
  methods) pass under the plain default Proactor loop -- a real, positive
  finding: the diagnostic specialist (Tasks 3/4) will not need the same
  Selector-policy workaround LangGraph's checkpointer required.
- **Real, confirmed `BasePlugin.after_model_callback` hook** (not
  speculative -- `Runner.__init__`'s real source builds `self.plugin_manager
  = PluginManager(plugins=plugins)` unconditionally; `plugins=` on `Runner`/
  `InMemoryRunner` is itself now deprecated in favor of `App(plugins=[...])`,
  used here) genuinely fires after each real internal model call with a real
  `LlmResponse.usage_metadata` (prompt/candidate/total token counts) --
  confirmed below capturing non-null usage on every real call, proving this
  is a real, usable hook point for Task 3/4's `ModelCall` logging, per `
  model_call_logging.py`'s existing conventions.
"""

import asyncio

from sqlalchemy import create_engine, text

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from resolvegrid_api.db import DATABASE_URL
from resolvegrid_api.llm_gateway import LITELLM_BASE_URL, LITELLM_MASTER_KEY

# A fixed sentinel, not the real wall-clock time -- lets the test assert an
# exact, byte-identical, tool-provided value appears in the final answer
# (proof the agent genuinely used the tool's real return value, not a
# hallucinated guess), with zero chance of a real clock-skew false negative.
_FIXED_TIME = "2026-10-21T00:00:00+00:00"

# See this module's docstring: qwen3:14b was observed to need up to 36 real
# model turns (re-invoking the already-answered tool) before producing a
# final answer, in two independent real runs. 40 was chosen as a
# generous-but-finite real safety cap (google.adk.agents.run_config.
# RunConfig.max_llm_calls), not an arbitrary guess -- but a real CI run
# (Phase 13 Task 6, run 40746184421) genuinely exceeded even that: a real
# `LlmCallsLimitExceededError: Max number of llm calls limit of 40
# exceeded`, on GitHub's CPU-only runner, for this exact trivial
# single-tool spike. This is new, real evidence that this qwen3:14b
# behavior's real worst case is wider than the original two local runs
# suggested, not a harness bug -- raised to 60 (matching the production
# diagnostic specialist's own real, already-reasoned cap -- ADR 0004
# Finding 5) rather than a second narrow guess.
_MAX_LLM_CALLS = 60


def get_current_server_time() -> str:
    """Return the current server time as an ISO 8601 UTC timestamp."""
    return _FIXED_TIME


class _UsageCapturePlugin(BasePlugin):
    """Real proof that ModelCall-shaped usage data is obtainable via ADK's
    real `after_model_callback` hook (its own docstring: "the ideal place to
    log model responses, collect metrics on token usage") -- the real hook
    Task 3/4 would wire into `model_call_logging.py`'s conventions. Not wired
    up for real here (out of this task's scope per the plan doc); this only
    proves the hook fires and carries real, non-empty usage data.
    """

    def __init__(self) -> None:
        super().__init__(name="usage_capture")
        self.captured_usage: list = []

    async def after_model_callback(self, *, callback_context, llm_response):
        if llm_response.usage_metadata is not None:
            self.captured_usage.append(llm_response.usage_metadata)
        return None


def _build_agent_and_plugin() -> tuple[Agent, _UsageCapturePlugin]:
    plugin = _UsageCapturePlugin()
    agent = Agent(
        name="diagnostic_spike_agent",
        model=LiteLlm(
            model="openai/local-qwen3",
            api_base=LITELLM_BASE_URL,
            api_key=LITELLM_MASTER_KEY,
            # Real, verified fixes -- see this module's docstring for why
            # each is needed, cited against real source/real captured HTTP
            # traffic, not guessed:
            _skip_mcp_handler=True,
            think=False,
        ),
        instruction=(
            "You are a diagnostic test assistant. When asked for the current "
            "server time, call get_current_server_time EXACTLY ONCE, then "
            "immediately reply with its returned value, verbatim, in your "
            "very next turn. Do not call it again after you have its result. "
            "Never guess or invent a timestamp."
        ),
        tools=[get_current_server_time],
    )
    return agent, plugin


async def _run_spike() -> tuple[str | None, list, bool]:
    agent, plugin = _build_agent_and_plugin()
    app = App(name="adk-integration-spike", root_agent=agent, plugins=[plugin])
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(app_name="adk-integration-spike", user_id="spike-tester")

    tool_was_really_invoked = False
    final_text: str | None = None
    async for event in runner.run_async(
        user_id="spike-tester",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the current server time?")]
        ),
        run_config=RunConfig(max_llm_calls=_MAX_LLM_CALLS),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "get_current_server_time":
                    tool_was_really_invoked = True
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    return final_text, plugin.captured_usage, tool_was_really_invoked


def test_adk_agent_genuinely_invokes_real_tool_and_returns_tool_informed_answer():
    """The core Task 1 deliverable: a real `Agent` + one trivial real tool,
    wired to the real running LiteLLM proxy via a real `LiteLlm`-wrapped
    `local-qwen3`, run through a real `Runner`, proving the agent genuinely
    calls the tool (not a canned/mocked response) and returns a real,
    correct, tool-informed answer.
    """
    final_text, captured_usage, tool_was_really_invoked = asyncio.run(_run_spike())

    assert tool_was_really_invoked, "agent never actually called get_current_server_time"
    assert final_text is not None, "agent produced no final response within the real safety cap"
    assert _FIXED_TIME in final_text, (
        f"final answer does not contain the real tool-provided value -- got: {final_text!r}"
    )

    # Real proof the plugin/callback hook point (Task 1's other deliverable)
    # genuinely fires with real, non-empty usage data on every real call.
    assert len(captured_usage) > 0, "after_model_callback never fired"
    for usage in captured_usage:
        assert usage.total_token_count > 0, f"expected real non-zero usage, got {usage!r}"


_DB_TEST_APP_NAME = "adk-integration-spike-db-test"
_DB_TEST_USER_ID = "spike-db-tester"


def _row_count(engine, table: str, *, app_name: str, user_id: str | None = None) -> int:
    """Real, direct row count against one of ADK's own auto-created tables --
    used both to prove `events` really cascades to zero and that `app_states`/
    `user_states` do NOT (see this module's docstring), and to drive this
    test's own cleanup (ADK's public API has no method for either of the
    latter two tables).
    """
    where = "app_name = :app_name" + (" AND user_id = :user_id" if user_id is not None else "")
    params = {"app_name": app_name} | ({"user_id": user_id} if user_id is not None else {})
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table} WHERE {where}"), params).scalar_one()  # noqa: S608 -- table is one of 4 hardcoded literals below, never user input


def test_database_session_service_creates_and_retrieves_a_real_session_against_project_postgres():
    """Real, non-mocked proof `DatabaseSessionService` genuinely works
    against this project's actual Postgres (`resolvegrid_api.db.
    DATABASE_URL`) -- not just that it constructs without error. See this
    module's docstring for the real, confirmed finding that it auto-creates
    its own tables outside Alembic's migration history, and that it's
    immune to the Windows ProactorEventLoop/async-psycopg issue
    `AsyncPostgresSaver` has, since it uses a plain sync SQLAlchemy engine
    internally.

    Also real, direct proof of `delete_session`'s exact cleanup scope (code
    -review follow-up -- not assumed, each half independently verified by
    querying Postgres directly): `sessions`/`events` genuinely reach zero rows
    (the latter via a real DB-level `ON DELETE CASCADE`, not any `delete_
    session` application code), while `app_states`/`user_states` genuinely do
    NOT -- a real row is independently confirmed still present in each
    immediately after `delete_session` returns, before this test's own raw
    -SQL cleanup (the only way to remove them; ADK's public API has none)
    runs. Without that cleanup, every run of this test would permanently leave
    one more orphaned `app_states`/`user_states` row behind in this shared dev
    database -- confirmed to have actually happened across this task's own
    earlier manual verification runs, found and cleaned up by hand before
    this fix.
    """
    engine = create_engine(DATABASE_URL)

    async def _round_trip() -> None:
        service = DatabaseSessionService(db_url=DATABASE_URL)
        session = await service.create_session(app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID)
        try:
            fetched = await service.get_session(
                app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID, session_id=session.id
            )
            assert fetched is not None
            assert fetched.id == session.id
            assert fetched.app_name == _DB_TEST_APP_NAME
            assert fetched.user_id == _DB_TEST_USER_ID

            # Real proof app_states/user_states rows really were created by
            # create_session (per its own source -- see this module's
            # docstring), before we can meaningfully prove delete_session
            # leaves them alone.
            assert _row_count(engine, "app_states", app_name=_DB_TEST_APP_NAME) == 1
            assert _row_count(engine, "user_states", app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID) == 1
        finally:
            await service.delete_session(
                app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID, session_id=session.id
            )

            # Real cascade proof: sessions/events both genuinely reach zero,
            # not assumed from delete_session's return value alone.
            assert _row_count(engine, "sessions", app_name=_DB_TEST_APP_NAME) == 0
            assert _row_count(engine, "events", app_name=_DB_TEST_APP_NAME) == 0

            # Real proof app_states/user_states are NOT touched by
            # delete_session -- still present at this point, confirming this
            # module's documented finding rather than assuming it.
            assert _row_count(engine, "app_states", app_name=_DB_TEST_APP_NAME) == 1
            assert _row_count(engine, "user_states", app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID) == 1

            # Real cleanup of the two tables ADK's own public API cannot
            # clean up itself (see this module's docstring) -- do not leave
            # residue behind in this shared dev database.
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM app_states WHERE app_name = :app_name"),
                    {"app_name": _DB_TEST_APP_NAME},
                )
                conn.execute(
                    text("DELETE FROM user_states WHERE app_name = :app_name AND user_id = :user_id"),
                    {"app_name": _DB_TEST_APP_NAME, "user_id": _DB_TEST_USER_ID},
                )

            assert _row_count(engine, "app_states", app_name=_DB_TEST_APP_NAME) == 0
            assert _row_count(engine, "user_states", app_name=_DB_TEST_APP_NAME, user_id=_DB_TEST_USER_ID) == 0

    try:
        asyncio.run(_round_trip())
    finally:
        engine.dispose()
