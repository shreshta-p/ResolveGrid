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
# final answer, in two independent real runs. 40 is a generous-but-finite
# real safety cap (google.adk.agents.run_config.RunConfig.max_llm_calls),
# not an arbitrary guess.
_MAX_LLM_CALLS = 40


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


def test_database_session_service_creates_and_retrieves_a_real_session_against_project_postgres():
    """Real, non-mocked proof `DatabaseSessionService` genuinely works
    against this project's actual Postgres (`resolvegrid_api.db.
    DATABASE_URL`) -- not just that it constructs without error. See this
    module's docstring for the real, confirmed finding that it auto-creates
    its own tables outside Alembic's migration history, and that it's
    immune to the Windows ProactorEventLoop/async-psycopg issue
    `AsyncPostgresSaver` has, since it uses a plain sync SQLAlchemy engine
    internally.
    """

    async def _round_trip() -> None:
        service = DatabaseSessionService(db_url=DATABASE_URL)
        session = await service.create_session(
            app_name="adk-integration-spike-db-test", user_id="spike-db-tester"
        )
        try:
            fetched = await service.get_session(
                app_name="adk-integration-spike-db-test",
                user_id="spike-db-tester",
                session_id=session.id,
            )
            assert fetched is not None
            assert fetched.id == session.id
            assert fetched.app_name == "adk-integration-spike-db-test"
            assert fetched.user_id == "spike-db-tester"
        finally:
            # Real cleanup -- do not leave spike rows behind in the shared
            # dev database (matches this repo's established test-hygiene
            # convention elsewhere, e.g. test_checkpoint_restore.py's own
            # `adelete_thread` cleanup).
            await service.delete_session(
                app_name="adk-integration-spike-db-test",
                user_id="spike-db-tester",
                session_id=session.id,
            )

    asyncio.run(_round_trip())
