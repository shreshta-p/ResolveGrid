# ADR 0003: Google ADK Real Integration Findings (Phase 13 Task 1)

## Status
Accepted

## Context
Phase 13's plan doc (`docs/superpowers/plans/2026-10-21-phase13-adk-diagnostic-specialist.md`) was written from secondhand web research against ADK's docs site (`adk.dev`), which could not confirm exact constructor signatures, method behavior, or several real integration risks. Task 1 was a real, non-mocked integration spike against the actually-installed `google-adk` package and this project's live dev stack (real Postgres, real LiteLLM proxy, real Ollama `qwen3:14b`), to empirically confirm or correct those assumptions before any later task builds the real diagnostic specialist on top of them. Every finding below is cited against real source read directly, real `inspect.signature()` output, or a real, reproduced call — not web search. Full detail and exact commands/outputs are in `apps/api/tests/test_adk_integration_spike.py`'s module docstring; this ADR records the decisions that follow from them.

## Finding 1: installed version is 1.14.1, not 2.12.0
The plan doc's web research found `2.12.0` "as currently published." `uv add --package resolvegrid-api google-adk` resolves and installs `1.14.1` into this workspace. Real signatures/behavior throughout this phase are pinned to 1.14.1, not 2.12.0 — re-verify against the installed package again if a future phase bumps this dependency, rather than trusting either version number blindly.

## Finding 2: `LiteLlm` requires `google-adk[extensions]`, not bare `google-adk`
`from google.adk.models.lite_llm import LiteLlm` raises `ModuleNotFoundError: No module named 'litellm'` on a bare install — `litellm` is gated behind the `extensions` extra (`pip`/`uv` metadata: `litellm>=1.75.5 ; extra == "extensions"`), not a core dependency. `apps/api/pyproject.toml` depends on `google-adk[extensions]>=1.14.1`, not the bare package.

## Finding 3 (real dependency conflict, real fix): `google-adk`'s `mcp<2.0.0` pin vs. `litellm`'s unconditional MCP-proxy import
`google-adk` pins `mcp>=1.8.0,<2.0.0`. `litellm`'s own `proxy`/`mcp` extras need `mcp>=2.2.0,<3` — mutually exclusive ranges with `google-adk`'s pin. Neither litellm extra is used by this project, but `litellm/main.py`'s real `completion()` unconditionally imports `litellm.responses.mcp.chat_completions_handler` (which needs `mcp>=2.2.0`'s `CacheableResult`, absent from the resolved `mcp==1.28.1`) whenever a call passes `tools=[...]` — i.e. on every real ADK tool-calling request, regardless of whether litellm's MCP-gateway feature is ever invoked.

**Decision**: use litellm's own real, documented escape hatch for this exact branch — `LiteLlm(..., _skip_mcp_handler=True)` — rather than pinning `mcp>=2.2.0` (which would break `google-adk` itself) or vendoring/patching litellm. `LiteLlm.__init__`'s `**kwargs` pass straight through to the real `litellm.acompletion()` call, confirmed via its source. Revisit trigger: either project relaxes its pin in a future release such that both extras' real ranges overlap.

## Finding 4: qwen3:14b's default "thinking" mode breaks tool-calling outright (not just cost/latency)
`llm_gateway.py` already documents qwen3:14b's reasoning mode costing ~60x tokens/latency for non-reasoning calls. This task found a second, more severe real effect: without `"think": false`, a real tool-calling request to `local-qwen3` returns a bare literal `"{}"` text reply (`finish_reason: "stop"`, 2 completion tokens) instead of a real tool call — reproduced independently of ADK via a direct `curl` against the live LiteLLM proxy with the identical tool schema. Adding `"think": false` to the same call immediately produces a correct `finish_reason: "tool_calls"` response.

**Decision**: `LiteLlm(..., think=False)` is required for every real tool-calling call to `local-qwen3` through this project's ADK integration, not optional tuning. `LiteLlm`'s `**kwargs` passthrough (same mechanism as Finding 3) is sufficient — no ADK-side change needed.

## Finding 5 (real, quantified model limitation — not an ADK/LiteLlm defect): qwen3:14b re-invokes an already-answered tool several times before finally answering
A `litellm._turn_on_debug()` capture of the real second-turn HTTP request (after the tool's real result was already returned) shows a correctly OpenAI-shaped conversation (`role: "tool"`, matching `tool_call_id`, the real tool result present, a system instruction explicitly saying "do not call it again") — and qwen3 still chose `finish_reason: "tool_calls"` again, re-invoking the identical tool with identical empty arguments. Reproduced across two independent real runs: one uncapped run took 36 real model turns before finally answering; a second, differently-capped run repeated the identical immediate-retry pattern from the very first opportunity.

This rules out an ADK/LiteLlm request-shaping bug — the real request litellm sent was correct — and points to a genuine qwen3:14b behavior under non-thinking-mode, real tool-calling decoding.

**Decision**: `google.adk.agents.run_config.RunConfig.max_llm_calls` (a real, confirmed hard cap — defaults to 500, raises `LlmCallsLimitExceededError` when exceeded) is the real, verified bounding mechanism, used in both the spike test (capped at 40, comfortably above the observed 36-call worst case) and required in Tasks 3/4's real production wiring. This is flagged as a real, open risk for Tasks 3–5's production diagnostic-specialist prompt/tool design to account for deliberately (e.g. removing a tool from the tool list after its first successful use, or a stricter per-turn termination check) — it is not solved by this spike, which only proves the mechanism and bounds the worst case safely.

## Finding 6: `DatabaseSessionService` genuinely works against this project's real Postgres, but bypasses Alembic
A real `create_session`/`get_session`/`delete_session` round trip against `resolvegrid_api.db.DATABASE_URL` succeeds. It silently auto-creates its own `sessions`/`events`/`app_states`/`user_states` tables on first use (confirmed via `psql`), outside this project's Alembic migration history — no `prepare_tables()` method exists on the real installed service (the plan doc's secondhand research mentioned one); `create_session` creates what it needs lazily instead.

**Decision (deferred, not resolved here)**: Tasks 3/4 must decide deliberately how to reconcile this with the project's "every schema change is a migration" convention (e.g. a dedicated Alembic migration mirroring ADK's own table DDL, or an explicitly accepted, documented exception) before any non-local/shared deployment — silently inheriting ADK-managed tables with no migration history is a real gap, flagged here per this phase's own "report honestly" standing directive.

## Finding 7 (positive): `Runner`/`DatabaseSessionService` are safe under the unmodified, default Windows `ProactorEventLoop`
`DatabaseSessionService`'s real source uses a plain synchronous SQLAlchemy `create_engine` internally (not an async engine/driver) — unlike `AsyncPostgresSaver` (LangGraph's checkpointer, ADR 0002), which genuinely needs `main.py`'s `WindowsSelectorEventLoopPolicy` workaround because it uses psycopg's async driver. `apps/api/tests/test_adk_integration_spike.py` deliberately never imports `resolvegrid_api.main` (so no event-loop-policy side effect applies), and both real async ADK calls (`Runner.run_async`, `DatabaseSessionService`'s own async methods) pass under the plain default Proactor loop. The diagnostic specialist (Tasks 3/4) will not need the Selector-policy workaround LangGraph's checkpointer required.

## Finding 8: `BasePlugin.after_model_callback` is a real, confirmed hook for per-call usage capture
`Runner.__init__`'s real source builds `self.plugin_manager = PluginManager(plugins=plugins)` unconditionally; `plugins=` on `Runner`/`InMemoryRunner` is itself now deprecated in favor of `App(plugins=[...])`. `BasePlugin.after_model_callback(*, callback_context, llm_response)` genuinely fires after each real internal model call with a real, non-null `LlmResponse.usage_metadata` (prompt/candidate/total token counts) — confirmed in the spike test, which asserts non-zero usage on every real call. This is the real hook point Tasks 3/4 should wire into `model_call_logging.py`'s existing conventions, replacing the plan doc's unconfirmed speculation about a `plugin_manager` attribute with a real, working mechanism.

## Consequences
- `apps/api/pyproject.toml` depends on `google-adk[extensions]>=1.14.1`, with `LiteLlm(..., _skip_mcp_handler=True, think=False)` required on every real tool-calling call to `local-qwen3` — both must be carried forward into Tasks 3/4's production wiring, not just this spike's test.
- Task 3/4 must make a deliberate, documented decision about `DatabaseSessionService`'s Alembic-bypassing table creation before any non-local use, and about a production-safe `max_llm_calls`/tool-availability strategy given qwen3's observed repeated-tool-call tendency.
- No Windows event-loop workaround is needed for the ADK integration itself, simplifying Tasks 3/4's wiring relative to the LangGraph checkpointer precedent.
