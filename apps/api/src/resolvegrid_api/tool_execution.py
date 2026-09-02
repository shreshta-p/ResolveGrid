"""Tool allowlist filtering + params schema validation (Phase 9 Task 4).

Three-step flow a caller (a future LangGraph "select tool" node -- Task 5/6,
NOT wired up here) is expected to follow, in order:

1. `available_tools_for_principal(principal, held_entitlements)` -- filter
   `resolvegrid_contracts.tools.TOOL_REGISTRY` down to the tools this
   specific principal may even be offered. **This must run first, and only
   its result may ever be formatted into a prompt or handed to a model** --
   never format the full `TOOL_REGISTRY` for a model and filter afterwards.
2. `select_tool(tool_name, available)` -- resolve a model's (or a UI's)
   chosen tool name against that filtered list.
3. `validate_tool_schema(tool, params)` -- validate the proposed call's
   params against the tool's `params_schema` before execution.

Deliberately NOT covered by this module: actually invoking a tool's
underlying adapter function (Task 3's `operational_adapters/entitlements.py`
et al.), approval-gating a `mutating`/`requires_approval` tool (Task 5), or
wiring any of this into the agent graph (Task 5/6).
"""

from __future__ import annotations

import jsonschema

from resolvegrid_authz import Principal, principal_has_role
from resolvegrid_contracts.tools import TOOL_REGISTRY, ToolContract


class ToolNotAllowedError(Exception):
    """Raised by `select_tool` when a requested tool name cannot be used.

    Deliberately raised for BOTH "no such tool exists in TOOL_REGISTRY at
    all" and "the tool exists but this principal isn't allowed it" -- same
    exception type, same message shape, in both cases. This is intentional:
    per plan.md's safe-error-envelope guidance, an error surfaced back to a
    model (or a UI) must never leak *why* a tool name was rejected, since
    "doesn't exist" vs. "exists but you're not permitted" is itself
    information a caller could use to probe the registry/allowlist. Callers
    must not pattern-match on this exception's message to recover which
    case occurred.
    """


class ToolValidationError(Exception):
    """Raised by `validate_tool_schema` when `params` doesn't conform to a
    tool's `params_schema`. Carries the underlying `jsonschema` validation
    error message (not a security-sensitive detail -- unlike
    `ToolNotAllowedError`, params-shape errors are meant to be actionable
    for whoever/whatever proposed the call, e.g. so a model can retry with
    a corrected payload).
    """


def available_tools_for_principal(
    principal: Principal, held_entitlements: frozenset[str] | None = None
) -> list[ToolContract]:
    """Filter `TOOL_REGISTRY` down to the tools `principal` may be offered.

    A tool is included iff:
    - `principal_has_role(principal, tool.required_role)` is true (delegates
      to `packages/authz`'s centralized role check -- see that function's
      docstring for why this isn't `authorize()` directly), AND
    - `tool.required_entitlement` is `None`, OR it's present in
      `held_entitlements`.

    `held_entitlements` design note: neither of today's two registered
    tools sets `required_entitlement` (both are `None`), so this parameter
    is exercised by no current tool -- it exists for completeness/forward
    compatibility with future tools that do set it. It's an explicit
    `frozenset[str] | None` (default `None`, treated as "holds no
    entitlements") rather than this function reaching into a DB itself,
    because this is meant to be a pure, synchronous allowlist filter with
    no I/O -- callers that need entitlement membership (e.g. Task 3's
    `lookup_employee_entitlements` DB-backed adapter) are expected to
    resolve `held_entitlements` themselves (typically via that same
    adapter) and pass the result in, rather than this function acquiring
    its own `Session`.

    This is the ONLY function whose output may be formatted into a prompt
    or shown in a UI as "tools available to this principal" -- see this
    module's docstring.
    """
    held = held_entitlements or frozenset()
    return [
        tool
        for tool in TOOL_REGISTRY.values()
        if principal_has_role(principal, tool.required_role)
        and (tool.required_entitlement is None or tool.required_entitlement in held)
    ]


def select_tool(tool_name: str, available: list[ToolContract]) -> ToolContract:
    """Resolve `tool_name` against an already-filtered `available` list
    (the output of `available_tools_for_principal` -- never `TOOL_REGISTRY`
    directly).

    Raises `ToolNotAllowedError` if no tool in `available` has a matching
    `.name`, whether because the name doesn't exist in `TOOL_REGISTRY` at
    all or because it was filtered out by `available_tools_for_principal`
    -- see `ToolNotAllowedError`'s docstring for why both cases are
    indistinguishable to the caller.
    """
    for tool in available:
        if tool.name == tool_name:
            return tool
    raise ToolNotAllowedError(f"tool not allowed: {tool_name}")


def validate_tool_schema(tool: ToolContract, params: dict) -> dict:
    """Validate `params` against `tool.params_schema` (a JSON Schema dict).

    Uses the `jsonschema` library (added as a new `apps/api` dependency for
    this task -- it wasn't already a workspace dependency). Chosen over a
    hand-rolled required-keys/type/additionalProperties check because
    `params_schema` is authored as real JSON Schema (see
    `resolvegrid_contracts.tools.TOOL_REGISTRY`'s `additionalProperties`,
    `required`, per-property `type` keys) and `jsonschema` is a small, pure
    -Python, widely-used library with no heavy transitive deps (unlike e.g.
    a reranker pulling in torch) -- re-implementing a subset of JSON Schema
    semantics inline would just be a second, drifting copy of what
    `jsonschema` already does correctly.

    Returns `params` unchanged if valid. Raises `ToolValidationError`
    (wrapping the underlying `jsonschema.exceptions.ValidationError`'s
    message) if not -- e.g. a missing required param, a wrong type, or an
    extra param not in `params_schema["properties"]` (every registered
    tool's schema sets `additionalProperties: false`).

    Does NOT guard against `jsonschema.exceptions.SchemaError` (a
    malformed `params_schema` itself, as opposed to malformed `params`).
    `params_schema` is static and developer-authored in `TOOL_REGISTRY`
    (see `resolvegrid_contracts.tools`), not user/model-controlled input,
    so a malformed schema is a registry bug to fix at the source rather
    than a runtime condition this function needs to translate into
    `ToolValidationError`.
    """
    try:
        jsonschema.validate(instance=params, schema=tool.params_schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ToolValidationError(
            f"params for tool '{tool.name}' failed schema validation: {exc.message}"
        ) from exc
    return params
