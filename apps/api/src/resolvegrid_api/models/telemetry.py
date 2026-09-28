from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class PricingVersion(Base):
    """A versioned snapshot of per-token pricing for one provider+model pair.

    Never mutated once created -- ModelCall rows reference a specific
    PricingVersion so historical cost never changes retroactively when
    current prices change (approved architecture plan §9).
    """

    __tablename__ = "pricing_version"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str]
    model: Mapped[str]
    input_cost_per_1k_tokens_usd: Mapped[float]
    output_cost_per_1k_tokens_usd: Mapped[float]
    effective_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ModelCall(Base):
    __tablename__ = "model_call"
    __table_args__ = (
        Index("ix_model_call_purpose", "purpose"),
        Index("ix_model_call_trace_id", "trace_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    purpose: Mapped[str]  # e.g. "ticket.summarize" -- what the call was for
    provider: Mapped[str]
    model: Mapped[str]
    # No fallback_reason/routing_reason free-text column: LiteLLM's response
    # never carries the upstream error text that caused a fallback (only a
    # successful fallback response's HEADERS do, and only whether/which --
    # x-litellm-attempted-fallbacks / x-litellm-model-group; the actual
    # "why" is swallowed internally after a successful retry, empirically
    # verified against real live providers). Storing a reason string here
    # would mean fabricating data this system can never actually observe.
    fallback_occurred: Mapped[bool] = mapped_column(default=False)
    # Nullable: a call with no fallback config at all (e.g. local-qwen3) has
    # no x-litellm-model-group concept worth recording, and a call that
    # errored before a response came back has no group to record either.
    serving_model_group: Mapped[str | None] = mapped_column(default=None)
    # Real FK (not left unconstrained): every provider/model this phase can
    # call always has a matching PricingVersion row seeded ahead of time
    # (migration 0006 seeds the $0 ollama/qwen3:14b row before any ModelCall
    # referencing it could be written), matching the precedent set by
    # AuditLog.actor_id / Ticket.assignee_id elsewhere in this codebase: a
    # nullable column can still carry a real FK constraint. Nullable because
    # a ModelCall that failed before pricing lookup (e.g. gateway error) may
    # legitimately have no pricing row to point at.
    pricing_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("pricing_version.id"), default=None
    )
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    latency_ms: Mapped[int]
    estimated_cost_usd: Mapped[float]
    status: Mapped[str]  # "success" | "error"
    error_message: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # The real OTel trace id (32 lowercase hex chars) the completion's span
    # was recorded under -- lets a ModelCall row be correlated to a real
    # trace in Langfuse (Phase 11 Task 5) once traces are actually exported
    # somewhere queryable. Nullable: rows written before this column existed
    # (migration 0014) have none, and that's fine -- never backfilled.
    # Indexed (ix_model_call_trace_id) for the same reason `purpose` is
    # indexed just above: "find all ModelCalls for this trace" is a
    # point-lookup-by-value access pattern, and this is exactly the query
    # Task 5's Langfuse correlation work will run.
    trace_id: Mapped[str | None] = mapped_column(default=None)
    # Phase 11 Task 4: which real POLICY INPUT drove the model= choice for
    # this call -- distinct from fallback_occurred/serving_model_group
    # above (those record what LiteLLM's ROUTER did after the request was
    # already sent); this records what THIS APPLICATION'S routing policy
    # decided BEFORE the request was ever sent, and why. Judged genuinely
    # not redundant with the existing columns: `model`/`provider` alone
    # tell you WHICH model served a call, never WHY that model was chosen
    # over another -- for `purpose="chat.compose_response"` specifically
    # (the only call site with a real routing policy as of this task -- see
    # `routing.py`), `model="cloud-primary"` is ambiguous between "this
    # call's risk_level was classified high" (the real, current policy) and
    # some future, different policy input entirely; a free-text
    # `routing_reason` (e.g. "risk_level=high") keeps that provenance
    # genuinely inspectable without forcing every future routing input this
    # system might add (time-of-day, principal role, ...) into its own
    # dedicated typed column. `None` for every call site with no real
    # routing policy yet (`ticket.summarize`, `chat.classify_intent`,
    # judge/eval closures -- all still always request `DEFAULT_MODEL`
    # unconditionally, per those modules' own docstrings).
    routing_reason: Mapped[str | None] = mapped_column(default=None)
