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
    __table_args__ = (Index("ix_model_call_purpose", "purpose"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    purpose: Mapped[str]  # e.g. "ticket.summarize" -- what the call was for
    provider: Mapped[str]
    model: Mapped[str]
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
