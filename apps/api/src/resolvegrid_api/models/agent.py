from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class AgentRun(Base):
    """One LangGraph agent invocation (Phase 6: a single `/chat` turn).

    `thread_id` is LangGraph's own checkpoint key -- unique because each
    AgentRun corresponds to exactly one checkpointed thread in the
    AsyncPostgresSaver tables, and this row is this codebase's queryable
    record of that thread (status/timing/output) alongside it.
    """

    __tablename__ = "agent_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(unique=True)  # LangGraph's checkpoint key
    # Real FK to employee.id (not left as a bare int): matches AuditLog.actor_id's
    # precedent of a nullable-but-constrained principal reference. The debug
    # employee-id header this codebase currently uses for auth always resolves
    # to a real seeded employee row before an AgentRun is created, so the FK
    # holds in practice; nullable because a future unauthenticated/system-
    # initiated run (e.g. a scheduled job) may legitimately have no employee
    # principal at all.
    principal_employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee.id"), default=None
    )
    status: Mapped[str] = mapped_column(default="running")  # "running" | "completed" | "error"
    input_text: Mapped[str]
    output_text: Mapped[str | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    # Phase 12 Task 3: True iff this run was served entirely from
    # `resolvegrid_api.response_cache`'s exact-response cache -- the real
    # graph (classify_intent/retrieve/compose_response/verify_citations)
    # never executed for this row. `status` stays "completed" either way
    # (a cache hit is a genuinely successful run, not a distinct status),
    # so this is a real, dedicated column rather than overloading `status`
    # with a third meaning or burying the signal in `Span.detail_json`
    # (see migration 0017's docstring for the full "why a column, and why
    # here, not there" reasoning). Never True for any row written before
    # this column existed -- `server_default=false` backfills every
    # pre-existing row to the honest value, since none of them could have
    # been a cache hit (this cache didn't exist yet).
    cache_hit: Mapped[bool] = mapped_column(default=False)


class Span(Base):
    """One stage (node) of an AgentRun's graph execution.

    Real FK to agent_run.id (not left unconstrained): the AgentRun row is
    always created before any Span referencing it, matching this codebase's
    established precedent (ModelCall.pricing_version_id, AuditLog.actor_id)
    of using real FKs wherever the referenced row is guaranteed to exist first.
    """

    __tablename__ = "span"
    __table_args__ = (Index("ix_span_agent_run_id", "agent_run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_run.id"))
    stage_name: Mapped[str]  # e.g. "classify_intent", "compose_response", "finalize"
    status: Mapped[str]  # "success" | "error"
    latency_ms: Mapped[int]
    detail_json: Mapped[str | None] = mapped_column(default=None)  # small structured extra (e.g. classified intent)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
