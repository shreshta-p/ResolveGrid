from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, false, func
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
    __table_args__ = (
        # Phase 12 Task 5 code-review follow-up (migration 0021): a real DB
        # CHECK, not just a convention both real write sites in `routers/
        # chat.py` happen to follow -- `cache_hit_kind` naming WHICH cache
        # served a run only ever makes sense when `cache_hit` itself is
        # true; `cache_hit_kind` set alongside `cache_hit=False` would be a
        # structurally nonsensical row ("which cache served a run that
        # wasn't a cache hit at all"). This makes that combination
        # impossible to write, not just unlikely.
        CheckConstraint(
            "cache_hit_kind IS NULL OR cache_hit = true",
            name="ck_agent_run_cache_hit_kind_requires_cache_hit",
        ),
    )

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
    # this column existed -- `server_default=false()` backfills every
    # pre-existing row to the honest value, since none of them could have
    # been a cache hit (this cache didn't exist yet). Declared at BOTH
    # layers (`default=False` for ORM-side inserts, `server_default=
    # false()` for the real DB-level default migration 0017 already
    # applies) -- matching `created_at`'s existing precedent just above of
    # declaring the DB-level default explicitly here too, not just in the
    # migration, so a future `alembic revision --autogenerate` sees the
    # model and the schema agree and proposes no spurious drift fix.
    cache_hit: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Phase 12 Task 5: which cache mechanism (if any) served this run --
    # "exact" (resolvegrid_api.response_cache, Task 3's byte-identical
    # SHA-256 match) | "semantic" (resolvegrid_api.semantic_cache, Task 5's
    # embedding-similarity match) | None (a genuine real-graph run; also
    # None for every row written before this column existed). Deliberately
    # a second, nullable column rather than widening `cache_hit` itself into
    # a 3-valued string: `cache_hit` already has an established boolean
    # contract (`test_chat_api.py`'s `run_2.cache_hit is True` assertion,
    # Task 3) that every existing caller/test/dashboard query keys off --
    # changing its type would be a breaking change to that contract for no
    # real benefit, since "was this a cache hit at all" (the boolean) and
    # "which cache kind" (this column) are two genuinely different
    # questions a reader may want answered independently (e.g. a dashboard
    # computing an overall cache-hit-rate only ever needs the boolean).
    # `cache_hit` stays True for BOTH "exact" and "semantic" rows -- this
    # column adds the finer-grained distinction on top, it never contradicts
    # the boolean.
    cache_hit_kind: Mapped[str | None] = mapped_column(default=None)


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
