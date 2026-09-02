from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class ApprovalRequest(Base):
    """A pending/decided request for human sign-off on a proposed agent action.

    `agent_run_id` is stored as a plain string (not a FK), matching
    `ToolCall.agent_run_id`'s rationale: `services/agent-orchestration` must
    not import `apps/api`'s DB models (this codebase's established
    dependency-direction rule -- see `graph.py`'s module docstring), so the
    node that creates this row identifies the run by LangGraph's own
    `thread_id`/run identifier rather than a real FK. Phase 10's
    `EvalRun`/telemetry work is expected to formalize this into a proper
    reference once that boundary is revisited.
    """

    __tablename__ = "approval_request"
    __table_args__ = (Index("ix_approval_request_status", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int | None] = mapped_column(ForeignKey("ticket.id"), default=None)
    agent_run_id: Mapped[str | None] = mapped_column(default=None)
    action_type: Mapped[str]
    action_params_json: Mapped[str]
    bound_evidence_refs_json: Mapped[str | None] = mapped_column(default=None)
    risk_context: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="pending")  # "pending" | "approved" | "rejected" | "expired"
    # sha256 hex digest over the approval's bound fields (see Task 5's
    # request_approval node for the exact composition) -- unique so a
    # re-executed graph node can upsert idempotently by this value instead
    # of inserting a duplicate row.
    snapshot_hash: Mapped[str] = mapped_column(unique=True)
    requested_by_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"), default=None)
    expires_at: Mapped[datetime]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class ApprovalDecision(Base):
    """One approver's decision (approve/reject) against an ApprovalRequest."""

    __tablename__ = "approval_decision"

    id: Mapped[int] = mapped_column(primary_key=True)
    approval_request_id: Mapped[int] = mapped_column(ForeignKey("approval_request.id"))
    approver_id: Mapped[int] = mapped_column(ForeignKey("employee.id"))
    decision: Mapped[str]  # "approved" | "rejected"
    comment: Mapped[str | None] = mapped_column(default=None)
    decision_evidence_snapshot_json: Mapped[str | None] = mapped_column(default=None)
    decided_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ApprovalPolicy(Base):
    """The staged approval requirement (e.g. peer-review then higher-authority) for one action_type."""

    __tablename__ = "approval_policy"

    id: Mapped[int] = mapped_column(primary_key=True)
    action_type: Mapped[str] = mapped_column(unique=True)
    stages_json: Mapped[str]  # ordered list of {role, scope} stage requirements
    description: Mapped[str | None] = mapped_column(default=None)
