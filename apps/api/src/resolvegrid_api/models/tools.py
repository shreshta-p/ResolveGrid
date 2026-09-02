from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class ToolCall(Base):
    """One invocation of a tool (read-only or mutating) by an agent run.

    `agent_run_id` is a plain string, not a FK -- same rationale as
    `ApprovalRequest.agent_run_id` (see that model's docstring):
    `services/agent-orchestration` identifies runs by LangGraph's own
    `thread_id`/run identifier and must not import `apps/api`'s DB models.
    """

    __tablename__ = "tool_call"
    __table_args__ = (
        Index("ix_tool_call_idempotency_key", "idempotency_key"),
        Index("ix_tool_call_approval_request_id", "approval_request_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_run_id: Mapped[str | None] = mapped_column(default=None)
    tool_name: Mapped[str]
    tool_version: Mapped[str]
    input_params_json: Mapped[str]
    output_json: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str]  # "success" | "error" | "timeout" | "dry_run"
    error_taxonomy_code: Mapped[str | None] = mapped_column(default=None)
    # Used by the mutation-execution path's duplicate-replay guard (Task 6):
    # e.g. f"approval:{approval_request_id}" so a re-executed graph node can
    # detect an already-successful call instead of re-running the adapter.
    idempotency_key: Mapped[str | None] = mapped_column(default=None)
    approval_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("approval_request.id"), default=None
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
