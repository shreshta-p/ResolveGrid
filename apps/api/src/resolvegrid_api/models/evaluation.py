from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class EvalRun(Base):
    """One batch execution of the Phase 10 evaluation suite (Task 7's Arq
    job) against the real running system.

    The full version tuple (`dataset_version` through `tool_schema_version`,
    plus `git_commit`) is recorded on every row so a later run can be
    compared against an earlier one with full confidence about what
    actually changed between them -- mirrors `ingestion_worker.py`'s
    `PARSER_VERSION`/`CHUNKING_VERSION` etc. constants, generalized across
    every version-tagged component this system has (prompt, workflow,
    retriever, reranker, embeddings, generation model, judge, tool schema).
    """

    __tablename__ = "eval_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version: Mapped[str]
    prompt_version: Mapped[str]
    workflow_version: Mapped[str]
    retriever_version: Mapped[str]
    reranker_version: Mapped[str]
    embeddings_version: Mapped[str]
    generation_model_version: Mapped[str]
    judge_version: Mapped[str]
    tool_schema_version: Mapped[str]
    git_commit: Mapped[str]
    status: Mapped[str] = mapped_column(default="running")  # "running" | "completed" | "error"
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)


class EvalCaseResult(Base):
    """One graded `EvalCase` outcome within an `EvalRun`.

    `case_id` is a plain string matching `EvalCase.case_id`
    (`resolvegrid_evaluation.schema`), not a FK -- the eval case fixtures
    live in versioned JSONL files (`eval/golden/v2/*.jsonl`,
    `eval/adversarial/v1.jsonl`), not a DB table, so there is no row for
    this column to reference. This mirrors `ApprovalRequest.agent_run_id`/
    `ToolCall.agent_run_id`'s established precedent for identifiers that
    name something outside this DB's own tables.
    """

    __tablename__ = "eval_case_result"
    __table_args__ = (Index("ix_eval_case_result_eval_run_id", "eval_run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_run_id: Mapped[int] = mapped_column(ForeignKey("eval_run.id"))
    case_id: Mapped[str]
    dimension: Mapped[str]  # "chat" | "retrieval" | "tool" | "approval" | "adversarial"
    passed: Mapped[bool]
    score: Mapped[float | None] = mapped_column(default=None)  # some graders are pass/fail only
    grader_type: Mapped[str]  # "deterministic" | "judge"
    details_json: Mapped[str | None] = mapped_column(default=None)  # grader's full structured output
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
