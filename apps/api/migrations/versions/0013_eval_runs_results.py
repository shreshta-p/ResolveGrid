"""eval runs results

Phase 10 Task 1 schema: the EvalRun/EvalCaseResult tables recording batch
executions of the unified evaluation suite (Task 7's Arq job) and each
graded EvalCase outcome within a run. Schema + migration only -- no
business logic (that's Task 7's eval_worker.py).

`eval_case_result.case_id` is a plain string column, not a FK: EvalCase
fixtures live in versioned JSONL files (eval/golden/v2/*.jsonl,
eval/adversarial/v1.jsonl), not a DB table, per
resolvegrid_api.models.evaluation.EvalCaseResult's docstring.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0013'
down_revision: Union[str, None] = '0012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('eval_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_version', sa.String(), nullable=False),
    sa.Column('prompt_version', sa.String(), nullable=False),
    sa.Column('workflow_version', sa.String(), nullable=False),
    sa.Column('retriever_version', sa.String(), nullable=False),
    sa.Column('reranker_version', sa.String(), nullable=False),
    sa.Column('embeddings_version', sa.String(), nullable=False),
    sa.Column('generation_model_version', sa.String(), nullable=False),
    sa.Column('judge_version', sa.String(), nullable=False),
    sa.Column('tool_schema_version', sa.String(), nullable=False),
    sa.Column('git_commit', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('error_message', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('eval_case_result',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('eval_run_id', sa.Integer(), nullable=False),
    sa.Column('case_id', sa.String(), nullable=False),
    sa.Column('dimension', sa.String(), nullable=False),
    sa.Column('passed', sa.Boolean(), nullable=False),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('grader_type', sa.String(), nullable=False),
    sa.Column('details_json', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['eval_run_id'], ['eval_run.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_eval_case_result_eval_run_id', 'eval_case_result', ['eval_run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_eval_case_result_eval_run_id', table_name='eval_case_result')
    op.drop_table('eval_case_result')
    op.drop_table('eval_run')
