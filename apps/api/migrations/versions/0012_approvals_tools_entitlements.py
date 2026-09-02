"""approvals tools entitlements

Phase 9 Task 1 schema: the ApprovalRequest/ApprovalDecision/ApprovalPolicy
human-in-the-loop approval schema, the ToolCall tool-execution log, and the
AccessGroup/Entitlement/EmployeeEntitlement entitlement schema. Schema +
migration only -- no business logic (that's Phase 9 Tasks 2-6).

`approval_request.agent_run_id` and `tool_call.agent_run_id` are plain
string columns, not FKs, per those models' docstrings: the run identifier
they store is LangGraph's own thread/run id, not a real FK to `agent_run`
(the DB-owning `apps/api` module boundary that `services/agent-orchestration`
must not depend on).

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0012'
down_revision: Union[str, None] = '0011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('approval_request',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticket_id', sa.Integer(), nullable=True),
    sa.Column('agent_run_id', sa.String(), nullable=True),
    sa.Column('action_type', sa.String(), nullable=False),
    sa.Column('action_params_json', sa.String(), nullable=False),
    sa.Column('bound_evidence_refs_json', sa.String(), nullable=True),
    sa.Column('risk_context', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('snapshot_hash', sa.String(), nullable=False),
    sa.Column('requested_by_id', sa.Integer(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['ticket_id'], ['ticket.id'], ),
    sa.ForeignKeyConstraint(['requested_by_id'], ['employee.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('snapshot_hash')
    )
    op.create_index('ix_approval_request_status', 'approval_request', ['status'], unique=False)

    op.create_table('approval_decision',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('approval_request_id', sa.Integer(), nullable=False),
    sa.Column('approver_id', sa.Integer(), nullable=False),
    sa.Column('decision', sa.String(), nullable=False),
    sa.Column('comment', sa.String(), nullable=True),
    sa.Column('decision_evidence_snapshot_json', sa.String(), nullable=True),
    sa.Column('decided_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['approval_request_id'], ['approval_request.id'], ),
    sa.ForeignKeyConstraint(['approver_id'], ['employee.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('approval_policy',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('action_type', sa.String(), nullable=False),
    sa.Column('stages_json', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('action_type')
    )

    op.create_table('access_group',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )

    op.create_table('entitlement',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('access_group_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['access_group_id'], ['access_group.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('employee_entitlement',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('employee_id', sa.Integer(), nullable=False),
    sa.Column('entitlement_id', sa.Integer(), nullable=False),
    sa.Column('granted_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.Column('source_ticket_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['employee_id'], ['employee.id'], ),
    sa.ForeignKeyConstraint(['entitlement_id'], ['entitlement.id'], ),
    sa.ForeignKeyConstraint(['source_ticket_id'], ['ticket.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_employee_entitlement_employee_id', 'employee_entitlement', ['employee_id'], unique=False)

    op.create_table('tool_call',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_run_id', sa.String(), nullable=True),
    sa.Column('tool_name', sa.String(), nullable=False),
    sa.Column('tool_version', sa.String(), nullable=False),
    sa.Column('input_params_json', sa.String(), nullable=False),
    sa.Column('output_json', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('error_taxonomy_code', sa.String(), nullable=True),
    sa.Column('idempotency_key', sa.String(), nullable=True),
    sa.Column('approval_request_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['approval_request_id'], ['approval_request.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tool_call_idempotency_key', 'tool_call', ['idempotency_key'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_tool_call_idempotency_key', table_name='tool_call')
    op.drop_table('tool_call')
    op.drop_index('ix_employee_entitlement_employee_id', table_name='employee_entitlement')
    op.drop_table('employee_entitlement')
    op.drop_table('entitlement')
    op.drop_table('access_group')
    op.drop_table('approval_policy')
    op.drop_table('approval_decision')
    op.drop_index('ix_approval_request_status', table_name='approval_request')
    op.drop_table('approval_request')
