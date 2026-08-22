"""org identity tables

Adds the Phase 2 org/identity domain schema: location, department, team,
employee, role_assignment.

employee and team have a circular FK dependency (employee.team_id ->
team.id, team.lead_employee_id -> employee.id), which is what triggered
Alembic's "Cannot correctly sort tables" autogenerate warning. Table
creation order below is fixed by hand: employee is created first without
the team_id FK, team is created next (it can reference employee, which
already exists), and the employee.team_id FK is added afterward via a
separate ALTER TABLE (op.create_foreign_key).

Retiring Phase 1's pipeline smoke-test table now that real domain schema
exists.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-21 23:44:53.027585

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "location",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("region", sa.String(), nullable=False),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "department",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("parent_department_id", sa.Integer(), nullable=True),
        sa.Column("cost_center", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["parent_department_id"], ["department.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    # employee is created before team, so its team_id FK is added later via
    # a separate ALTER TABLE once team exists (see below).
    op.create_table(
        "employee",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("employment_status", sa.String(), nullable=False),
        sa.Column("hire_date", sa.DateTime(), nullable=False),
        sa.Column("termination_date", sa.DateTime(), nullable=True),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("location_id", sa.Integer(), nullable=True),
        sa.Column("department_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("manager_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["department_id"], ["department.id"]),
        sa.ForeignKeyConstraint(["location_id"], ["location.id"]),
        sa.ForeignKeyConstraint(["manager_id"], ["employee.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_employee_manager_id", "employee", ["manager_id"], unique=False)
    op.create_table(
        "team",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("department_id", sa.Integer(), nullable=False),
        sa.Column("lead_employee_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["department_id"], ["department.id"]),
        sa.ForeignKeyConstraint(["lead_employee_id"], ["employee.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_foreign_key(
        "fk_employee_team_id_team",
        "employee",
        "team",
        ["team_id"],
        ["id"],
    )
    op.create_table(
        "role_assignment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employee_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("scope_id", sa.Integer(), nullable=True),
        sa.Column("granted_by_id", sa.Integer(), nullable=True),
        sa.Column("granted_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["employee_id"], ["employee.id"]),
        sa.ForeignKeyConstraint(["granted_by_id"], ["employee.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    # Retiring Phase 1's pipeline smoke-test table now that real domain schema exists.
    op.drop_table("health_check")


def downgrade() -> None:
    op.create_table(
        "health_check",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "checked_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("health_check_pkey")),
    )
    op.drop_table("role_assignment")
    op.drop_constraint("fk_employee_team_id_team", "employee", type_="foreignkey")
    op.drop_table("team")
    op.drop_index("ix_employee_manager_id", table_name="employee")
    op.drop_table("employee")
    op.drop_table("department")
    op.drop_table("location")
