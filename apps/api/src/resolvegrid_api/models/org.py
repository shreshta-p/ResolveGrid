from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from resolvegrid_api.models.base import Base


class Location(Base):
    __tablename__ = "location"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    region: Mapped[str]
    timezone: Mapped[str]


class Department(Base):
    __tablename__ = "department"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    parent_department_id: Mapped[int | None] = mapped_column(ForeignKey("department.id"))
    cost_center: Mapped[str | None] = mapped_column(default=None)

    parent: Mapped["Department | None"] = relationship(remote_side=[id])


class Team(Base):
    __tablename__ = "team"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    department_id: Mapped[int] = mapped_column(ForeignKey("department.id"))
    lead_employee_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))


class Employee(Base):
    __tablename__ = "employee"
    __table_args__ = (Index("ix_employee_manager_id", "manager_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str]
    employment_status: Mapped[str] = mapped_column(default="active")
    hire_date: Mapped[datetime]
    termination_date: Mapped[datetime | None] = mapped_column(default=None)
    timezone: Mapped[str]
    location_id: Mapped[int | None] = mapped_column(ForeignKey("location.id"))
    department_id: Mapped[int | None] = mapped_column(ForeignKey("department.id"))
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("team.id", use_alter=True, name="fk_employee_team_id_team")
    )
    manager_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))

    manager: Mapped["Employee | None"] = relationship(remote_side=[id])


class RoleAssignment(Base):
    __tablename__ = "role_assignment"
    __table_args__ = (Index("ix_role_assignment_employee_id", "employee_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employee.id"))
    role: Mapped[str]
    scope: Mapped[str] = mapped_column(default="global")
    scope_id: Mapped[int | None] = mapped_column(default=None)
    granted_by_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))
    granted_at: Mapped[datetime] = mapped_column(server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(default=None)


class AccessGroup(Base):
    __tablename__ = "access_group"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str | None] = mapped_column(default=None)


class Entitlement(Base):
    __tablename__ = "entitlement"

    id: Mapped[int] = mapped_column(primary_key=True)
    access_group_id: Mapped[int] = mapped_column(ForeignKey("access_group.id"))
    name: Mapped[str]
    description: Mapped[str | None] = mapped_column(default=None)


class EmployeeEntitlement(Base):
    __tablename__ = "employee_entitlement"
    __table_args__ = (Index("ix_employee_entitlement_employee_id", "employee_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employee.id"))
    entitlement_id: Mapped[int] = mapped_column(ForeignKey("entitlement.id"))
    granted_at: Mapped[datetime] = mapped_column(server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(default=None)
    source_ticket_id: Mapped[int | None] = mapped_column(ForeignKey("ticket.id"), default=None)
