from datetime import datetime

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from resolvegrid_api.models.base import Base


class Queue(Base):
    __tablename__ = "queue"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    department_id: Mapped[int | None] = mapped_column(ForeignKey("department.id"))
    description: Mapped[str | None] = mapped_column(default=None)


class Ticket(Base):
    __tablename__ = "ticket"
    __table_args__ = (
        Index("ix_ticket_queue_id", "queue_id"),
        Index("ix_ticket_requester_id", "requester_id"),
        Index("ix_ticket_assignee_id", "assignee_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str]
    type: Mapped[str]
    category: Mapped[str | None] = mapped_column(default=None)
    priority: Mapped[str] = mapped_column(default="medium")
    status: Mapped[str] = mapped_column(default="open")
    queue_id: Mapped[int] = mapped_column(ForeignKey("queue.id"))
    requester_id: Mapped[int] = mapped_column(ForeignKey("employee.id"))
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))
    sla_due_at: Mapped[datetime | None] = mapped_column(default=None)
    source: Mapped[str] = mapped_column(default="web")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class TicketMessage(Base):
    __tablename__ = "ticket_message"
    __table_args__ = (Index("ix_ticket_message_ticket_id", "ticket_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("ticket.id"))
    author_type: Mapped[str]
    author_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))
    body: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TicketStateTransition(Base):
    __tablename__ = "ticket_state_transition"
    __table_args__ = (Index("ix_ticket_state_transition_ticket_id", "ticket_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("ticket.id"))
    from_status: Mapped[str]
    to_status: Mapped[str]
    actor_type: Mapped[str]
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))
    reason: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_type: Mapped[str]
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("employee.id"))
    action: Mapped[str]
    entity_type: Mapped[str]
    entity_id: Mapped[int]
    before_json: Mapped[str | None] = mapped_column(default=None)
    after_json: Mapped[str | None] = mapped_column(default=None)
    # Named metadata_json, not metadata -- "metadata" is a reserved attribute
    # name on SQLAlchemy declarative models (collides with Base.metadata).
    metadata_json: Mapped[str | None] = mapped_column(default=None)
    record_hash: Mapped[str] = mapped_column(unique=True)
    previous_record_hash: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
