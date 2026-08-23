from resolvegrid_api.models import AuditLog, Department, Employee, Location, Queue, Ticket, TicketMessage, TicketStateTransition


def test_can_create_full_ticket_graph(db_session):
    location = Location(name="Test HQ 3", region="US", timezone="America/Chicago")
    department = Department(name="Test Dept 3")
    db_session.add_all([location, department])
    db_session.flush()

    requester = Employee(
        display_name="Requester", email="requester3@example.test", title="Engineer",
        hire_date="2024-01-01T00:00:00", timezone=location.timezone,
        location_id=location.id, department_id=department.id,
    )
    db_session.add(requester)
    db_session.flush()

    queue = Queue(name="Test Queue 3", department_id=department.id)
    db_session.add(queue)
    db_session.flush()

    ticket = Ticket(
        subject="VPN down", type="incident", priority="high",
        queue_id=queue.id, requester_id=requester.id,
    )
    db_session.add(ticket)
    db_session.flush()

    message = TicketMessage(
        ticket_id=ticket.id, author_type="employee", author_id=requester.id,
        body="Can't connect since this morning",
    )
    transition = TicketStateTransition(
        ticket_id=ticket.id, from_status="open", to_status="in_progress",
        actor_type="analyst", reason="picked up",
    )
    audit = AuditLog(
        actor_type="employee", actor_id=requester.id, action="ticket.create",
        entity_type="ticket", entity_id=ticket.id, record_hash="deadbeef" * 8,
    )
    db_session.add_all([message, transition, audit])
    db_session.flush()

    fetched = db_session.get(Ticket, ticket.id)
    assert fetched is not None
    assert fetched.status == "open"  # default, unaffected by the unrelated TicketStateTransition row above
