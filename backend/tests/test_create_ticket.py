"""Plan test (c): create_ticket happy path.

Exercises the tool directly (no LLM/API key needed): it should persist a ticket scoped
to the current user and report the new id.
"""
from app.context import set_current_user_id
from app.db.database import SessionLocal, init_db
from app.db.models import Ticket
from app.tools.business import create_ticket


def test_create_ticket_persists_and_is_user_scoped():
    init_db()
    set_current_user_id("test-user")

    result = create_ticket.invoke(
        {"title": "VPN keeps disconnecting", "description": "Finance team affected"}
    )

    assert "created" in result.lower()

    db = SessionLocal()
    try:
        tickets = db.query(Ticket).filter_by(user_id="test-user").all()
        assert len(tickets) == 1
        assert tickets[0].title == "VPN keeps disconnecting"
        assert tickets[0].status == "open"
    finally:
        db.close()
