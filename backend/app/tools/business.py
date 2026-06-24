"""Business-action tools (spec §7).

Each is a LangChain ``@tool`` so its schema can be bound to the LLM. User scoping comes
from ``get_current_user_id()`` (the context seam), never from an LLM-supplied argument —
the model must not be able to set the user id.
"""
import json
from pathlib import Path

from langchain_core.tools import tool

from app.config import DATA_DIR
from app.context import get_current_user_id
from app.db.database import SessionLocal
from app.db.models import Ticket


def _load(filename: str) -> list[dict]:
    path: Path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def create_ticket(title: str, description: str = "") -> str:
    """Create and persist a support ticket for the current user.

    Use when the user reports a problem or asks to open/create/log/file a ticket.
    Returns the new ticket id and status.
    """
    user_id = get_current_user_id()
    db = SessionLocal()
    try:
        ticket = Ticket(user_id=user_id, title=title, description=description, status="open")
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        return f"Ticket #{ticket.id} created with status '{ticket.status}': {title}."
    finally:
        db.close()


@tool
def fetch_employee_info(employee_id: str) -> str:
    """Look up an employee by id (e.g. 'E1001').

    Returns name, department, role, remaining leave balance, and manager.
    """
    for e in _load("employees.json"):
        if e["id"].lower() == employee_id.strip().lower():
            return (
                f"Employee {e['id']} ({e['name']}): department={e['department']}, "
                f"role={e['role']}, leave_balance={e['leave_balance']} days, "
                f"manager={e['manager']}."
            )
    return f"No employee found with id '{employee_id}'."


@tool
def fetch_customer_info(customer_id: str) -> str:
    """Look up a customer by id (e.g. 'C2001').

    Returns name, tier, account status, and number of open issues.
    """
    for c in _load("customers.json"):
        if c["id"].lower() == customer_id.strip().lower():
            return (
                f"Customer {c['id']} ({c['name']}): tier={c['tier']}, "
                f"account_status={c['account_status']}, open_issues={c['open_issues']}."
            )
    return f"No customer found with id '{customer_id}'."


@tool
def generate_report(topic: str = "summary") -> str:
    """Generate a short summary report aggregating available data.

    Covers employee and customer counts and the current user's tickets. Use when the
    user asks for a report, summary, or overview.
    """
    employees = _load("employees.json")
    customers = _load("customers.json")
    user_id = get_current_user_id()
    db = SessionLocal()
    try:
        total_tickets = db.query(Ticket).filter_by(user_id=user_id).count()
        open_tickets = db.query(Ticket).filter_by(user_id=user_id, status="open").count()
    finally:
        db.close()
    return (
        f"Report ({topic}): {len(employees)} employees, {len(customers)} customers, "
        f"{total_tickets} tickets for you ({open_tickets} open)."
    )


ALL_TOOLS = [create_ticket, fetch_employee_info, fetch_customer_info, generate_report]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
