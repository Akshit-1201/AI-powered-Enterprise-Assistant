"""Offline graph-flow tests for the /ask spine (no API key needed).

The LLM is replaced with a deterministic stub so we can validate the *wiring*:
intent routing, tool-gating (D8), tool execution + tool_used surfacing, answer
generation, conversation memory, session_id minting (D6), and ConversationMeta
upsert (D7). LLM classification *quality* is out of scope here (needs a real key).
"""
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.graph.nodes as nodes
from app.db.database import SessionLocal, engine
from app.db.models import ConversationMeta, Ticket
from app.graph.graph import get_graph
from app.main import app
from app.rag import embeddings as rag_embeddings

from fastapi.testclient import TestClient


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


class StubLLM:
    """Mimics ChatOpenAI's invoke/bind_tools surface, deterministically."""

    def __init__(self):
        self.bind_tools_calls = []

    def bind_tools(self, tools):
        self.bind_tools_calls.append([t.name for t in tools])
        return self  # invoke() behaves the same; binding only matters for tracking

    def invoke(self, messages):
        system = messages[0].content.lower() if messages else ""
        if "scope filter" in system:  # guardrail classifier
            return AIMessage(content="ALLOW")
        if "intent router" in system:
            return AIMessage(content=self._route(_last_human(messages)))
        return self._agent(messages)

    @staticmethod
    def _route(question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["ticket", "look up", "employee", "customer", "report"]) or re.search(r"\b[ec]\d{3,}\b", q):
            return "action"
        first = q.strip().split()[:1]
        if first and first[0] in {"hi", "hello", "hey", "thanks", "thank"}:
            return "general"
        return "knowledge"

    def _agent(self, messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            return AIMessage(content=f"Result: {tool_msgs[-1].content}")
        q = _last_human(messages)
        ql = q.lower()
        emp = re.search(r"\b(e\d{3,})\b", ql)
        cust = re.search(r"\b(c\d{3,})\b", ql)
        if "ticket" in ql:
            return AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {"title": q[:60], "description": q}, "id": "c1", "type": "tool_call"}])
        if emp:
            return AIMessage(content="", tool_calls=[{"name": "fetch_employee_info", "args": {"employee_id": emp.group(1).upper()}, "id": "c2", "type": "tool_call"}])
        if cust:
            return AIMessage(content="", tool_calls=[{"name": "fetch_customer_info", "args": {"customer_id": cust.group(1).upper()}, "id": "c3", "type": "tool_call"}])
        if "report" in ql:
            return AIMessage(content="", tool_calls=[{"name": "generate_report", "args": {"topic": "summary"}, "id": "c4", "type": "tool_call"}])
        return AIMessage(content="A direct answer (no tool).")


@pytest.fixture
def stub(monkeypatch):
    s = StubLLM()
    monkeypatch.setattr(nodes, "get_chat_llm", lambda: s)
    # The knowledge path now runs Retrieve, which needs embeddings; stub them so the
    # graph runs offline (no docs uploaded here, so retrieval returns empty).
    monkeypatch.setattr(rag_embeddings, "embed_query", lambda t: [0.0] * 8)
    monkeypatch.setattr(rag_embeddings, "embed_texts", lambda texts: [[0.0] * 8 for _ in texts])
    return s


@pytest.fixture
def client():
    return TestClient(app)


def test_action_creates_ticket(stub, client, default_user):
    r = client.post("/ask", json={"question": "Create a ticket: VPN keeps disconnecting for finance."})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "action"
    assert body["tool_used"] == "create_ticket"
    assert body["answer"]
    assert body["session_id"]  # D6: minted
    assert stub.bind_tools_calls  # D8: tools bound on the action path

    db = SessionLocal()
    try:
        assert db.query(Ticket).filter_by(user_id=str(default_user.id)).count() >= 1
    finally:
        db.close()


def test_knowledge_binds_no_tools(stub, client):
    r = client.post("/ask", json={"question": "What is the capital of France?"})
    body = r.json()
    assert body["intent"] == "knowledge"
    assert body["tool_used"] is None
    assert stub.bind_tools_calls == []  # D8: no tools off the action path
    assert body["answer"]


def test_general_greeting(stub, client):
    r = client.post("/ask", json={"question": "hello there"})
    assert r.json()["intent"] == "general"


def test_employee_lookup_tool(stub, client):
    r = client.post("/ask", json={"question": "Look up employee E1001"})
    body = r.json()
    assert body["tool_used"] == "fetch_employee_info"
    assert "Priya Sharma" in body["answer"]


def test_memory_accumulates_across_turns(stub, client, default_user):
    r1 = client.post("/ask", json={"question": "hello"})
    sid = r1.json()["session_id"]
    client.post("/ask", json={"question": "and again", "session_id": sid})
    state = get_graph().get_state({"configurable": {"thread_id": f"{default_user.id}:{sid}"}})
    assert len(state.values.get("messages", [])) >= 4  # 2 turns persisted


def test_provided_session_id_is_echoed(stub, client):
    r = client.post("/ask", json={"question": "hi", "session_id": "fixed-123"})
    assert r.json()["session_id"] == "fixed-123"


def test_conversation_meta_recorded(stub, client, default_user):
    client.post("/ask", json={"question": "hello", "session_id": "meta-xyz"})
    db = SessionLocal()
    try:
        # Composite PK is (user_id, session_id).
        assert db.get(ConversationMeta, (str(default_user.id), "meta-xyz")) is not None
    finally:
        db.close()


def test_checkpointer_tables_exist(stub, client):
    from sqlalchemy import inspect

    client.post("/ask", json={"question": "hello"})
    tables = inspect(engine).get_table_names()
    assert any("checkpoint" in t for t in tables)


@pytest.mark.realauth
def test_memory_is_isolated_per_user_even_with_shared_session_id(stub, client):
    """Two users using the SAME session_id must not see each other's history
    (thread_id is keyed by user_id:session_id)."""
    from app.db.models import User

    def headers(email):
        client.post("/auth/register", json={"email": email, "password": "password123"})
        token = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    ha, hb = headers("mem_a@example.com"), headers("mem_b@example.com")
    client.post("/ask", json={"question": "remember the word apple", "session_id": "shared"}, headers=ha)
    client.post("/ask", json={"question": "remember the word banana", "session_id": "shared"}, headers=hb)

    db = SessionLocal()
    try:
        uid_a = db.query(User).filter_by(email="mem_a@example.com").first().id
        uid_b = db.query(User).filter_by(email="mem_b@example.com").first().id
    finally:
        db.close()

    msgs_a = " ".join(m.content for m in get_graph().get_state({"configurable": {"thread_id": f"{uid_a}:shared"}}).values["messages"])
    msgs_b = " ".join(m.content for m in get_graph().get_state({"configurable": {"thread_id": f"{uid_b}:shared"}}).values["messages"])
    assert "apple" in msgs_a and "banana" not in msgs_a
    assert "banana" in msgs_b and "apple" not in msgs_b


@pytest.mark.realauth
def test_conversation_meta_is_per_user_for_shared_session_id(stub, client):
    """Two users reusing the SAME session_id must get two distinct, correctly-attributed
    index rows — not one colliding/mis-attributed row (P0.3)."""
    from app.db.models import User

    def headers(email):
        client.post("/auth/register", json={"email": email, "password": "password123"})
        token = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    ha, hb = headers("meta_a@example.com"), headers("meta_b@example.com")
    client.post("/ask", json={"question": "hello", "session_id": "dup"}, headers=ha)
    client.post("/ask", json={"question": "hello", "session_id": "dup"}, headers=hb)

    db = SessionLocal()
    try:
        uid_a = db.query(User).filter_by(email="meta_a@example.com").first().id
        uid_b = db.query(User).filter_by(email="meta_b@example.com").first().id
        rows = db.query(ConversationMeta).filter_by(session_id="dup").all()
        owners = {r.user_id for r in rows}
    finally:
        db.close()

    assert len(rows) == 2  # one row per (user, session), not a single colliding row
    assert owners == {str(uid_a), str(uid_b)}
