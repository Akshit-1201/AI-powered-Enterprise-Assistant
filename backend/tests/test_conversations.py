"""Saved chat history (/conversations) — list, replay, and per-user isolation.

The LLM is stubbed (reusing the graph-flow stub style) so turns run offline; what we
assert is that conversations are saved, titled, replayable, and never cross tenants.
"""
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.graph.nodes as nodes
from app.main import app
from app.rag import embeddings as rag_embeddings

from fastapi.testclient import TestClient


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


class StubLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system = messages[0].content.lower() if messages else ""
        if "scope filter" in system:
            return AIMessage(content="ALLOW")
        if "intent router" in system:
            q = _last_human(messages).lower()
            first = q.strip().split()[:1]
            if first and first[0] in {"hi", "hello", "hey"}:
                return AIMessage(content="general")
            return AIMessage(content="knowledge")
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="done")
        return AIMessage(content=f"Echo: {_last_human(messages)}")


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(nodes, "get_chat_llm", lambda: StubLLM())
    monkeypatch.setattr(rag_embeddings, "embed_query", lambda t: [0.0] * 8)
    monkeypatch.setattr(rag_embeddings, "embed_texts", lambda texts: [[0.0] * 8 for _ in texts])


@pytest.fixture
def client():
    return TestClient(app)


def test_list_starts_empty(stub, client):
    assert client.get("/conversations").json() == []


def test_conversation_is_listed_and_titled(stub, client):
    client.post("/ask", json={"question": "What is our leave policy?", "session_id": "c1"})
    rows = client.get("/conversations").json()
    assert any(r["session_id"] == "c1" for r in rows)
    row = next(r for r in rows if r["session_id"] == "c1")
    assert row["title"] == "What is our leave policy?"  # titled from the first question


def test_replay_returns_user_and_assistant_turns(stub, client):
    client.post("/ask", json={"question": "hello there", "session_id": "c2"})
    client.post("/ask", json={"question": "and a follow up", "session_id": "c2"})
    detail = client.get("/conversations/c2").json()
    roles = [m["role"] for m in detail["messages"]]
    texts = " ".join(m["text"] for m in detail["messages"])
    assert roles == ["user", "assistant", "user", "assistant"]  # two turns, in order
    assert "hello there" in texts and "and a follow up" in texts


def test_missing_conversation_is_404(stub, client):
    assert client.get("/conversations/does-not-exist").status_code == 404


def test_continuing_a_session_keeps_one_listing(stub, client):
    client.post("/ask", json={"question": "first", "session_id": "c3"})
    client.post("/ask", json={"question": "second", "session_id": "c3"})
    rows = [r for r in client.get("/conversations").json() if r["session_id"] == "c3"]
    assert len(rows) == 1  # same session continued, not duplicated


def test_delete_conversation_removes_listing_and_state(stub, client, default_user):
    client.post("/ask", json={"question": "to be deleted", "session_id": "d1"})
    assert client.get("/conversations/d1").status_code == 200

    assert client.delete("/conversations/d1").status_code == 204

    # Gone from the list and not replayable...
    assert client.get("/conversations/d1").status_code == 404
    assert all(c["session_id"] != "d1" for c in client.get("/conversations").json())
    # ...and the checkpointer state is cleared (no lingering messages).
    from app.graph.graph import get_conversation_messages

    assert get_conversation_messages(str(default_user.id), "d1") == []


def test_delete_missing_conversation_is_404(stub, client):
    assert client.delete("/conversations/does-not-exist").status_code == 404


@pytest.mark.realauth
def test_cannot_delete_another_users_conversation(stub, client):
    def headers(email):
        client.post("/auth/register", json={"email": email, "password": "password123"})
        token = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    ha, hb = headers("del_a@example.com"), headers("del_b@example.com")
    client.post("/ask", json={"question": "A's private chat", "session_id": "owned"}, headers=ha)

    # Intruder gets 404 (no cross-tenant delete, no existence leak)...
    assert client.delete("/conversations/owned", headers=hb).status_code == 404
    # ...and the owner's chat is untouched.
    assert client.get("/conversations/owned", headers=ha).status_code == 200


@pytest.mark.realauth
def test_chat_history_is_isolated_per_user(stub, client):
    def headers(email):
        client.post("/auth/register", json={"email": email, "password": "password123"})
        token = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    ha, hb = headers("chat_a@example.com"), headers("chat_b@example.com")
    client.post("/ask", json={"question": "user A secret topic", "session_id": "shared"}, headers=ha)
    client.post("/ask", json={"question": "user B secret topic", "session_id": "shared"}, headers=hb)

    # Each user lists only their own conversation...
    a_titles = [r["title"] for r in client.get("/conversations", headers=ha).json()]
    b_titles = [r["title"] for r in client.get("/conversations", headers=hb).json()]
    assert "user A secret topic" in a_titles and "user B secret topic" not in a_titles
    assert "user B secret topic" in b_titles and "user A secret topic" not in b_titles

    # ...and replaying the shared session id returns only the caller's own turns.
    a_detail = " ".join(m["text"] for m in client.get("/conversations/shared", headers=ha).json()["messages"])
    assert "user A secret topic" in a_detail and "user B secret topic" not in a_detail
