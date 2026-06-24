"""Robustness tests: the service degrades gracefully instead of crashing or 500ing.

(The Phase 4 content-guardrail node, retry/backoff, and templated fallback *answers* are
separate; these cover defensive handling of the existing Phase 1/2 code.)
"""
import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.graph.nodes as nodes
from app.graph import llm as llm_module
from app.graph.llm import LLMNotConfigured
from app.main import app
from app.rag import embeddings as rag_embeddings
from app.rag import retrieve as rag_retrieve
from app.rag import store as rag_store

from fastapi.testclient import TestClient


def _timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/x"))


@pytest.fixture
def client():
    return TestClient(app)


# ---- input boundary hardening ---------------------------------------------

def test_blank_question_rejected(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_oversized_question_rejected(client):
    # Above the structural Pydantic cap -> clean 422 at the boundary.
    from app.schemas.ask import MAX_QUESTION_CHARS

    assert client.post("/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)}).status_code == 422


def test_overlong_session_id_rejected(client):
    assert client.post("/ask", json={"question": "hi", "session_id": "s" * 200}).status_code == 422


# ---- tool failure does not crash the graph --------------------------------

def test_tool_failure_is_graceful(monkeypatch):
    class _Boom:
        name = "boom"

        def invoke(self, _args):
            raise RuntimeError("kaboom")

    monkeypatch.setitem(nodes.TOOLS_BY_NAME, "boom", _Boom())
    ai = AIMessage(content="", tool_calls=[{"name": "boom", "args": {}, "id": "x", "type": "tool_call"}])
    out = nodes.tool_node({"messages": [ai], "tool_used": None})
    content = out["messages"][0].content
    assert "could not be completed" in content  # error captured, no raise
    assert "kaboom" not in content  # raw exception text never leaks into model context (P1.7)


# ---- retrieval degrades when the vector store is down ----------------------

def test_retrieve_degrades_on_store_failure(monkeypatch):
    monkeypatch.setattr(rag_embeddings, "embed_query", lambda t: [0.0] * 8)

    def _down(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(rag_store, "query", _down)
    docs, sources = rag_retrieve.retrieve_context("u", "q")
    assert docs == [] and sources == []


# ---- provider outage -> clean 503, not a raw 500 --------------------------

def test_provider_error_returns_clean_503(monkeypatch, client):
    class _Flaky:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            raise _timeout_error()

    monkeypatch.setattr(nodes, "get_chat_llm", lambda: _Flaky())
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 503
    assert "temporarily unavailable" in r.json()["detail"]


# ---- retry / backoff (D10) -------------------------------------------------

def test_invoke_with_retry_retries_transient_then_succeeds():
    calls = {"n": 0}

    class _Flaky:
        def invoke(self, _messages):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _timeout_error()
            return AIMessage(content="ok")

    result = llm_module.invoke_with_retry(_Flaky(), [])
    assert result.content == "ok"
    assert calls["n"] == 3  # retried twice, succeeded on the third


def test_invoke_with_retry_does_not_retry_non_transient():
    calls = {"n": 0}

    class _Boom:
        def invoke(self, _messages):
            calls["n"] += 1
            raise ValueError("not a transient error")

    with pytest.raises(ValueError):
        llm_module.invoke_with_retry(_Boom(), [])
    assert calls["n"] == 1  # no retry on non-transient errors


def test_embed_query_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _FakeClient:
        def embed_query(self, _text):
            calls["n"] += 1
            if calls["n"] < 2:
                raise _timeout_error()
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(rag_embeddings, "_client", lambda: _FakeClient())
    assert rag_embeddings.embed_query("hi") == [0.1, 0.2, 0.3]
    assert calls["n"] == 2


def test_embed_query_does_not_retry_missing_key(monkeypatch):
    calls = {"n": 0}

    def _no_key():
        calls["n"] += 1
        raise LLMNotConfigured("no key")

    monkeypatch.setattr(rag_embeddings, "_client", _no_key)
    with pytest.raises(LLMNotConfigured):
        rag_embeddings.embed_query("hi")
    assert calls["n"] == 1  # missing key is not transient -> no retry


# ---- guardrail fails open if its LLM check errors --------------------------

def test_guardrail_fails_open_on_llm_error(monkeypatch):
    class _Boom:
        def invoke(self, _messages):
            raise RuntimeError("guardrail llm down")

    monkeypatch.setattr(nodes, "get_chat_llm", lambda: _Boom())
    out = nodes.guardrail_node({"messages": [HumanMessage(content="a normal question")]})
    assert out["blocked"] is False  # never block legitimate input on a flaky check


# ---- multi-tenancy seam fails loud, never silently (P0.4) ------------------

def test_current_user_id_raises_when_unset():
    import contextvars

    from app.context import get_current_user_id

    # A fresh context never had the id set -> fail loud rather than return a phantom default.
    with pytest.raises(RuntimeError):
        contextvars.Context().run(get_current_user_id)


def test_current_user_id_returns_after_set():
    import contextvars

    from app.context import get_current_user_id, set_current_user_id

    def _inside():
        set_current_user_id("u42")
        return get_current_user_id()

    assert contextvars.Context().run(_inside) == "u42"
