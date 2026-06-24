"""Assemble and run the LangGraph orchestration spine.

The SQLite checkpointer keyed by ``thread_id = "{user_id}:{session_id}"`` IS the
conversation memory (plan D7). The graph and its checkpointer connection are built once
and reused.
"""
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.context import set_current_user_id
from app.graph.nodes import (
    agent_node,
    generate_node,
    guardrail_node,
    retrieve_node,
    route_after_agent,
    route_after_guardrail,
    route_after_router,
    router_node,
    tool_node,
)
from app.graph.state import GraphState

_graph = None
_conn: sqlite3.Connection | None = None
_saver: SqliteSaver | None = None


def _build(checkpointer) -> "object":
    builder = StateGraph(GraphState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "guardrail")
    # Blocked input short-circuits to END with a templated answer already set.
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail, {"blocked": END, "router": "router"}
    )
    # knowledge -> retrieve -> agent; action/general -> agent directly.
    builder.add_conditional_edges(
        "router", route_after_router, {"retrieve": "retrieve", "agent": "agent"}
    )
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "generate": "generate"}
    )
    builder.add_edge("tools", "agent")  # ReAct loop: re-enter agent after tool results
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


def get_graph():
    global _graph, _conn, _saver
    if _graph is None:
        settings = get_settings()
        _conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        _saver = SqliteSaver(_conn)
        _graph = _build(_saver)
    return _graph


def delete_conversation_state(user_id: str, session_id: str) -> None:
    """Delete a conversation's persisted checkpointer state (both the checkpoints and the
    pending writes for the thread), so a deleted chat leaves nothing behind."""
    get_graph()  # ensure the saver is built
    _saver.delete_thread(f"{user_id}:{session_id}")


def _initial_state(question: str, user_id: str, session_id: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "user_id": user_id,
        "session_id": session_id,
        "intent": "",
        "blocked": False,
        "context": [],
        "tool_used": None,
        "sources": [],
        "answer": "",
    }


def run_ask(question: str, user_id: str, session_id: str) -> dict:
    """Run one /ask turn through the graph and return the final state dict."""
    set_current_user_id(user_id)  # context seam; tools read this
    graph = get_graph()
    config = {
        "configurable": {"thread_id": f"{user_id}:{session_id}"},
        "recursion_limit": 10,
    }
    return graph.invoke(_initial_state(question, user_id, session_id), config=config)


def get_conversation_messages(user_id: str, session_id: str) -> list[dict]:
    """Replay a saved conversation from the checkpointer as a clean user/assistant
    transcript. Scoped by the thread_id, so it only ever returns this user's history.
    Tool-call turns and empty messages are dropped; what's left is the visible exchange."""
    snapshot = get_graph().get_state(
        {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    )
    messages = (snapshot.values or {}).get("messages", [])
    transcript: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            transcript.append({"role": "user", "text": str(message.content)})
        elif isinstance(message, AIMessage):
            text = message.content if isinstance(message.content, str) else str(message.content)
            if text.strip():  # skip the empty tool-call turn
                transcript.append({"role": "assistant", "text": text})
    return transcript


def run_ask_stream(question: str, user_id: str, session_id: str):
    """Stream one /ask turn. Yields:
        ("token", text) — one chunk of the visible answer, as it is generated.
        ("meta", {intent, tool_used, sources}) — once, after the answer completes.

    Only the *agent* node's text tokens are surfaced. The router and guardrail LLM outputs
    (e.g. the word "action") and the empty tool-call turn carry ``langgraph_node`` values
    other than "agent" or have no text, so they are filtered out and never leak into the
    visible answer. On a guardrail block there are no agent tokens, so the templated block
    message (from the ``updates`` stream) is emitted as the answer instead."""
    set_current_user_id(user_id)  # context seam; tools read this
    graph = get_graph()
    config = {
        "configurable": {"thread_id": f"{user_id}:{session_id}"},
        "recursion_limit": 10,
    }

    intent = "general"
    tool_used = None
    sources: list = []

    for mode, chunk in graph.stream(
        _initial_state(question, user_id, session_id),
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") == "agent":
                text = getattr(message_chunk, "content", "") or ""
                if text:
                    yield ("token", text)
        else:  # "updates": node state deltas — harvest the demo metadata
            for _node, update in chunk.items():
                if not update:
                    continue
                if update.get("intent"):
                    intent = update["intent"]
                if update.get("tool_used"):
                    tool_used = update["tool_used"]
                if update.get("sources"):
                    sources = update["sources"]
                if update.get("blocked"):
                    intent = "blocked"
                    if update.get("answer"):
                        yield ("token", update["answer"])  # no agent tokens on the blocked path

    yield ("meta", {"intent": intent, "tool_used": tool_used, "sources": sources})
