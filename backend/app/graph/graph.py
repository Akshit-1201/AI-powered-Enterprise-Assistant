"""Assemble and run the LangGraph orchestration spine.

The SQLite checkpointer keyed by ``thread_id = "{user_id}:{session_id}"`` IS the
conversation memory (plan D7). The graph and its checkpointer connection are built once
and reused.
"""
import sqlite3

from langchain_core.messages import HumanMessage
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
    global _graph, _conn
    if _graph is None:
        settings = get_settings()
        _conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        _graph = _build(SqliteSaver(_conn))
    return _graph


def run_ask(question: str, user_id: str, session_id: str) -> dict:
    """Run one /ask turn through the graph and return the final state dict."""
    set_current_user_id(user_id)  # context seam; tools read this
    graph = get_graph()
    config = {
        "configurable": {"thread_id": f"{user_id}:{session_id}"},
        "recursion_limit": 10,
    }
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "session_id": session_id,
            "intent": "",
            "blocked": False,
            "context": [],
            "tool_used": None,
            "sources": [],
            "answer": "",
        },
        config=config,
    )
