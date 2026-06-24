"""Graph state.

Extends LangGraph's MessagesState (which provides ``messages`` with the add_messages
reducer, so turns accumulate across a thread via the checkpointer = conversation memory).
"""
from typing import Any, Optional

from langgraph.graph import MessagesState


class GraphState(MessagesState):
    user_id: str
    session_id: str
    intent: str
    blocked: bool  # set by the guardrail node to short-circuit to END
    context: list[str]  # retrieved document chunks (knowledge path); ephemeral per turn
    tool_used: Optional[str]
    sources: list[dict[str, Any]]
    answer: str
