"""Schemas for saved chat history (list + replay)."""
import datetime as dt

from pydantic import BaseModel, ConfigDict


class ConversationOut(BaseModel):
    """One row in the user's chat list."""

    model_config = ConfigDict(from_attributes=True)

    session_id: str
    title: str | None
    created_at: dt.datetime
    last_active: dt.datetime


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    text: str


class ConversationDetail(BaseModel):
    session_id: str
    title: str | None
    messages: list[ChatMessage]
