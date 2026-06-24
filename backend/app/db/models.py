"""Relational models.

Phase 1 owns Ticket and ConversationMeta. User (Phase 3) and Document (Phase 2) are
added in their phases.

Note (plan D7): ConversationMeta is a lightweight *index* of sessions — it does NOT
store conversation state. The LangGraph SQLite checkpointer owns the actual message
history. Keep them separate; do not duplicate state here.
"""
import datetime as dt

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.database import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="open", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class ConversationMeta(Base):
    __tablename__ = "conversation_meta"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    last_active = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


class Document(Base):
    """Index of uploaded documents. The chunk vectors live in Chroma (tagged with
    user_id + document_id); this row is the relational record for listing/auditing."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    uploaded_at = Column(DateTime, default=utcnow, nullable=False)
