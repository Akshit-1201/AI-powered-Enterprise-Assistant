"""Saved chat history: list a user's conversations and replay one to continue it.

The message history itself lives in the LangGraph checkpointer (keyed by user_id:session_id);
``ConversationMeta`` is the per-user index that makes the chat list possible. Both are
strictly user-scoped, so one user can never list or open another's chats.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.db.database import SessionLocal
from app.db.models import ConversationMeta, User
from app.graph.graph import delete_conversation_state, get_conversation_messages
from app.schemas.conversations import ConversationDetail, ConversationOut

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(current_user: User = Depends(get_current_user)) -> list[ConversationOut]:
    user_id = str(current_user.id)
    db = SessionLocal()
    try:
        rows = (
            db.query(ConversationMeta)
            .filter_by(user_id=user_id)
            .order_by(ConversationMeta.last_active.desc())
            .all()
        )
        return [ConversationOut.model_validate(r) for r in rows]
    finally:
        db.close()


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
def get_conversation(session_id: str, current_user: User = Depends(get_current_user)) -> ConversationDetail:
    user_id = str(current_user.id)
    db = SessionLocal()
    try:
        meta = db.get(ConversationMeta, (user_id, session_id))
        if meta is None:  # not the user's session (or doesn't exist) -> 404, no existence leak
            raise HTTPException(status_code=404, detail="Conversation not found.")
        title = meta.title
    finally:
        db.close()
    messages = get_conversation_messages(user_id, session_id)
    return ConversationDetail(session_id=session_id, title=title, messages=messages)


@router.delete("/conversations/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(session_id: str, current_user: User = Depends(get_current_user)) -> None:
    """Delete one of the current user's chats: its checkpointer state (the messages) and
    its index row. 404 if it doesn't exist or belongs to another user."""
    user_id = str(current_user.id)
    db = SessionLocal()
    try:
        meta = db.get(ConversationMeta, (user_id, session_id))
        if meta is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        # Clear the persisted messages first so a deleted chat can never resurface, then
        # drop the index row.
        delete_conversation_state(user_id, session_id)
        db.delete(meta)
        db.commit()
    finally:
        db.close()
