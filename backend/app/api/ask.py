"""POST /ask — run a question through the LangGraph spine.

Phase 1: identity is the placeholder user (config). Phase 3 replaces this with the
authenticated user resolved from the JWT, via the same context seam.
"""
import logging
from uuid import uuid4

import openai
from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user
from app.db.database import SessionLocal
from app.db.models import ConversationMeta, User, utcnow
from app.graph.graph import run_ask
from app.graph.llm import LLMNotConfigured
from app.schemas.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _touch_conversation_meta(session_id: str, user_id: str) -> None:
    """Upsert the session index (D7: metadata only, not conversation state)."""
    db = SessionLocal()
    try:
        meta = db.get(ConversationMeta, session_id)
        if meta is None:
            db.add(ConversationMeta(session_id=session_id, user_id=user_id))
        else:
            meta.last_active = utcnow()
        db.commit()
    finally:
        db.close()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, current_user: User = Depends(get_current_user)) -> AskResponse:
    # D6: mint a thread id when the client doesn't supply one.
    session_id = request.session_id or f"sess-{uuid4().hex[:12]}"
    user_id = str(current_user.id)  # the seam: real user scopes memory + RAG now

    try:
        result = run_ask(request.question, user_id, session_id)
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except openai.APIError as exc:
        logger.exception("LLM/provider call failed during /ask")
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again.",
        ) from exc

    _touch_conversation_meta(session_id, user_id)

    return AskResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "general"),
        tool_used=result.get("tool_used"),
        sources=result.get("sources", []),
        session_id=session_id,
    )
