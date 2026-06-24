"""POST /ask — run a question through the LangGraph spine.

Phase 1: identity is the placeholder user (config). Phase 3 replaces this with the
authenticated user resolved from the JWT, via the same context seam.
"""
import json
import logging
from uuid import uuid4

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth.dependencies import get_current_user
from app.db.database import SessionLocal
from app.db.models import ConversationMeta, User, utcnow
from app.graph.graph import run_ask, run_ask_stream
from app.graph.llm import LLMNotConfigured
from app.schemas.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _touch_conversation_meta(session_id: str, user_id: str, first_question: str = "") -> None:
    """Upsert the per-(user, session) index (D7: metadata only, not conversation state).
    On first touch, the session is titled from the opening question for the chat list."""
    db = SessionLocal()
    try:
        meta = db.get(ConversationMeta, (user_id, session_id))  # composite PK order
        if meta is None:
            title = (first_question or "").strip()[:80] or None
            db.add(ConversationMeta(session_id=session_id, user_id=user_id, title=title))
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

    _touch_conversation_meta(session_id, user_id, request.question)

    return AskResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "general"),
        tool_used=result.get("tool_used"),
        sources=result.get("sources", []),
        session_id=session_id,
    )


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


@router.post("/ask/stream")
def ask_stream(request: AskRequest, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    """Token-by-token variant of /ask over SSE (the UI uses this; /ask stays the
    standalone, curl-gradeable endpoint). Event frames: {type:"token",text}, then
    {type:"meta", intent, tool_used, sources, session_id}, then {type:"done"};
    {type:"error", detail} on failure. Note: once streaming starts the HTTP status is
    fixed at 200, so a mid-stream provider failure surfaces as an `error` event, not a 503."""
    session_id = request.session_id or f"sess-{uuid4().hex[:12]}"
    user_id = str(current_user.id)

    def gen():
        try:
            for ev_type, payload in run_ask_stream(request.question, user_id, session_id):
                if ev_type == "token":
                    yield _sse({"type": "token", "text": payload})
                elif ev_type == "meta":
                    yield _sse({"type": "meta", "session_id": session_id, **payload})
            _touch_conversation_meta(session_id, user_id, request.question)
            yield _sse({"type": "done"})
        except LLMNotConfigured as exc:
            yield _sse({"type": "error", "detail": str(exc)})
        except openai.APIError:
            logger.exception("LLM/provider call failed during /ask/stream")
            yield _sse({"type": "error", "detail": "The AI service is temporarily unavailable. Please try again."})
        except Exception:
            logger.exception("Unexpected error during /ask/stream")
            yield _sse({"type": "error", "detail": "Internal server error."})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
