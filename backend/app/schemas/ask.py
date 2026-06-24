"""Request/response models for POST /ask."""
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Structural upper bound at the API boundary: stops absurd payloads cheaply with a 422.
# Deliberately higher than the guardrail's content cap (guardrail.MAX_CHARS) so the band
# between them reaches the graph and gets a friendly templated "please shorten" answer
# (200) instead of a raw 422 — the Phase 4 "content vs. structural" split.
MAX_QUESTION_CHARS = 16000


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_CHARS, description="The user's question or instruction."
    )
    session_id: Optional[str] = Field(
        None, max_length=128, description="Thread id for conversation memory. Server mints one if omitted (D6)."
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AskResponse(BaseModel):
    answer: str
    intent: str = Field(
        ...,
        description="Routed intent: action | knowledge | general, or 'blocked' when the guardrail short-circuits the request.",
    )
    tool_used: Optional[str] = Field(None, description="Business tool invoked, if any.")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved sources (populated from Phase 2)."
    )
    session_id: str = Field(..., description="Thread id to reuse for follow-up turns.")
