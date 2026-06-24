"""Lazy LLM accessor + retry helper.

Built on first use (not at import) so the app and /health work without an API key.
Nodes that need the model call get_chat_llm(); /ask surfaces a clean error if the key
is missing. invoke_with_retry adds bounded retry/backoff on transient provider errors (D10).
"""
from functools import lru_cache

import openai
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

# Transient errors worth retrying. NOT BadRequest/Authentication (those won't self-heal).
RETRYABLE_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
)


class LLMNotConfigured(RuntimeError):
    """Raised when an LLM call is attempted without OPENAI_API_KEY set."""


@lru_cache
def get_chat_llm() -> ChatOpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMNotConfigured(
            "OPENAI_API_KEY is not set - /ask requires it. Copy .env.example to .env and add a key."
        )
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.openai_api_key,
        timeout=settings.request_timeout,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
)
def invoke_with_retry(llm, messages):
    """Invoke a chat model (or tool-bound model) with retry/backoff on transient errors."""
    return llm.invoke(messages)
