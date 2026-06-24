"""OpenAI embeddings (D1: text-embedding-3-small), lazily constructed.

Built on first use so the app/health start without a key. Callers reference the module
functions (embed_texts/embed_query) so tests can monkeypatch them.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.graph.llm import RETRYABLE_ERRORS, LLMNotConfigured


@lru_cache
def _client() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMNotConfigured(
            "OPENAI_API_KEY is not set - document upload and RAG retrieval require it."
        )
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


# Retry only transient provider errors; LLMNotConfigured (no key) propagates immediately.
_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
)


@_retry
def embed_texts(texts: list[str]) -> list[list[float]]:
    return _client().embed_documents(list(texts))


@_retry
def embed_query(text: str) -> list[float]:
    return _client().embed_query(text)
