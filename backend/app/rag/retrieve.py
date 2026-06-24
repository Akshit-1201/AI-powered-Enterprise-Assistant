"""Retrieval: top-k chunks for a question, scoped to the user (D5)."""
import logging

from app.rag import embeddings, store

logger = logging.getLogger(__name__)

TOP_K = 4  # D5


def retrieve_context(user_id: str, question: str, k: int = TOP_K):
    """Return (chunk_texts, sources) for the user's documents.

    Empty lists when the user has no matching documents OR when the vector store is
    unavailable — the caller answers without grounding rather than crashing. Embedding /
    missing-key errors are left to propagate so they surface as a clean 503 upstream.
    """
    query_vec = embeddings.embed_query(question)
    try:
        result = store.query(query_vec, k, user_id)
    except Exception:
        logger.exception("Vector store query failed; degrading to no retrieved context")
        return [], []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    sources = [
        {"filename": m.get("filename"), "chunk_index": m.get("chunk_index")}
        for m in metadatas
    ]
    return documents, sources
