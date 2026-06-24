"""Chroma vector store access.

Chunks are tagged with ``user_id`` metadata; the Retrieve node filters by it so a user
only ever sees their own documents (the multi-tenancy invariant; enforced for real once
auth lands in Phase 3). Embeddings are always supplied explicitly, so Chroma's default
embedding function is never invoked.
"""
import chromadb

from app.config import get_settings

COLLECTION = "documents"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=get_settings().chroma_path)
    return _client


def get_collection():
    return _get_client().get_or_create_collection(COLLECTION)


def reset_collection():
    """Drop and recreate the collection (used by tests for isolation)."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION)


def add_chunks(ids, embeddings, documents, metadatas) -> None:
    get_collection().add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def query(query_embedding, k: int, user_id: str):
    return get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"user_id": user_id},
    )


def delete_chunks(document_id: int, user_id: str) -> None:
    """Delete one document's chunk vectors, scoped to the owner (the user_id guard is
    defense-in-depth on top of the globally-unique document_id)."""
    get_collection().delete(
        where={"$and": [{"document_id": document_id}, {"user_id": user_id}]}
    )
