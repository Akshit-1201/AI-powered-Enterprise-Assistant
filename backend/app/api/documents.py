"""Document upload + listing.

Phase 2: identity is the placeholder user (config). Phase 3 swaps in the authenticated
user via the same context seam, which also makes the per-user Chroma scoping real.
"""
import logging
from pathlib import Path

import openai
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.auth.dependencies import get_current_user
from app.db.database import SessionLocal
from app.db.models import Document, User
from app.graph.llm import LLMNotConfigured
from app.rag import store
from app.rag.ingest import ALLOWED_EXTENSIONS, UnsupportedFileType, chunk_text, extract_text, index_chunks
from app.schemas.documents import DocumentOut, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

MAX_BYTES = 10 * 1024 * 1024  # D3: 10 MB
_READ_CHUNK = 1024 * 1024  # 1 MB read granularity


def _read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in bounded chunks, aborting with 413 the moment it exceeds the cap.

    Never buffers an unbounded body into memory: a malicious multi-GB upload is rejected
    after reading at most ~max_bytes, not after fully reading it (P0.2)."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
        parts.append(chunk)
    return b"".join(parts)


@router.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> UploadResponse:
    user_id = str(current_user.id)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: .pdf, .txt, .md.",
        )

    data = _read_capped(file, MAX_BYTES)  # bounded read; 413 before buffering an oversized body
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_text(file.filename, data)
    except UnsupportedFileType:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'.")
    except Exception as exc:  # corrupt/unreadable file
        raise HTTPException(status_code=400, detail=f"Could not read document: {exc}") from exc

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in the document.")

    db = SessionLocal()
    try:
        doc = Document(user_id=user_id, filename=file.filename, chunk_count=len(chunks))
        db.add(doc)
        db.flush()  # assign doc.id without committing
        try:
            index_chunks(doc.id, user_id, file.filename, chunks)
        except LLMNotConfigured as exc:
            db.rollback()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except openai.APIError as exc:
            db.rollback()
            logger.exception("Embedding/provider call failed during upload")
            raise HTTPException(
                status_code=503,
                detail="The embedding service is temporarily unavailable. Please try again.",
            ) from exc
        db.commit()
        db.refresh(doc)
        return UploadResponse(id=doc.id, filename=doc.filename, chunk_count=doc.chunk_count)
    finally:
        db.close()


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(current_user: User = Depends(get_current_user)) -> list[DocumentOut]:
    user_id = str(current_user.id)
    db = SessionLocal()
    try:
        docs = (
            db.query(Document)
            .filter_by(user_id=user_id)
            .order_by(Document.uploaded_at.desc())
            .all()
        )
        return [DocumentOut.model_validate(d) for d in docs]
    finally:
        db.close()


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, current_user: User = Depends(get_current_user)) -> None:
    """Delete one of the current user's documents and its chunk vectors. 404 if it doesn't
    exist or belongs to another user (no cross-tenant deletion, no existence leak)."""
    user_id = str(current_user.id)
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None or doc.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found.")
        # Remove the vectors first so a deleted document can never resurface in retrieval,
        # then drop the relational row.
        store.delete_chunks(document_id, user_id)
        db.delete(doc)
        db.commit()
    finally:
        db.close()
