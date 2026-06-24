"""Document ingestion: extract text -> chunk -> embed -> store in Chroma (D2/D4)."""
import io
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag import embeddings, store

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHUNK_SIZE = 1000  # D4
CHUNK_OVERLAP = 150  # D4


class UnsupportedFileType(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFileType(ext)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [c for c in splitter.split_text(text or "") if c.strip()]


def index_chunks(document_id: int, user_id: str, filename: str, chunks: list[str]) -> None:
    """Embed chunks and add them to Chroma, tagged for per-user retrieval."""
    vectors = embeddings.embed_texts(chunks)
    ids = [f"{document_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "user_id": user_id,
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    store.add_chunks(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
