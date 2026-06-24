"""Phase 2 RAG tests (no API key needed).

Embeddings are stubbed with a deterministic hashed bag-of-words so nearest-neighbour
ranking is meaningful offline. This validates the ingestion + retrieval wiring and the
upload/list endpoints, not real OpenAI embedding quality.
"""
import hashlib
import io
import math

import pytest
from langchain_core.messages import AIMessage, SystemMessage

import app.graph.nodes as nodes
from app.db.database import SessionLocal
from app.db.models import Document
from app.graph.llm import LLMNotConfigured
from app.rag import embeddings, store
from app.rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, extract_text, index_chunks
from app.rag.retrieve import retrieve_context

from fastapi.testclient import TestClient

DIM = 96


def _make_pdf(text: str) -> bytes:
    """Build a minimal valid single-page PDF with extractable text (no extra deps)."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    content = b"BT /F1 24 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
    objs.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    n = len(objs) + 1
    out += b"xref\n0 " + str(n).encode() + b"\n0000000000 65535 f \n"
    for o in offsets:
        out += ("%010d 00000 n \n" % o).encode()
    out += b"trailer\n<< /Size " + str(n).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF"
    return out


def _fake_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def _isolate_store():
    store.reset_collection()
    yield


@pytest.fixture
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [_fake_embed(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", lambda t: _fake_embed(t))


class _RagStubLLM:
    """Routes everything to knowledge and answers from injected context."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system = messages[0].content.lower() if messages else ""
        if "scope filter" in system:  # guardrail classifier
            return AIMessage(content="ALLOW")
        if "intent router" in system:
            return AIMessage(content="knowledge")
        ctx = next(
            (m.content for m in messages if isinstance(m, SystemMessage) and "uploaded documents" in m.content.lower()),
            "",
        )
        return AIMessage(content=f"From docs: {ctx}" if ctx else "No relevant documents found.")


@pytest.fixture
def stub_llm(monkeypatch):
    monkeypatch.setattr(nodes, "get_chat_llm", lambda: _RagStubLLM())


@pytest.fixture
def client():
    return TestClient(__import__("app.main", fromlist=["app"]).app)


# ---- extraction / chunking (no embeddings) --------------------------------

def test_extract_txt():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_chunking_splits_long_text():
    chunks = chunk_text("word " * 1000)  # ~5000 chars
    assert len(chunks) > 1


def test_unsupported_extension_raises():
    from app.rag.ingest import UnsupportedFileType

    with pytest.raises(UnsupportedFileType):
        extract_text("malware.exe", b"\x00\x01")


# ---- retrieval ranking ----------------------------------------------------

def test_retrieve_returns_relevant_chunk(stub_embeddings):
    index_chunks(1, "u1", "doc.txt", [
        "The Project Phoenix budget is 2 million dollars for fiscal year 2026.",
        "The office cafeteria serves lunch from noon until two.",
    ])
    docs, sources = retrieve_context("u1", "What is the Project Phoenix budget?")
    assert docs
    assert "Phoenix budget" in docs[0]
    assert sources[0]["filename"] == "doc.txt"


def test_retrieval_is_user_scoped(stub_embeddings):
    index_chunks(7, "owner", "secret.txt", ["The launch code is alpha-tango-9."])
    docs, sources = retrieve_context("intruder", "What is the launch code?")
    assert docs == []  # other user's chunks are not visible (multi-tenancy invariant)


# ---- endpoints ------------------------------------------------------------

def test_upload_then_list(stub_embeddings, client):
    r = client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"Remote work is allowed three days per week.", "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunk_count"] >= 1

    listing = client.get("/documents").json()
    assert any(d["filename"] == "policy.txt" for d in listing)


def test_upload_rejects_unsupported_type(stub_embeddings, client):
    r = client.post(
        "/documents/upload",
        files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert r.status_code == 415


def test_upload_rejects_empty_file(stub_embeddings, client):
    r = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 400


def test_ask_knowledge_grounds_with_sources(stub_embeddings, stub_llm, client):
    client.post(
        "/documents/upload",
        files={"file": ("hr.txt", b"Employees accrue 20 vacation days per year under the leave policy.", "text/plain")},
    )
    r = client.post("/ask", json={"question": "How many vacation days do employees get?"})
    body = r.json()
    assert r.status_code == 200
    assert body["intent"] == "knowledge"
    assert body["sources"]  # grounded: retrieval found the chunk
    assert "vacation days" in body["answer"]


def test_ask_knowledge_empty_store_is_graceful(stub_embeddings, stub_llm, client):
    r = client.post("/ask", json={"question": "What is our parental leave policy?"})
    body = r.json()
    assert r.status_code == 200  # no crash with an empty store
    assert body["intent"] == "knowledge"
    assert body["sources"] == []


# ---- extraction: PDF + markdown -------------------------------------------

def test_extract_pdf():
    text = extract_text("doc.pdf", _make_pdf("Quarterly revenue grew by fifteen percent"))
    assert "Quarterly revenue" in text


def test_upload_pdf_end_to_end(stub_embeddings, client):
    r = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", _make_pdf("Annual report figures for review"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["chunk_count"] >= 1


def test_upload_markdown(stub_embeddings, client):
    r = client.post(
        "/documents/upload",
        files={"file": ("notes.md", b"# Heading\n\nMarkdown body text.", "text/markdown")},
    )
    assert r.status_code == 200, r.text


# ---- chunking parameters (D4) ---------------------------------------------

def test_chunk_params_respected():
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 150)
    chunks = chunk_text("sentence number %d. " % 0 + "lorem ipsum dolor sit amet " * 400)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert len(chunks) > 1


# ---- size limit (D3) ------------------------------------------------------

def test_upload_rejects_oversized_file(client):
    big = b"a" * (10 * 1024 * 1024 + 1)
    r = client.post("/documents/upload", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413


# ---- Chroma metadata tagging ----------------------------------------------

def test_chunks_tagged_with_metadata(stub_embeddings):
    index_chunks(42, "alice", "spec.txt", ["chunk one", "chunk two", "chunk three"])
    got = store.get_collection().get(where={"document_id": 42})
    assert sorted(got["ids"]) == ["42-0", "42-1", "42-2"]
    for meta in got["metadatas"]:
        assert meta["user_id"] == "alice"
        assert meta["document_id"] == 42
        assert meta["filename"] == "spec.txt"
        assert isinstance(meta["chunk_index"], int)


# ---- top-k cap (D5) -------------------------------------------------------

def test_retrieval_caps_at_top_k(stub_embeddings):
    index_chunks(5, "bob", "many.txt", [f"shared topic line number {i}" for i in range(6)])
    docs, sources = retrieve_context("bob", "shared topic")
    assert len(docs) == 4  # top-k = 4
    assert len(sources) == 4


# ---- persistence across client restart ------------------------------------

def test_chroma_persists_across_client_reinit(stub_embeddings):
    index_chunks(99, "carol", "persist.txt", ["a durable fact about widgets"])
    store._client = None  # simulate process restart
    docs, _ = retrieve_context("carol", "widgets")
    assert docs and "widgets" in docs[0]


# ---- DB rollback when indexing fails --------------------------------------

def test_upload_rolls_back_document_row_on_index_failure(monkeypatch, client):
    def boom(_texts):
        raise LLMNotConfigured("simulated embedding outage")

    monkeypatch.setattr(embeddings, "embed_texts", boom)
    r = client.post(
        "/documents/upload",
        files={"file": ("rollback_unique.txt", b"some content to index", "text/plain")},
    )
    assert r.status_code == 503
    db = SessionLocal()
    try:
        assert db.query(Document).filter_by(filename="rollback_unique.txt").count() == 0
    finally:
        db.close()


# ---- Phase 3: cross-user isolation end-to-end (real auth, test (a)) --------

@pytest.mark.realauth
def test_cross_user_rag_isolation(stub_embeddings, stub_llm, client):
    def auth_headers(email):
        client.post("/auth/register", json={"email": email, "password": "password123"})
        token = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    owner = auth_headers("owner@example.com")
    intruder = auth_headers("intruder@example.com")

    up = client.post(
        "/documents/upload",
        files={"file": ("secret.txt", b"The vault code is gryphon-zenith-9.", "text/plain")},
        headers=owner,
    )
    assert up.status_code == 200

    # Intruder sees no documents and cannot retrieve the owner's content.
    assert client.get("/documents", headers=intruder).json() == []
    rb = client.post("/ask", json={"question": "What is the vault code?"}, headers=intruder)
    assert rb.status_code == 200
    assert rb.json()["sources"] == []
    assert "gryphon" not in rb.json()["answer"].lower()

    # Owner sees and retrieves their own document.
    assert any(d["filename"] == "secret.txt" for d in client.get("/documents", headers=owner).json())
    ra = client.post("/ask", json={"question": "What is the vault code?"}, headers=owner)
    assert ra.json()["sources"]
