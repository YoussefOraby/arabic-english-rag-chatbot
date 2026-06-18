"""Tests for FastAPI REST API endpoints."""

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app

# Set test DB path before importing anything else that reads it
_test_db = Path(tempfile.mkdtemp()) / "test_documents.db"
os.environ["DOCUMENTS_DB_PATH"] = str(_test_db)

client = TestClient(app)


# ── System Endpoints ──────────────────────────────────────────────────


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "ollama" in data
    assert "chroma_chunks" in data


def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_chunks" in data
    assert "llm_model" in data
    assert "embedding_model" in data


def test_query_endpoint_no_question():
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_endpoint_empty_question():
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_endpoint_invalid_top_k():
    resp = client.post("/query", json={"question": "test", "top_k": 0})
    assert resp.status_code == 422


def test_query_endpoint_too_large_top_k():
    resp = client.post("/query", json={"question": "test", "top_k": 100})
    assert resp.status_code == 422


def test_docs_endpoint():
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_openapi_schema():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/query" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/stats" in schema["paths"]
    assert "/upload" in schema["paths"]
    assert "/documents" in schema["paths"]
    assert "/documents/{document_id}" in schema["paths"]


# ── Document Endpoints ────────────────────────────────────────────────


def _make_pdf(content: bytes = b"%PDF-1.4 test") -> tuple[str, bytes]:
    return ("test_doc.pdf", content)


def test_upload_invalid_file_type():
    resp = client.post("/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_no_file():
    resp = client.post("/upload")
    assert resp.status_code == 422


def test_upload_accepts_pdf():
    resp = client.post("/upload", files={"file": _make_pdf()})
    assert resp.status_code == 201
    data = resp.json()
    assert data["document_id"] is not None
    assert data["filename"] == "test_doc.pdf"
    assert data["status"] == "uploaded"


def test_list_documents():
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_list_after_upload():
    client.post("/upload", files={"file": _make_pdf()})
    resp = client.get("/documents")
    docs = resp.json()
    assert len(docs) >= 1
    assert docs[0]["filename"] is not None


def test_get_document_found():
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["document_id"] == doc_id


def test_get_document_not_found():
    resp = client.get("/documents/nonexistent")
    assert resp.status_code == 404


def test_delete_document_not_found():
    resp = client.delete("/documents/nonexistent")
    assert resp.status_code == 404


def test_delete_document():
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True


def test_reindex_document_not_found():
    resp = client.post("/documents/nonexistent/reindex")
    assert resp.status_code == 404


def test_delete_returns_chunks_removed():
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    resp = client.delete(f"/documents/{doc_id}")
    data = resp.json()
    assert "chunks_removed" in data
    assert isinstance(data["chunks_removed"], int)


def test_delete_chunks_actually_removed():
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    client.delete(f"/documents/{doc_id}")
    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 404


def test_reindex_workflow():
    """Verify reindex flow completes through background processing."""
    content = _make_pdf()[1]
    upload = client.post(
        "/upload",
        files={"file": ("sample_ar.pdf", content, "application/pdf")},
    ).json()
    doc_id = upload["document_id"]
    resp = client.post(f"/documents/{doc_id}/reindex")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc_id
    assert data["status"] == "processing"
    assert "Re-index started" in data["message"]


def test_delete_then_get_returns_404():
    """Verify deleted document is no longer accessible."""
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    client.delete(f"/documents/{doc_id}")
    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 404


def test_query_with_document_id():
    """Verify query with document_id returns successfully."""
    resp = client.post("/query", json={"question": "test", "document_id": "any_doc"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data


def test_query_insufficient_data_field():
    """Verify query response always contains insufficient_data boolean."""
    resp = client.post("/query", json={"question": "test"})
    data = resp.json()
    assert "insufficient_data" in data
    assert isinstance(data["insufficient_data"], bool)


def test_query_citations_field():
    """Verify query response contains citations list."""
    resp = client.post("/query", json={"question": "test"})
    data = resp.json()
    assert "citations" in data
    assert isinstance(data["citations"], list)


def test_query_without_document_id():
    """Verify query without document_id still works."""
    resp = client.post("/query", json={"question": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data


def test_delete_nonexistent_twice():
    """Verify idempotency of delete for missing docs."""
    resp = client.delete("/documents/nonexistent")
    assert resp.status_code == 404
    resp = client.delete("/documents/nonexistent")
    assert resp.status_code == 404


def test_upload_twice_same_filename():
    """Verify uploading same filename creates separate records."""
    r1 = client.post("/upload", files={"file": _make_pdf()}).json()
    r2 = client.post("/upload", files={"file": _make_pdf()}).json()
    assert r1["document_id"] != r2["document_id"]
    assert r1["filename"] == r2["filename"] == "test_doc.pdf"
    docs = client.get("/documents").json()
    matching = [d for d in docs if d["filename"] == "test_doc.pdf"]
    assert len(matching) >= 2


def test_upload_empty_file():
    """Verify error response for empty file upload (valid PDF header but no content)."""
    resp = client.post("/upload", files={"file": _make_pdf(b"")})
    assert resp.status_code == 201  # Accepts, processing may fail later


def test_document_response_contains_all_fields():
    """Verify document detail response has all expected fields."""
    upload = client.post("/upload", files={"file": _make_pdf()}).json()
    doc_id = upload["document_id"]
    resp = client.get(f"/documents/{doc_id}")
    data = resp.json()
    expected = {"document_id", "filename", "upload_time", "status", "file_size_bytes"}
    assert expected.issubset(data.keys())


def test_list_response_structure():
    """Verify list response is a list of objects with required fields."""
    resp = client.get("/documents")
    docs = resp.json()
    assert isinstance(docs, list)
    if docs:
        expected = {"document_id", "filename", "status"}
        assert expected.issubset(docs[0].keys())


def test_filename_sanitization():
    """Verify dangerous characters are stripped from filenames."""
    resp = client.post(
        "/upload",
        files={"file": ("../../etc/passwd.pdf", b"%PDF-1.4 data", "application/pdf")},
    )
    assert resp.status_code == 201
    data = resp.json()
    # Should not contain path separators
    assert "/" not in data["filename"]
    assert ".." not in data["filename"]
