"""Tests for document Pydantic schemas."""

from datetime import UTC, datetime

from src.document.schemas import DocumentRecord, DocumentStatus


def test_document_status_values():
    assert DocumentStatus.UPLOADED.value == "uploaded"
    assert DocumentStatus.PROCESSING.value == "processing"
    assert DocumentStatus.INDEXED.value == "indexed"
    assert DocumentStatus.FAILED.value == "failed"


def test_document_status_members():
    assert len(DocumentStatus) == 4


def test_document_record_defaults():
    doc = DocumentRecord(filename="test.pdf")
    assert doc.document_id is not None
    assert len(doc.document_id) == 12
    assert isinstance(doc.upload_time, datetime)
    assert doc.status == DocumentStatus.UPLOADED
    assert doc.file_size_bytes == 0
    assert doc.num_pages is None
    assert doc.num_chunks is None
    assert doc.error_message is None
    assert doc.file_hash == ""


def test_document_record_all_fields():
    doc = DocumentRecord(
        document_id="abc123def456",
        filename="report.pdf",
        upload_time=datetime(2025, 6, 1, tzinfo=UTC),
        status=DocumentStatus.INDEXED,
        file_size_bytes=204800,
        file_hash="abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
        num_pages=10,
        num_chunks=42,
        error_message=None,
    )
    assert doc.document_id == "abc123def456"
    assert doc.filename == "report.pdf"
    assert doc.upload_time == datetime(2025, 6, 1, tzinfo=UTC)
    assert doc.status == DocumentStatus.INDEXED
    assert doc.file_size_bytes == 204800
    assert doc.file_hash == "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"
    assert doc.num_pages == 10
    assert doc.num_chunks == 42
    assert doc.error_message is None


def test_document_record_with_error():
    doc = DocumentRecord(
        filename="broken.pdf",
        status=DocumentStatus.FAILED,
        error_message="Corrupted PDF file",
    )
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message == "Corrupted PDF file"


def test_document_record_from_dict():
    data = {
        "document_id": "abc123def456",
        "filename": "test.pdf",
        "upload_time": "2025-06-01T12:00:00+00:00",
        "status": "indexed",
        "file_size_bytes": 1024,
        "file_hash": "def456abc123def456abc123def456abc123def456abc123def456abc123def456",
        "num_pages": 5,
        "num_chunks": 20,
        "error_message": None,
    }
    doc = DocumentRecord.model_validate(data)
    assert doc.document_id == "abc123def456"
    assert doc.filename == "test.pdf"
    assert doc.status == DocumentStatus.INDEXED
    assert doc.file_size_bytes == 1024
    assert doc.file_hash == "def456abc123def456abc123def456abc123def456abc123def456abc123def456"
    assert doc.num_pages == 5
    assert doc.num_chunks == 20


def test_document_record_serialize():
    doc = DocumentRecord(filename="serialize.pdf", file_size_bytes=512)
    data = doc.model_dump()
    assert data["filename"] == "serialize.pdf"
    assert data["file_size_bytes"] == 512
    assert "document_id" in data
    assert "upload_time" in data
    assert "status" in data


def test_document_record_unique_ids():
    doc1 = DocumentRecord(filename="a.pdf")
    doc2 = DocumentRecord(filename="b.pdf")
    assert doc1.document_id != doc2.document_id
