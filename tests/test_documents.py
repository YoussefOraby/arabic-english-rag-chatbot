"""Tests for the SQLite document registry."""

from datetime import datetime
from pathlib import Path

import pytest

from src.document.registry import DocumentRegistry
from src.document.schemas import DocumentStatus


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_documents.db"


@pytest.fixture
def registry(db_path: Path) -> DocumentRegistry:
    return DocumentRegistry(db_path)


class TestDocumentRegistry:
    def test_create(self, registry: DocumentRegistry):
        doc = registry.create("test.pdf", file_size_bytes=1024)
        assert doc.document_id is not None
        assert doc.filename == "test.pdf"
        assert doc.status == DocumentStatus.UPLOADED
        assert doc.file_size_bytes == 1024
        assert isinstance(doc.upload_time, datetime)

    def test_get(self, registry: DocumentRegistry):
        created = registry.create("doc1.pdf")
        fetched = registry.get(created.document_id)
        assert fetched is not None
        assert fetched.document_id == created.document_id
        assert fetched.filename == "doc1.pdf"

    def test_get_not_found(self, registry: DocumentRegistry):
        assert registry.get("nonexistent") is None

    def test_get_by_filename(self, registry: DocumentRegistry):
        registry.create("test.pdf")
        fetched = registry.get_by_filename("test.pdf")
        assert fetched is not None
        assert fetched.filename == "test.pdf"

    def test_list(self, registry: DocumentRegistry):
        registry.create("a.pdf")
        registry.create("b.pdf")
        docs = registry.list()
        assert len(docs) == 2

    def test_list_empty(self, registry: DocumentRegistry):
        assert registry.list() == []

    def test_update_status(self, registry: DocumentRegistry):
        doc = registry.create("doc.pdf")
        registry.update_status(
            doc.document_id,
            DocumentStatus.PROCESSING,
            num_pages=5,
            num_chunks=12,
        )
        updated = registry.get(doc.document_id)
        assert updated is not None
        assert updated.status == DocumentStatus.PROCESSING
        assert updated.num_pages == 5
        assert updated.num_chunks == 12

    def test_update_status_failed(self, registry: DocumentRegistry):
        doc = registry.create("doc.pdf")
        registry.update_status(
            doc.document_id,
            DocumentStatus.FAILED,
            error_message="Corrupted PDF",
        )
        updated = registry.get(doc.document_id)
        assert updated is not None
        assert updated.status == DocumentStatus.FAILED
        assert updated.error_message == "Corrupted PDF"

    def test_delete(self, registry: DocumentRegistry):
        doc = registry.create("delete_me.pdf")
        assert registry.delete(doc.document_id) is True
        assert registry.get(doc.document_id) is None

    def test_delete_not_found(self, registry: DocumentRegistry):
        assert registry.delete("nonexistent") is False

    def test_clear(self, registry: DocumentRegistry):
        registry.create("a.pdf")
        registry.create("b.pdf")
        assert len(registry.list()) == 2
        registry.clear()
        assert registry.list() == []

    def test_list_ordered_by_upload_time(self, registry: DocumentRegistry):
        from time import sleep

        d1 = registry.create("first.pdf")
        sleep(0.01)
        d2 = registry.create("second.pdf")
        docs = registry.list()
        assert docs[0].document_id == d2.document_id
        assert docs[1].document_id == d1.document_id

    def test_create_with_hash(self, registry: DocumentRegistry):
        doc = registry.create("hashed.pdf", file_size_bytes=2048, file_hash="a" * 64)
        assert doc.file_hash == "a" * 64
        fetched = registry.get(doc.document_id)
        assert fetched is not None
        assert fetched.file_hash == "a" * 64

    def test_get_by_hash_found(self, registry: DocumentRegistry):
        doc = registry.create("doc.pdf", file_hash="b" * 64)
        found = registry.get_by_hash("b" * 64)
        assert found is not None
        assert found.document_id == doc.document_id
        assert found.file_hash == "b" * 64

    def test_get_by_hash_not_found(self, registry: DocumentRegistry):
        assert registry.get_by_hash("nonexistent") is None

    def test_get_by_hash_empty(self, registry: DocumentRegistry):
        assert registry.get_by_hash("") is None

    def test_duplicate_hash_returns_most_recent(self, registry: DocumentRegistry):
        from time import sleep

        registry.create("first.pdf", file_hash="dup_hash")
        sleep(0.01)
        d2 = registry.create("second.pdf", file_hash="dup_hash")
        found = registry.get_by_hash("dup_hash")
        assert found is not None
        assert found.document_id == d2.document_id

    def test_create_without_hash_defaults_empty(self, registry: DocumentRegistry):
        doc = registry.create("nohash.pdf")
        assert doc.file_hash == ""
