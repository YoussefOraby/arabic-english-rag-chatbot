"""Tests for the document background processor."""

from unittest.mock import Mock, patch

import pytest

from src.document.schemas import DocumentRecord, DocumentStatus


@pytest.fixture
def mock_registry():
    """Create a mock DocumentRegistry."""
    registry = Mock()
    doc = DocumentRecord(
        document_id="test_doc_001",
        filename="test.pdf",
        status=DocumentStatus.UPLOADED,
    )
    registry.get.return_value = doc
    registry.create.return_value = doc
    return registry


def test_process_document_success(mock_registry, tmp_path):
    """Verify process_document completes the full pipeline."""
    from src.document.processor import process_document

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf content")

    mock_embedder = Mock()
    mock_store = Mock()
    mock_store.add_chunks.return_value = None
    mock_store.count_chunks.return_value = 3

    mock_page = Mock()
    mock_page.text = "Sample page content"
    mock_page.page_num = 1

    mock_chunk = Mock()
    mock_chunk.text = "Sample chunk"
    mock_chunk.page_num = 1
    mock_chunk.source_file = "test.pdf"
    mock_chunk.chunk_id = "chunk_001"
    mock_chunk.chunk_type = "text"

    with (
        patch("src.document.processor.Embedder", return_value=mock_embedder),
        patch("src.document.processor.ChromaStore", return_value=mock_store),
        patch(
            "src.document.processor.extract_pages_with_fallback",
            return_value=[mock_page],
        ),
        patch(
            "src.document.processor.chunk_pages",
            return_value=[mock_chunk],
        ),
        patch("src.document.processor._get_bm25", return_value=Mock()),
    ):
        process_document("test_doc_001", pdf_path, "test.pdf", mock_registry)

    mock_registry.update_status.assert_any_call(
        "test_doc_001",
        DocumentStatus.PROCESSING,
    )
    mock_store.add_chunks.assert_called_once()
    mock_registry.update_status.assert_any_call(
        "test_doc_001",
        DocumentStatus.INDEXED,
        num_pages=1,
        num_chunks=1,
    )


def test_process_document_metadata_set(mock_registry, tmp_path):
    """Verify chunk metadata is set before adding to store."""
    from src.document.processor import process_document

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf content")

    mock_embedder = Mock()
    mock_store = Mock()

    mock_page = Mock()
    mock_page.text = "Content"
    mock_page.page_num = 1

    chunk = Mock()
    chunk.text = "Chunk text"
    chunk.page_num = 1
    chunk.source_file = "test.pdf"
    chunk.chunk_id = "chunk_001"
    chunk.chunk_type = "text"
    chunk.document_id = None
    chunk.original_text = None
    chunk.processed_text = None
    chunk.embedding_model = None
    chunk.created_at = None

    with (
        patch("src.document.processor.Embedder", return_value=mock_embedder),
        patch("src.document.processor.ChromaStore", return_value=mock_store),
        patch(
            "src.document.processor.extract_pages_with_fallback",
            return_value=[mock_page],
        ),
        patch(
            "src.document.processor.chunk_pages",
            return_value=[chunk],
        ),
        patch("src.document.processor._get_bm25", return_value=Mock()),
    ):
        process_document("doc_001", pdf_path, "test.pdf", mock_registry)

    stored_chunks = mock_store.add_chunks.call_args[0][0]
    stored = stored_chunks[0]
    assert stored.document_id == "doc_001"
    assert stored.original_text == "Chunk text"
    assert stored.processed_text == "Chunk text"
    assert stored.embedding_model is not None
    assert stored.created_at is not None


def test_process_document_failure(mock_registry, tmp_path):
    """Verify process_document updates status to FAILED on error."""
    from src.document.processor import process_document

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf content")

    with patch("src.document.processor.Embedder", side_effect=ValueError("Model load failed")):
        process_document("doc_fail", pdf_path, "fail.pdf", mock_registry)

    mock_registry.update_status.assert_any_call(
        "doc_fail",
        DocumentStatus.PROCESSING,
    )
    mock_registry.update_status.assert_any_call(
        "doc_fail",
        DocumentStatus.FAILED,
        error_message="Model load failed",
    )


def test_process_document_io_error(mock_registry, tmp_path):
    """Verify IO errors during extraction are captured."""
    from src.document.processor import process_document

    pdf_path = tmp_path / "missing.pdf"
    mock_embedder = Mock()
    with (
        patch("src.document.processor.Embedder", return_value=mock_embedder),
        patch(
            "src.document.processor.extract_pages_with_fallback",
            side_effect=FileNotFoundError("No such file"),
        ),
    ):
        process_document("doc_io", pdf_path, "missing.pdf", mock_registry)

    mock_registry.update_status.assert_any_call(
        "doc_io",
        DocumentStatus.FAILED,
        error_message="No such file",
    )


def test_process_document_extract_empty_pages(mock_registry, tmp_path):
    """Verify empty page list is handled."""
    from src.document.processor import process_document

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_text("")

    mock_embedder = Mock()
    mock_store = Mock()
    mock_bm25 = Mock()

    with (
        patch("src.document.processor.Embedder", return_value=mock_embedder),
        patch("src.document.processor.ChromaStore", return_value=mock_store),
        patch(
            "src.document.processor.extract_pages_with_fallback",
            return_value=[],
        ),
        patch(
            "src.document.processor.chunk_pages",
            return_value=[],
        ),
        patch("src.document.processor._get_bm25", return_value=mock_bm25),
    ):
        process_document("doc_empty", pdf_path, "empty.pdf", mock_registry)

    mock_store.add_chunks.assert_called_once_with([])
    mock_bm25.rebuild.assert_called_once_with(mock_store)
    mock_registry.update_status.assert_any_call(
        "doc_empty",
        DocumentStatus.INDEXED,
        num_pages=0,
        num_chunks=0,
    )
