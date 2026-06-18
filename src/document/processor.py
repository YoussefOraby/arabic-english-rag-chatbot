"""Background document processing — extract, chunk, embed, store.

Used by the API upload endpoint and can be reused by the UI.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from src.config import settings
from src.document.registry import DocumentRegistry
from src.document.schemas import DocumentStatus
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.pdf.chunker import chunk_pages
from src.pdf.extractor import extract_pages
from src.pdf.ocr import extract_pages_with_fallback
from src.retrieval.hybrid import BM25Index

logger = logging.getLogger(__name__)


_bm25_instance: BM25Index | None = None


def _get_bm25() -> BM25Index:
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Index()
    return _bm25_instance


def process_document(
    doc_id: str,
    pdf_path: Path,
    filename: str,
    registry: DocumentRegistry,
) -> None:
    """Full pipeline: extract → chunk → embed → store → rebuild BM25.

    Updates the registry through uploaded → processing → indexed / failed.
    """
    try:
        registry.update_status(doc_id, DocumentStatus.PROCESSING)

        embedder = Embedder(settings.embeddings)
        store = ChromaStore(settings.vector_store, embedder)

        pages = extract_pages_with_fallback(
            pdf_path,
            text_extract_fn=extract_pages,
            lang=settings.pdf.ocr_language,
            ocr_threshold=settings.pdf.ocr_threshold_chars,
        )

        chunks = chunk_pages(
            pages,
            settings.pdf,
            source_file=filename,
            embedder=embedder,
            pdf_path=pdf_path,
        )

        now_iso = datetime.now(UTC).isoformat()
        embedding_model = settings.embeddings.model_name
        for c in chunks:
            c.document_id = doc_id
            c.original_text = c.text
            c.processed_text = c.text
            c.embedding_model = embedding_model
            c.created_at = now_iso

        store.add_chunks(chunks)

        bm25 = _get_bm25()
        bm25.rebuild(store)

        registry.update_status(
            doc_id,
            DocumentStatus.INDEXED,
            num_pages=len(pages),
            num_chunks=len(chunks),
        )

        logger.info(
            "Document %s (%s) indexed: %d pages, %d chunks",
            doc_id,
            filename,
            len(pages),
            len(chunks),
        )

    except Exception as e:
        logger.exception("Failed to process document %s (%s)", doc_id, filename)
        registry.update_status(
            doc_id,
            DocumentStatus.FAILED,
            error_message=str(e),
        )
