"""Tests for PDF text chunking."""


def test_chunk_pages_exists():
    """Verify chunk_pages function exists."""
    import dataclasses

    from src.pdf.chunker import Chunk, chunk_pages

    assert callable(chunk_pages)
    fields = {f.name for f in dataclasses.fields(Chunk)}
    assert "text" in fields
    assert "page_num" in fields
    assert "source_file" in fields
    assert "chunk_id" in fields
    assert "chunk_type" in fields
    assert "table_index" in fields


def test_chunk_pages_basic(data_dir):
    """Test chunking on English PDF."""
    from src.config import settings
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.page_num >= 1
        assert chunk.source_file == pdf_path.name
        assert len(chunk.chunk_id) > 0


def test_chunk_has_page_numbers(data_dir):
    """Verify chunk page numbers match original pages."""
    from src.config import settings
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name)

    # Chunk page numbers should be within page range
    max_page = max(p.page_num for p in pages)
    for chunk in chunks:
        assert 1 <= chunk.page_num <= max_page


def test_chunk_does_not_exceed_size(data_dir):
    """Verify no chunk exceeds max chunk_size (with tolerance for semantic merging)."""
    from src.config import settings
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name)

    for chunk in chunks:
        assert len(chunk.text) <= settings.pdf.chunk_size + 100  # tolerance for sentence boundaries


def test_chunk_arabic(data_dir):
    """Test chunking on Arabic PDF."""
    from src.config import settings
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_ar.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name)

    assert len(chunks) > 0
    arabic_count = sum(
        1
        for c in chunks[0].text
        if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
    )
    assert arabic_count > 0


def test_semantic_chunk_basic(data_dir):
    """Test semantic chunking with embedder."""
    from src.config import settings
    from src.embeddings.embedder import Embedder
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    embedder = Embedder(settings.embeddings)
    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name, embedder=embedder)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.page_num >= 1


def test_semantic_chunk_arabic(data_dir):
    """Test semantic chunking on Arabic PDF with embedder."""
    from src.config import settings
    from src.embeddings.embedder import Embedder
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    embedder = Embedder(settings.embeddings)
    pdf_path = data_dir / "sample_ar.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name, embedder=embedder)

    assert len(chunks) > 0
    arabic_count = sum(
        1
        for c in chunks[0].text
        if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
    )
    assert arabic_count > 0


def test_extract_table_chunks_basic(data_dir):
    """Verify tables are extracted from the arxiv paper."""
    from src.pdf.chunker import extract_table_chunks

    pdf_path = data_dir / "sample_arxiv.pdf"
    chunks = extract_table_chunks(pdf_path, source_file=pdf_path.name)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_type == "table"
        assert chunk.table_index is not None
        assert chunk.table_index >= 1
        assert chunk.page_num >= 1
        assert "|" in chunk.text


def test_extract_table_chunks_quality(data_dir):
    """Verify low-quality tables are filtered out."""
    from src.pdf.chunker import extract_table_chunks

    pdf_path = data_dir / "sample_arxiv.pdf"
    chunks = extract_table_chunks(pdf_path, source_file=pdf_path.name)

    for chunk in chunks:
        rows = chunk.text.strip().split("\n")
        assert len(rows) >= 3


def test_extract_table_chunks_no_tables(data_dir):
    """Verify Arabic PDF (no tables) returns empty list."""
    from src.pdf.chunker import extract_table_chunks

    pdf_path = data_dir / "sample_ar.pdf"
    chunks = extract_table_chunks(pdf_path, source_file=pdf_path.name)

    assert len(chunks) == 0


def test_extract_table_chunks_nonexistent():
    """Verify nonexistent PDF returns empty list (graceful degradation)."""
    from pathlib import Path

    from src.pdf.chunker import extract_table_chunks

    chunks = extract_table_chunks(Path("nonexistent.pdf"), source_file="x.pdf")
    assert len(chunks) == 0


def test_chunk_pages_with_tables(data_dir):
    """Verify chunk_pages returns table chunks when pdf_path is given."""
    from src.config import settings
    from src.pdf.chunker import chunk_pages
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, settings.pdf, source_file=pdf_path.name, pdf_path=pdf_path)

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    assert len(text_chunks) > 0
    assert len(table_chunks) > 0

    chunks_text_only = chunk_pages(pages, settings.pdf, source_file=pdf_path.name)
    assert len(chunks) > len(chunks_text_only)
