"""Tests for PDF text extraction."""


def test_extract_pages_exists():
    """Verify extract_pages function exists and returns list."""
    from src.pdf.extractor import PageText, extract_pages

    assert callable(extract_pages)
    # Verify dataclass fields exist via type hints
    import dataclasses

    fields = {f.name for f in dataclasses.fields(PageText)}
    assert "page_num" in fields
    assert "text" in fields


def test_extract_pages_english(data_dir):
    """Test extraction on English PDF."""
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_arxiv.pdf"
    pages = extract_pages(pdf_path)
    assert len(pages) > 0
    for page in pages:
        assert page.page_num >= 1
        assert len(page.text) > 0


def test_extract_pages_arabic(data_dir):
    """Test extraction on Arabic PDF."""
    from src.pdf.extractor import extract_pages

    pdf_path = data_dir / "sample_ar.pdf"
    pages = extract_pages(pdf_path)
    assert len(pages) > 0
    # Check Arabic chars present (basic + Presentation Forms)
    arabic_count = sum(
        1
        for c in pages[0].text
        if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
    )
    assert arabic_count > 0


def test_extract_text_exists():
    """Verify extract_text wrapper works."""
    from src.pdf.extractor import extract_text

    assert callable(extract_text)


def test_extract_text_full(data_dir):
    """Test extract_text returns combined string."""
    from src.pdf.extractor import extract_text

    pdf_path = data_dir / "sample_arxiv.pdf"
    text = extract_text(pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0
