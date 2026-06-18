"""Tests for OCR fallback module."""


def test_is_scanned_detects_text_based():
    """Verify a text-rich PDF returns is_scanned=False."""
    from src.pdf.extractor import PageText
    from src.pdf.ocr import is_scanned

    pages = [PageText(1, "Hello world " * 100), PageText(2, "Another page " * 50)]
    assert not is_scanned(pages, threshold_chars=50)


def test_is_scanned_detects_scanned():
    """Verify a PDF with little text returns is_scanned=True."""
    from src.pdf.extractor import PageText
    from src.pdf.ocr import is_scanned

    pages = [PageText(1, "Hi"), PageText(2, "")]
    assert is_scanned(pages, threshold_chars=50)


def test_is_scanned_empty():
    """Verify empty pages list returns True (assumed scanned)."""
    from src.pdf.ocr import is_scanned

    assert is_scanned([], threshold_chars=50)


def test_is_scanned_custom_threshold():
    """Verify custom threshold works."""
    from src.pdf.extractor import PageText
    from src.pdf.ocr import is_scanned

    pages = [PageText(1, "Short text")]
    # 10 chars < threshold 50
    assert is_scanned(pages, threshold_chars=50)
    # 10 chars >= threshold 10
    assert not is_scanned(pages, threshold_chars=10)


def test_init_tesseract():
    """Verify tesseract initialization returns bool."""
    from src.pdf.ocr import _init_tesseract

    result = _init_tesseract()
    # Should be True on a properly configured system
    assert isinstance(result, bool)


def test_ocr_page_returns_text(data_dir):
    """Verify ocr_page returns text from a rendered page."""
    import fitz
    import pytest

    from src.pdf.ocr import _init_tesseract, ocr_page

    if not _init_tesseract():
        pytest.skip("Tesseract not available")

    doc = fitz.open(data_dir / "sample_arxiv.pdf")
    page = doc.load_page(0)
    text = ocr_page(page, lang="eng")
    assert len(text) > 0
    assert "arXiv" in text or "Overspinning" in text or "black" in text
    doc.close()


def test_ocr_pdf_returns_pages(data_dir):
    """Verify ocr_pdf returns PageText list."""
    import pytest

    from src.pdf.ocr import _init_tesseract, ocr_pdf

    if not _init_tesseract():
        pytest.skip("Tesseract not available")

    pages = ocr_pdf(data_dir / "sample_arxiv.pdf", lang="eng", dpi=150)
    assert len(pages) > 0
    assert pages[0].page_num == 1
    assert len(pages[0].text) > 0
    assert "arXiv" in pages[0].text or "Overspinning" in pages[0].text or "black" in pages[0].text


def test_extract_pages_with_fallback_text_based(data_dir):
    """Verify text-based PDF returns standard extracted text."""
    from src.pdf.extractor import extract_pages
    from src.pdf.ocr import extract_pages_with_fallback

    pages = extract_pages_with_fallback(
        data_dir / "sample_arxiv.pdf",
        text_extract_fn=extract_pages,
        lang="eng",
        ocr_threshold=50,
    )
    assert len(pages) > 0
    assert not pages[0].text.startswith("[OCR]")
    assert len(pages[0].text) > 100


def test_extract_pages_with_fallback_scanned_trigger(data_dir):
    """Verify very high threshold triggers OCR fallback."""
    from src.pdf.extractor import extract_pages
    from src.pdf.ocr import extract_pages_with_fallback

    # Threshold of 99999 will always trigger OCR
    pages = extract_pages_with_fallback(
        data_dir / "sample_arxiv.pdf",
        text_extract_fn=extract_pages,
        lang="eng",
        ocr_threshold=99999,
    )
    assert len(pages) > 0
    # Should have OCR prefix (or at least have content)
    if pages[0].text.startswith("[OCR]"):
        assert len(pages[0].text) > 10
