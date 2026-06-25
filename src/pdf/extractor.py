"""PDF text extraction using PyMuPDF (fitz).
Extracts text from each page, handling Arabic RTL and CJK text correctly."""

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageText:
    """Represents extracted text from one page of a PDF."""

    page_num: int  # 1-indexed
    text: str  # Raw text content


def extract_pages(pdf_path: Path) -> list[PageText]:
    """
    Extract text from each page of a PDF.

    Uses PyMuPDF's "text" mode which preserves reading order
    better than "blocks" mode, especially for Arabic RTL text.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PageText objects, one per page

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF is empty or corrupted
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # "text" mode preserves reading order and handles RTL correctly
        text = page.get_text("text")
        # Normalize Arabic Presentation Forms (U+FE70-U+FEFF) to standard
        # Arabic block (U+0600-U+06FF), which PyMuPDF's TextWriter may
        # produce when rendering RTL text with a font like Tahoma.
        text = unicodedata.normalize("NFKC", text)
        # Strip leading/trailing whitespace per page
        text = text.strip()
        pages.append(PageText(page_num=page_num + 1, text=text))

    doc.close()

    if not pages:
        raise ValueError(f"PDF is empty or unreadable: {pdf_path}")

    return pages


def extract_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF as a single string.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Combined text from all pages separated by page breaks
    """
    pages = extract_pages(pdf_path)
    combined = []
    for page in pages:
        combined.append(page.text)
    return "\n\n".join(combined)
