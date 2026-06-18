"""OCR fallback for scanned/image-based PDFs using Tesseract + PyMuPDF."""

import os
from pathlib import Path

import fitz
from PIL import Image

from src.pdf.extractor import PageText, extract_pages

# ── Tesseract path ───────────────────────────────────────────────────
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\UB-Mannheim\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]
_TESSDATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tessdata"


def _init_tesseract() -> bool:
    """Configure pytesseract path and tessdata. Returns True if available."""
    try:
        import pytesseract
    except ImportError:
        return False

    # Find tesseract binary
    tesseract_cmd = None
    for p in _TESSERACT_PATHS:
        if Path(p).exists():
            tesseract_cmd = p
            break
    if not tesseract_cmd:
        return False

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Set tessdata dir if we have our own
    if _TESSDATA_DIR.exists() and any(_TESSDATA_DIR.glob("*.traineddata")):
        os.environ.setdefault("TESSDATA_PREFIX", str(_TESSDATA_DIR))

    return True


def is_scanned(pages: list[PageText], threshold_chars: int = 50) -> bool:
    """Detect if a PDF is scanned (image-based) by checking text density.

    Returns True if the average characters per page is below *threshold*.
    """
    if not pages:
        return True
    total = sum(len(p.text) for p in pages)
    avg = total / len(pages)
    return avg < threshold_chars


def ocr_page(page: fitz.Page, lang: str = "eng+ara", dpi: int = 200) -> str:
    """Render a PyMuPDF page to image and run Tesseract OCR."""
    import pytesseract

    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang=lang, timeout=60)


def ocr_pdf(
    pdf_path: Path,
    lang: str = "eng+ara",
    dpi: int = 200,
) -> list[PageText]:
    """OCR an entire PDF, returning PageText for each page."""
    if not _init_tesseract():
        raise RuntimeError(
            "Tesseract OCR not found. Install Tesseract for Windows:\n"
            "  winget install UB-Mannheim.TesseractOCR\n"
            "Or download from: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    doc = fitz.open(pdf_path)
    pages: list[PageText] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = ocr_page(page, lang=lang, dpi=dpi)
        pages.append(PageText(page_num=page_num + 1, text=text.strip()))

    doc.close()
    return pages


def extract_pages_with_fallback(
    pdf_path: Path,
    text_extract_fn=None,
    lang: str = "eng+ara",
    ocr_threshold: int = 50,
) -> list[PageText]:
    """Try text extraction first; fall back to OCR if the PDF appears scanned.

    Args:
        pdf_path: Path to the PDF file
        text_extract_fn: Function to call for text extraction (default: extract_pages)
        lang: Tesseract language string (e.g. 'eng+ara')
        ocr_threshold: Avg chars/page below which triggers OCR fallback

    Returns:
        List of PageText objects
    """
    if text_extract_fn is None:
        text_extract_fn = extract_pages

    text_pages = text_extract_fn(pdf_path)

    if not is_scanned(text_pages, threshold_chars=ocr_threshold):
        return text_pages

    try:
        ocr_pages = ocr_pdf(pdf_path, lang=lang)
        if ocr_pages:
            # Mark as OCR-derived (store in text field with prefix for metadata)
            for p in ocr_pages:
                p.text = f"[OCR] {p.text}"
            return ocr_pages
    except Exception:
        pass

    return text_pages
