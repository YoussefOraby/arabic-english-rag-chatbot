"""Chart/diagram detection and extraction using pix2text layout analysis."""

import os
import uuid
from pathlib import Path

import fitz

from src.pdf.chunker import Chunk


class ChartDetector:
    """Lazy singleton wrapper around pix2text for figure detection.

    The underlying model (~1 GB) is loaded on first use.
    """

    _model = None

    @classmethod
    def _get_model(cls, device: str = "cpu"):
        if cls._model is None:
            try:
                from pix2text import Pix2Text

                cls._model = Pix2Text(device=device, enable_formula=False, enable_table=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load pix2text model: {e}")
        return cls._model

    @classmethod
    def is_available(cls) -> bool:
        """Check if pix2text is installed and importable."""
        try:
            import pix2text  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def detect_figures(cls, pdf_path: Path, page_numbers: list[int] | None = None) -> list[dict]:
        """Run pix2text layout analysis on a PDF, returning figure metadata.

        Returns:
            List of dicts with keys: page_num, box (x1,y1,x2,y2), width, height
        """
        model = cls._get_model()
        doc = model.recognize_pdf(str(pdf_path), page_numbers=page_numbers)

        # Force materialization of pages to avoid lazy-eval issues
        pages = list(doc.pages)

        figures = []
        for page_idx, page in enumerate(pages):
            for elem in page.elements:
                if str(getattr(elem, "type", "")) != "FIGURE":
                    continue
                box = getattr(elem, "box", None)
                if box is None or len(box) != 4:
                    continue
                figures.append(
                    {
                        "page_num": page_idx + 1,
                        "box": box,
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                    }
                )
        return figures


def extract_chart_chunks(
    pdf_path: Path,
    source_file: str,
    page_numbers: list[int] | None = None,
    dpi: int = 200,
) -> list[Chunk]:
    """Detect figures/charts in a PDF, OCR them, return as Chunk objects.

    Uses pix2text layout analysis to find figures, then renders each
    figure region with PyMuPDF and runs Tesseract OCR to extract text labels.

    Args:
        pdf_path: Path to the PDF file
        source_file: Source filename for chunk metadata
        page_numbers: Pages to analyze (0-indexed, None = all)
        dpi: Resolution for rendering + OCR

    Returns:
        List of Chunk objects with chunk_type="chart"
    """
    if not ChartDetector.is_available():
        return []

    try:
        figures = ChartDetector.detect_figures(pdf_path, page_numbers=page_numbers)
    except Exception:
        return []

    if not figures:
        return []

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    from src.pdf.ocr import _TESSDATA_DIR

    if _TESSDATA_DIR.exists() and any(_TESSDATA_DIR.glob("*.traineddata")):
        os.environ.setdefault("TESSDATA_PREFIX", str(_TESSDATA_DIR))

    doc = fitz.open(pdf_path)
    chunks: list[Chunk] = []

    for fig in figures:
        page_num = fig["page_num"]
        bx = fig["box"]
        page = doc.load_page(page_num - 1 if page_num > 0 else 0)

        # Render page, then crop figure region
        # pix2text boxes are in coordinates matching the rendered image (2550x3300 @200dpi)
        # We render at the same resolution
        pix = page.get_pixmap(dpi=dpi)
        from PIL import Image

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Crop figure region
        x1, y1, x2, y2 = [int(v) for v in bx]
        # Ensure within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pix.width, x2), min(pix.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        fig_img = img.crop((x1, y1, x2, y2))

        # OCR the cropped figure
        try:
            fig_text = pytesseract.image_to_string(fig_img, lang="eng+ara", timeout=30).strip()
        except Exception:
            fig_text = ""

        if not fig_text:
            fig_text = "(Chart/diagram detected — no text labels extracted)"

        chunk_id = f"{source_file}_p{page_num}_chart_{uuid.uuid4().hex[:8]}"
        chunks.append(
            Chunk(
                text=f"[Chart on page {page_num}]: {fig_text}",
                page_num=page_num,
                source_file=source_file,
                chunk_id=chunk_id,
                chunk_type="chart",
                table_index=None,
            )
        )

    doc.close()
    return chunks
