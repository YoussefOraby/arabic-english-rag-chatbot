"""PDF table extraction using PyMuPDF (fitz)."""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class ExtractedTable:
    page_num: int
    table_index: int
    rows: int
    cols: int
    markdown: str
    headers: list[str]


def extract_tables(pdf_path: Path) -> list[ExtractedTable]:
    """Extract all tables from a PDF.

    Returns:
        List of ExtractedTable with clean markdown representation.
        Low-quality tables (<2 content rows) are filtered out.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    tables: list[ExtractedTable] = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        found = page.find_tables()
        for ti, t in enumerate(found.tables):
            raw = t.extract()
            md = _clean_table(raw)
            if md is None:
                continue
            header = _extract_header(raw)
            tables.append(
                ExtractedTable(
                    page_num=page_num + 1,
                    table_index=ti + 1,
                    rows=len(raw),
                    cols=max((len(r) for r in raw), default=0),
                    markdown=md,
                    headers=header,
                )
            )

    doc.close()
    return tables


def _extract_header(raw: list[list[str | None]]) -> list[str]:
    if not raw:
        return []
    first = raw[0]
    return [str(c or "").strip() for c in first]


def _clean_table(raw: list[list[str | None]]) -> str | None:
    """Convert raw table data to clean markdown. Returns None if table is too sparse."""
    cleaned: list[list[str]] = []
    for row in raw:
        cells = [str(c or "").strip() for c in row]
        cleaned.append(cells)

    while cleaned and all(not c for c in cleaned[-1]):
        cleaned.pop()

    if len(cleaned) < 2:
        return None

    content_rows = sum(1 for row in cleaned if any(len(c) > 3 for c in row))
    if content_rows < 3:
        return None

    # Clean cell content
    for row in cleaned:
        for i, cell in enumerate(row):
            row[i] = re.sub(r"\s+", " ", cell).strip()

    header = cleaned[0]
    data = cleaned[1:]

    sep = "|" + "|".join("---" for _ in header) + "|"
    header_line = "|" + "|".join(header) + "|"
    data_lines = ["|" + "|".join(row) + "|" for row in data]

    return "\n".join([header_line, sep] + data_lines)


def count_tables(pdf_path: Path) -> int:
    """Quick count of tables in a PDF (returns total, not just high-quality)."""
    try:
        doc = fitz.open(pdf_path)
        total = 0
        for i in range(doc.page_count):
            total += len(doc[i].find_tables().tables)
        doc.close()
        return total
    except Exception:
        return 0
