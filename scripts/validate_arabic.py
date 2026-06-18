"""Validate Arabic PDF extraction and chunking, save report to file."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.pdf.chunker import chunk_pages
from src.pdf.extractor import extract_pages


def main():
    pdf_path = Path("data/raw/sample_ar.pdf")

    pages = extract_pages(pdf_path)
    total_arabic = sum(
        1
        for p in pages
        for c in p.text
        if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
    )

    print(f"Pages: {len(pages)}")
    print(f"Total Arabic/Presentation chars: {total_arabic}")
    for p in pages:
        ar = sum(
            1
            for c in p.text
            if "\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
        )
        print(f"  Page {p.page_num}: {len(p.text)} chars ({ar} Arabic)")

    chunks = chunk_pages(pages, settings.pdf, source_file="sample_ar.pdf")
    print(f"\nChunks: {len(chunks)}")

    # Save report to file (UTF-8 for Arabic display)
    report_path = Path("data/raw/arabic_chunks_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Arabic PDF Chunking Validation Report ===\n\n")
        f.write(f"Source: {pdf_path.name}\n")
        f.write(f"Pages extracted: {len(pages)}\n")
        f.write(f"Chunks created: {len(chunks)}\n")
        f.write(f"Separators used: {settings.pdf.separators}\n\n")
        f.write("=" * 60 + "\n\n")

        for i, c in enumerate(chunks, 1):
            ar = sum(1 for ch in c.text if "\u0600" <= ch <= "\u06ff")
            f.write(f"--- Chunk {i} ---\n")
            f.write(f"  ID: {c.chunk_id}\n")
            f.write(f"  Page: {c.page_num}\n")
            f.write(f"  Length: {len(c.text)} chars\n")
            f.write(f"  Arabic chars: {ar}\n")
            f.write(f"  Text: {c.text}\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")

    print(f"\nFull Arabic chunk report saved to: {report_path}")
    print("Open with:")
    print(f"  notepad {report_path}")
    print("Or copy-paste the file content to see Arabic text")


if __name__ == "__main__":
    main()
