"""Test PyMuPDF Arabic extraction modes and report to file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz


def main():
    pdf_path = Path("data/raw/sample_ar.pdf")
    report_path = Path("data/raw/arabic_extraction_test.txt")

    doc = fitz.open(pdf_path)
    page = doc[0]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Arabic PDF Extraction Mode Comparison ===\n\n")
        f.write(f"PDF: {pdf_path.name}\n")
        f.write(f"Pages: {len(doc)}\n\n")

        modes = ["text", "blocks", "dict", "rawdict"]
        for mode in modes:
            f.write(f"--- Mode: {mode} ---\n")
            data = page.get_text(mode)
            if isinstance(data, str):
                # Show first 300 chars
                f.write(data[:300] + "\n\n")
            elif isinstance(data, dict):
                # Show text from first block
                blocks = data.get("blocks", [])
                if blocks and blocks[0].get("type") == 0:  # text block
                    lines = blocks[0].get("lines", [])
                    for line in lines[:3]:
                        spans = line.get("spans", [])
                        for span in spans:
                            f.write(span.get("text", "") + " | ")
                f.write("\n\n")
            elif isinstance(data, list):
                for item in data[:5]:
                    if isinstance(item, dict):
                        f.write(str(item.get("text", ""))[:80] + "\n")
            else:
                f.write(str(data)[:300] + "\n\n")

        f.write("=== End of Report ===\n")

    doc.close()
    print(f"Report saved to: {report_path}")
    print("Open with: notepad " + str(report_path))


if __name__ == "__main__":
    main()
