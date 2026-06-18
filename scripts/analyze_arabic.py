"""Analyze Arabic extraction quality and save to file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unicodedata

import fitz


def main():
    pdf_path = Path("data/raw/sample_ar.pdf")
    report_path = Path("data/raw/arabic_quality_report.txt")

    doc = fitz.open(pdf_path)
    text = doc[0].get_text("text")
    doc.close()

    normalized = unicodedata.normalize("NFKC", text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Arabic Extraction Quality Report ===\n\n")
        f.write(f"PDF: {pdf_path.name}\n")
        f.write(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB\n\n")

        f.write("--- Raw extracted text (first 300 chars) ---\n")
        f.write(text[:300] + "\n\n")

        f.write("--- NFKC normalized text (first 300 chars) ---\n")
        f.write(normalized[:300] + "\n\n")

        # Check for expected Arabic words
        expected = ["الفصل", "الأول", "الذكاء", "الاصطناعي", "معالجة", "اللغة", "الطبيعية", "Mixed"]
        f.write("--- Word Check ---\n")
        for word in expected:
            found = word in normalized
            f.write(f"  '{word}' found in text: {found}\n")

        # Stats
        basic_ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        present_ar = sum(1 for c in text if "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff")
        f.write("\n--- Stats ---\n")
        f.write(f"Basic Arabic chars (U+0600-06FF): {basic_ar}\n")
        f.write(f"Presentation Forms (U+FB50-FDFF, U+FE70-FEFF): {present_ar}\n")
        f.write(f"Total chars: {len(text)}\n")
        f.write(f"Lines: {len(text.split(chr(10)))}\n")

        # Character mapping sample
        f.write("\n--- First 30 chars comparison ---\n")
        for i, (a, b) in enumerate(zip(text[:30], normalized[:30])):
            f.write(f"  [{i}] raw={repr(a)} norm={repr(b)}\n")

        f.write("\n=== End of Report ===\n")

    print(f"Report saved to: {report_path}")
    print(f"Raw chars: {len(text)}, Basic Arabic: N/A (not printable)")
    print("Open with: notepad " + str(report_path))


if __name__ == "__main__":
    main()
