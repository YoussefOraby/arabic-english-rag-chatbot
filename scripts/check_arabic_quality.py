"""Arabic quality check for Valeo PDF extraction."""
import io
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pdf.extractor import extract_pages
from src.config import settings

pdf = Path("D:/RAG/data/raw/valeo-code-of-business-ethics-2023_ar.pdf")
pages = extract_pages(pdf)

print("=== ARABIC QUALITY CHECK ===\n")

# 1. OCR threshold
avg_chars = sum(len(p.text) for p in pages) / len(pages)
print(f"1. OCR threshold: {settings.pdf.ocr_threshold_chars} chars")
print(f"   Avg chars/page: {avg_chars:.0f}")
low_pages = [p for p in pages if len(p.text) < settings.pdf.ocr_threshold_chars]
if low_pages:
    print(f"   Pages below threshold: {[p.page_num for p in low_pages]}")
else:
    print("   All pages above threshold - OCR not needed")
print()

# 2. Garbled / replacement chars
total = 0
garbled = 0
replacement = 0
for p in pages:
    for ch in p.text:
        total += 1
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in "\n\r\t":
            garbled += 1
        if ord(ch) == 0xFFFD:
            replacement += 1
            garbled += 1
print(f"2. Total characters: {total}")
print(f"   Replacement chars (U+FFFD): {replacement}")
print(f"   Suspicious control chars: {garbled - replacement}")
if garbled == 0:
    print("   No garbled characters found")
else:
    print(f"   WARNING: {garbled} garbled characters")
print()

# 3. Arabic vs Latin ratio
arabic_count = sum(
    1 for p in pages for c in p.text
    if "\u0600" <= c <= "\u06FF"
    or "\uFB50" <= c <= "\uFDFF"
    or "\uFE70" <= c <= "\uFEFF"
)
pct = arabic_count / total * 100 if total else 0
print(f"3. Arabic chars: {arabic_count} ({pct:.1f}%)")
print(f"   Non-Arabic:   {total - arabic_count} ({100-pct:.1f}%)")
print()

# 4. Hidden formatting characters
format_issues = []
for p in pages:
    for ch in p.text:
        if ord(ch) in (0xFEFF, 0x200B, 0x200E, 0x200F):
            format_issues.append((p.page_num, ord(ch)))
            break
if format_issues:
    print(f"4. Found {len(format_issues)} pages with formatting chars")
else:
    print("4. No hidden formatting characters found")
print()

# 5. Text direction markers
lrm = sum(1 for p in pages for c in p.text if c == "\u200e")
rlm = sum(1 for p in pages for c in p.text if c == "\u200f")
print(f"5. LRM marks: {lrm}, RLM marks: {rlm}")
print()

# 6. Show per-page stats
print("6. Per-page character counts:")
for p in pages:
    flag = " *** LOW ***" if len(p.text) < settings.pdf.ocr_threshold_chars else ""
    print(f"   Page {p.page_num:2d}: {len(p.text):5d} chars{flag}")
print()

# 7. Summary
print("=== SUMMARY ===")
problems = []
if garbled > 0:
    problems.append(f"{garbled} garbled characters")
if replacement > 0:
    problems.append(f"{replacement} replacement chars")
if low_pages:
    problems.append(f"{len(low_pages)} low-content pages (would trigger OCR)")
if problems:
    print("ISSUES FOUND: " + ", ".join(problems))
else:
    print("PASSED - Arabic text is clean and readable")
print(f"{len(pages)} pages, {total} chars, {arabic_count} Arabic ({pct:.1f}%)")
