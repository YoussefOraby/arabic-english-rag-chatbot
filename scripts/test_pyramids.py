"""Quick test: extract + chunk pyramids.pdf, print stats."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pdf.extractor import extract_pages
from src.pdf.chunker import chunk_pages
from src.config import settings

pdf = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "pyramids.pdf"
assert pdf.exists(), f"{pdf} not found"

pages = extract_pages(pdf)
print(f"Total pages extracted: {len(pages)}")
for p in pages:
    print(f"  Page {p.page_num}: {len(p.text)} chars")

chunks = chunk_pages(pages, settings.pdf, source_file=pdf.name)
print(f"\nTotal chunks created: {len(chunks)}")

# Count chunks per page
from collections import Counter
page_counts = Counter(c.page_num for c in chunks)
for p in sorted(page_counts):
    print(f"  Page {p}: {page_counts[p]} chunks ({sum(len(c.text) for c in chunks if c.page_num == p)} chars)")

page22 = [c for c in chunks if c.page_num == 22]
print(f"\n=== Page 22: {len(page22)} chunks ===")
for i, c in enumerate(page22):
    print(f"\n--- Chunk {i+1} (page {c.page_num}, {len(c.text)} chars) ---")
    print(c.text)
