"""Extract and chunk Valeo Arabic PDF — show results before saving."""
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pdf.extractor import extract_pages
from src.pdf.chunker import chunk_pages
from src.config import settings

pdf = Path("D:/RAG/data/raw/valeo-code-of-business-ethics-2023_ar.pdf")
pages = extract_pages(pdf)
chunks = chunk_pages(pages, settings.pdf, source_file=pdf.name)

print(f"Total chunks created: {len(chunks)}")
print()

# First 3 chunks
for i, c in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} (page {c.page_num}, {len(c.text)} chars) ---")
    print(c.text)
    print()

# Average chunk size
sizes = [len(c.text) for c in chunks]
avg = sum(sizes) / len(sizes)
print(f"Average chunk size: {avg:.0f} chars")
print(f"Min: {min(sizes)} chars, Max: {max(sizes)} chars")

# Chunks under 50 chars
short = [c for c in chunks if len(c.text.strip()) < 50]
print(f"\nChunks under 50 chars: {len(short)}")
if short:
    for c in short:
        print(f"  Page {c.page_num}, {len(c.text)} chars: {c.text.strip()!r}")

# Save to JSON
out = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
path = out / "valeo_ar_chunks.json"
data = [asdict(c) for c in chunks]
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(chunks)} chunks to {path}")
print(f"File size: {path.stat().st_size / 1024:.1f} KB")
