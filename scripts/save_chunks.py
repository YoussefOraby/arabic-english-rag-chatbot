"""Extract, chunk, and save pyramids.pdf chunks to JSON."""
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

pdf = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "pyramids.pdf"
pages = extract_pages(pdf)
chunks = chunk_pages(pages, settings.pdf, source_file=pdf.name)

out = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
path = out / "pyramids_chunks.json"

# Convert Chunk dataclasses to dicts for JSON serialization
data = []
for c in chunks:
    d = asdict(c)
    d["created_at"] = ""  # keep empty, not set at this stage
    data.append(d)

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(chunks)} chunks to {path}")
print(f"File size: {path.stat().st_size / 1024:.1f} KB")
