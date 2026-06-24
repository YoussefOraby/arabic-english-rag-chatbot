"""Quality check on extracted chunks from pyramids.pdf."""
import io
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pdf.extractor import extract_pages
from src.pdf.chunker import chunk_pages
from src.config import settings

pdf = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "pyramids.pdf"
pages = extract_pages(pdf)
chunks = chunk_pages(pages, settings.pdf, source_file=pdf.name)

total = len(chunks)
print(f"{'─'*60}")
print(f"  QUALITY CHECK — data/raw/pyramids.pdf")
print(f"{'─'*60}")
print(f"  Total pages:       {len(pages)}")
print(f"  Total chunks:      {total}")
print()

# ── 1. Empty chunks ──
empty = [c for c in chunks if not c.text.strip()]
print(f"  1. Empty chunks:       {len(empty)}")
if empty:
    for c in empty:
        print(f"     Chunk {c.chunk_id} (page {c.page_num}): [empty]")
print()

# ── 2. Short chunks (< 50 chars) ──
short = [c for c in chunks if 0 < len(c.text.strip()) < 50]
print(f"  2. Chunks < 50 chars:  {len(short)}")
if short:
    for c in short:
        print(f"     Chunk (page {c.page_num}, {len(c.text)} chars): {c.text.strip()[:80]!r}")
print()

# ── 3. Short chunks (< 100 chars) ──
short100 = [c for c in chunks if 0 < len(c.text.strip()) < 100]
print(f"  3. Chunks < 100 chars: {len(short100)}")
if short100:
    for c in short100[:10]:
        print(f"     Chunk (page {c.page_num}, {len(c.text)} chars): {c.text.strip()[:120]!r}")
    if len(short100) > 10:
        print(f"     ... and {len(short100) - 10} more")
print()

# ── 4. Longest & shortest ──
longest = max(chunks, key=lambda c: len(c.text))
shortest = min(chunks, key=lambda c: len(c.text))
print(f"  4. Longest chunk:  page {longest.page_num}, {len(longest.text)} chars")
print(f"     Text: {longest.text[:200]!r}...")
print(f"     Shortest chunk: page {shortest.page_num}, {len(shortest.text)} chars")
print(f"     Text: {shortest.text[:200]!r}")
print()

# ── 5. Average size ──
sizes = [len(c.text) for c in chunks]
avg = sum(sizes) / len(sizes)
print(f"  5. Average chunk size: {avg:.0f} chars")
print(f"     Min: {min(sizes)} chars, Max: {max(sizes)} chars, Median: {sorted(sizes)[len(sizes)//2]}")
print()

# ── 6. Weird characters ──
# Harmless zero-width / formatting chars common in PDFs
HARMLESS_FMT = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
SUSPICIOUS_CATS = {"Cc", "Cf", "Cs"}  # control, format, surrogate
weird_chunks = []
for c in chunks:
    weird_chars = []
    for ch in c.text:
        cat = unicodedata.category(ch)
        if cat in SUSPICIOUS_CATS and ord(ch) not in HARMLESS_FMT and ch not in "\n\r\t":
            weird_chars.append(f"U+{ord(ch):04X} ({unicodedata.name(ch, '?')})")
        if ord(ch) == 0xFFFD:
            weird_chars.append("U+FFFD (REPLACEMENT CHARACTER)")
    if weird_chars:
        weird_chunks.append((c, weird_chars[:5]))

udhr_count = sum(1 for c in chunks if "\ufffd" in c.text)
print(f"  6. Chunks with suspicious characters: {len(weird_chunks)}")
if weird_chunks:
    for c, chars in weird_chunks[:5]:
        print(f"     Chunk (page {c.page_num}): {', '.join(chars)}")
        print(f"     Text excerpt: {c.text[:150]!r}")
    if len(weird_chunks) > 5:
        print(f"     ... and {len(weird_chunks) - 5} more")
print()

# ── 7. Garbled patterns ──
garbled = []
for c in chunks:
    lines = c.text.split("\n")
    for line in lines:
        stripped = line.strip()
        # Check for repeated garbage patterns
        if len(stripped) > 10 and all(ch in "߷ߴߵ߶߷߸߹ߺ߽߾߿ࠀࠁࠂࠃ" for ch in stripped):
            garbled.append((c, stripped[:80]))
            break
        # Check for high density of non-ASCII non-letter chars
        if len(stripped) > 20:
            non_alpha = sum(1 for ch in stripped if not ch.isalpha() and not ch.isspace() and ch not in ".,:;!?()-[]{}'\"%/@#$&*+=<>0123456789")
            if non_alpha > len(stripped) * 0.5:
                garbled.append((c, f"non-alpha ratio {non_alpha}/{len(stripped)}: {stripped[:80]}"))
                break

print(f"  7. Chunks with likely garbled text: {len(garbled)}")
if garbled:
    for c, desc in garbled[:5]:
        print(f"     Chunk (page {c.page_num}): {desc}")
    if len(garbled) > 5:
        print(f"     ... and {len(garbled) - 5} more")
print()

# ── Summary ──
print(f"{'─'*60}")
issues = []
if empty:
    issues.append(f"{len(empty)} empty chunks")
if short:
    issues.append(f"{len(short)} very short chunks (< 50 chars)")
if weird_chunks:
    issues.append(f"{len(weird_chunks)} chunks with suspicious characters")
if garbled:
    issues.append(f"{len(garbled)} chunks with likely garbled text")

if not issues:
    print(f"  ✅ Quality check passed — ready to save")
else:
    print(f"  ❌ Issues found — needs fixing")
    for issue in issues:
        print(f"     • {issue}")
print(f"{'─'*60}")
