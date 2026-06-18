"""One-shot PDF ingestion pipeline.
Usage: python scripts/ingest.py [--dry-run]
Processes ALL PDFs in data/raw/ → extract → chunk → embed → ChromaDB"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.pdf.chunker import chunk_pages
from src.pdf.extractor import extract_pages
from src.pdf.ocr import extract_pages_with_fallback


def safe_preview(text: str, max_len: int = 60) -> str:
    """Return ASCII-safe preview of text (handles Arabic display on Windows)."""
    preview = text[:max_len]
    result = preview.encode("ascii", errors="replace").decode("ascii")
    return result


def find_pdfs(data_dir: Path) -> list[Path]:
    """Find all PDF files in the data directory."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    return pdfs


def process_pdfs(pdf_paths: list[Path], dry_run: bool = False) -> int:
    """
    Process all PDFs through extraction → chunking → embedding → ChromaDB.

    Args:
        pdf_paths: List of paths to PDF files
        dry_run: If True, only print what would be done

    Returns:
        Total number of chunks stored
    """
    total_chunks = 0

    if not dry_run:
        embedder = Embedder(settings.embeddings)
        store = ChromaStore(settings.vector_store, embedder)
        dim = embedder.dim
        print(f"[Embedding] Model: {settings.embeddings.model_name}")
        print(f"[Embedding] Dimension: {dim}")
        print(f"[ChromaDB] Directory: {settings.vector_store.persist_directory}")
        print(f"[ChromaDB] Collection: {settings.vector_store.collection_name}")
    else:
        embedder = None
        store = None
        print("[DRY RUN] Would initialize Embedder + ChromaDB")

    for pdf_path in pdf_paths:
        print(f"\n{'=' * 50}")
        print(f"File: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        print(f"{'=' * 50}")

        if dry_run:
            print(f"   [DRY RUN] Would extract pages from {pdf_path.name}")
            print(f"   [DRY RUN] Would chunk into ~{pdf_path.stat().st_size // 500} chunks")
            print("   [DRY RUN] Would embed and store in ChromaDB")
            total_chunks += pdf_path.stat().st_size // 500
            continue

        try:
            pages = extract_pages_with_fallback(
                pdf_path,
                text_extract_fn=extract_pages,
                lang=settings.pdf.ocr_language,
                ocr_threshold=settings.pdf.ocr_threshold_chars,
            )
            ocr_note = ""
            if pages and pages[0].text.startswith("[OCR]"):
                ocr_note = " (via OCR)"
            print(f"   Pages extracted: {len(pages)}{ocr_note}")

            chunks = chunk_pages(
                pages, settings.pdf, source_file=pdf_path.name, embedder=embedder, pdf_path=pdf_path
            )
            text_count = len([c for c in chunks if c.chunk_type == "text"])
            table_count = len([c for c in chunks if c.chunk_type == "table"])
            chart_count = len([c for c in chunks if c.chunk_type == "chart"])
            parts = [f"{text_count} text"]
            if table_count:
                parts.append(f"{table_count} tables")
            if chart_count:
                parts.append(f"{chart_count} charts")
            print(f"   Chunks created: {len(chunks)} ({', '.join(parts)})")

            embedder.embed_texts([c.text for c in chunks[:1]])
            print(f"   Embedding dimension: {embedder.dim}")

            store.add_chunks(chunks)
            print(f"   Stored in ChromaDB: {len(chunks)} chunks")

            for c in chunks[:2]:
                print(
                    f"     [{c.chunk_id}] Page {c.page_num} ({len(c.text)} chars): {safe_preview(c.text[:60])}..."
                )

            total_chunks += len(chunks)

        except Exception as e:
            print(f"   [!] Error processing {pdf_path.name}: {e}")

    if not dry_run and total_chunks > 0:
        final_count = store.count_chunks()
        print(f"\n[ChromaDB] Total chunks stored: {final_count}")

    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually processing",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing ChromaDB collection before ingesting"
    )
    args = parser.parse_args()

    data_dir = Path(settings.vector_store.persist_directory).parent.parent / "raw"

    if not data_dir.exists():
        print(f"[!] Data directory not found: {data_dir}")
        print("[*] Run 'python scripts/download_samples.py' first")
        sys.exit(1)

    pdfs = find_pdfs(data_dir)

    if not pdfs:
        print(f"[!] No PDFs found in {data_dir}")
        print("[*] Run 'python scripts/download_samples.py' to download samples")
        sys.exit(1)

    print(f"[PDF] Found {len(pdfs)} PDF(s):")
    for pdf in pdfs:
        size_kb = pdf.stat().st_size / 1024
        print(f"   \u2022 {pdf.name} ({size_kb:.1f} KB)")

    if args.dry_run:
        print("\n[DRY RUN MODE] No changes will be made")

    if args.clear and not args.dry_run:
        print("\n[ChromaDB] Clearing existing collection...")
        embedder = Embedder(settings.embeddings)
        store = ChromaStore(settings.vector_store, embedder)
        store.clear()
        print("[ChromaDB] Cleared")

    total = process_pdfs(pdfs, dry_run=args.dry_run)

    print(f"\n{'=' * 50}")
    if args.dry_run:
        print(f"[OK] Dry run complete — would create ~{total} chunks")
    else:
        print(f"[OK] Ingestion complete — {total} chunks stored")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
