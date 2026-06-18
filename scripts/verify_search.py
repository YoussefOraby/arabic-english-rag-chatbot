"""Verify ChromaDB search results and save to file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore


def main():
    embedder = Embedder(settings.embeddings)
    store = ChromaStore(settings.vector_store, embedder)

    report = []
    report.append(f"Chunks in store: {store.count_chunks()}\n")

    # English query
    results = store.similarity_search("black hole event horizon", k=3)
    report.append("=== English query: 'black hole event horizon' ===\n")
    for r in results:
        report.append(f"  [{r.chunk.source_file} p{r.chunk.page_num}] score={r.score:.3f}\n")
        report.append(f"  Text: {r.chunk.text[:120]}...\n")
        report.append("\n")

    # Arabic query
    report.append("=== Arabic query: 'ذكاء اصطناعي' ===\n")
    results = store.similarity_search("ذكاء اصطناعي", k=3)
    for r in results:
        report.append(f"  [{r.chunk.source_file} p{r.chunk.page_num}] score={r.score:.3f}\n")
        report.append(f"  Text: {r.chunk.text[:200]}\n")
        report.append("\n")

    # Mixed query
    report.append("=== Mixed query: 'quantum black hole' ===\n")
    results = store.similarity_search("quantum black hole", k=2)
    for r in results:
        report.append(f"  [{r.chunk.source_file} p{r.chunk.page_num}] score={r.score:.3f}\n")
        report.append(f"  Text: {r.chunk.text[:120]}...\n")
        report.append("\n")

    report_path = Path("data/raw/search_verify.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)

    print(f"Report saved: {report_path}")
    print(f"Open with: notepad {report_path}")


if __name__ == "__main__":
    main()
