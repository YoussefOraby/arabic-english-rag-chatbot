"""Test Arabic search quality with multiple queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore


def main():
    embedder = Embedder(settings.embeddings)
    store = ChromaStore(settings.vector_store, embedder)
    total = store.count_chunks()
    report = [f"Chunks in store: {total}\n"]

    queries = [
        "ما هو الذكاء الاصطناعي",
        "التعلم الآلي والشبكات العصبية",
        "معالجة اللغة العربية",
        "تطبيقات الذكاء الاصطناعي",
        "التعلم العميق",
        "الترجمة الآلية",
    ]

    ar_found = 0
    for q in queries:
        report.append(f"{'=' * 60}")
        report.append(f"Q: {q}")
        report.append(f"{'=' * 60}")
        results = store.similarity_search(q, k=5, score_threshold=0.0)
        for r in results:
            is_ar = "sample_ar.pdf" in r.chunk.source_file
            if is_ar:
                ar_found += 1
            tag = "[AR]" if is_ar else "[EN]"
            report.append(f"  {tag} {r.chunk.source_file} p{r.chunk.page_num} score={r.score:.3f}")
            # Show first 80 chars of normalized text
            import unicodedata

            norm = unicodedata.normalize("NFKC", r.chunk.text)[:80]
            report.append(f"      {norm}")
        report.append("")

    report.append(f"\nArabic hits: {ar_found} out of {len(queries) * 5} top slots")

    report_path = Path("data/raw/arabic_search_test.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report saved: {report_path}")
    print(f"Open with: notepad {report_path}")


if __name__ == "__main__":
    main()
