"""Embed Arabic chunks into a separate ChromaDB collection and test retrieval."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.config import settings, VectorStoreConfig
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.pdf.chunker import Chunk

# 1. Load chunks
chunks_path = Path("D:/RAG/data/processed/valeo_ar_chunks.json")
with open(chunks_path, encoding="utf-8") as f:
    data = json.load(f)
chunks = [Chunk(**d) for d in data]
print(f"Loaded {len(chunks)} chunks")

# 2. Create embedder
embedder = Embedder(settings.embeddings)

# 3. Create a SEPARATE ChromaDB collection (does not touch "documents" collection)
ar_config = VectorStoreConfig(
    persist_directory=settings.vector_store.persist_directory,
    collection_name="valeo_ar",
    distance_metric=settings.vector_store.distance_metric,
)
store = ChromaStore(ar_config, embedder)

# Clear any previous data in this collection and re-embed
store.clear()
store.add_chunks(chunks)
print(f"Stored {store.count_chunks()} chunks in collection 'valeo_ar'")
print(f"  (English pyramids remain untouched in 'documents' collection)\n")

# 4. Test 3 Arabic queries
queries = [
    "ما هي قيم شركة فاليو؟",
    "ما هو سلوك الأعمال المقبول؟",
    "كيف تتعامل الشركة مع الفساد؟",
]

for q in queries:
    print(f"{'='*60}")
    print(f"Query: {q}")
    print(f"{'='*60}")
    results = store.similarity_search(q, k=3)
    for i, r in enumerate(results):
        print(f"\n  Result {i+1} (page {r.chunk.page_num}, score={r.score:.4f})")
        print(f"  {r.chunk.text[:300]}")
    print()
