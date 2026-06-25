"""Phase 3: embed, store, retrieve, and answer a question about pyramids.pdf."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.pdf.chunker import Chunk

# 1. Load chunks from JSON
chunks_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "pyramids_chunks.json"
with open(chunks_path, encoding="utf-8") as f:
    data = json.load(f)

chunks = [Chunk(**d) for d in data]
print(f"Loaded {len(chunks)} chunks")

# 2. Create embedder and store
embedder = Embedder(settings.embeddings)
store = ChromaStore(settings.vector_store, embedder)
store.clear()
store.add_chunks(chunks)
print(f"Stored {store.count_chunks()} chunks in ChromaDB")

# 3. Search
query = "Who built the pyramids?"
results = store.similarity_search(query, k=3)
print(f"\nQuery: {query}")
print(f"Top {len(results)} results:\n")
for i, r in enumerate(results):
    print(f"--- Result {i+1} (page {r.chunk.page_num}, score={r.score:.4f}) ---")
    print(r.chunk.text[:300])
    print()
