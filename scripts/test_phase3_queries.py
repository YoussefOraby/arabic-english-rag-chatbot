"""Run multiple queries against the stored pyramids chunks."""
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

QUERIES = [
    "What is the height of the Great Pyramid?",
    "When was the pyramid built?",
    "What is inside the pyramid?",
]

# Initialize store
embedder = Embedder(settings.embeddings)
store = ChromaStore(settings.vector_store, embedder)

# Load and store if empty
if store.count_chunks() == 0:
    chunks_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "pyramids_chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Chunk(**d) for d in data]
    store.clear()
    store.add_chunks(chunks)
    print(f"Stored {store.count_chunks()} chunks\n")
else:
    print(f"Store already has {store.count_chunks()} chunks\n")

for query in QUERIES:
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    results = store.similarity_search(query, k=3)
    for i, r in enumerate(results):
        print(f"\n  Result {i+1} (page {r.chunk.page_num}, score={r.score:.4f})")
        print(f"  {r.chunk.text[:250]}")
    print()
