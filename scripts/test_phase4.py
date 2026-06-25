"""Phase 4: Full RAG pipeline — retrieve + generate with Ollama."""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain

QUESTIONS = [
    "Who built the pyramids?",
    "What is the height of the Great Pyramid?",
    "What is inside the pyramid?",
]

# Init store (chunks already embedded from Phase 3)
embedder = Embedder(settings.embeddings)
store = ChromaStore(settings.vector_store, embedder)

chunk_count = store.count_chunks()
print(f"Store has {chunk_count} chunks\n")

# Init Ollama + RAGChain
llm = OllamaLLM(
    model=settings.llm.ollama.model,
    base_url=settings.llm.ollama.base_url,
    temperature=settings.llm.ollama.temperature,
)
chain = RAGChain(store, llm)

for q in QUESTIONS:
    print(f"{'='*60}")
    print(f"Question: {q}")
    print(f"{'='*60}")

    start = time.time()
    result = chain.query(q)
    elapsed = time.time() - start

    pages = result.get("source_documents", {})

    print(f"\nAnswer ({elapsed:.1f}s):")
    print(f"  {result['answer']}")
    print(f"\nCited pages: {pages}")
    print(f"Insufficient data: {result.get('insufficient_data', False)}")
    print()
