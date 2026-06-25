"""Full Arabic RAG pipeline: retrieve from valeo_ar collection + Ollama."""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.config import settings, VectorStoreConfig
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain

QUESTIONS = [
    "ما هي قيم شركة فاليو؟",
    "كيف تتعامل الشركة مع الفساد؟",
    "ما هي سياسة الشركة تجاه الرشوة؟",
]

# Load the valeo_ar collection (separate from English pyramids)
embedder = Embedder(settings.embeddings)
ar_config = VectorStoreConfig(
    persist_directory=settings.vector_store.persist_directory,
    collection_name="valeo_ar",
    distance_metric=settings.vector_store.distance_metric,
)
store = ChromaStore(ar_config, embedder)
print(f"Loaded {store.count_chunks()} chunks from 'valeo_ar' collection\n")

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
