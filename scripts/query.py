"""Interactive query script for the RAG chatbot.
Usage: python scripts/query.py [--question "your question"] [--interactive]

Process:
  1. Embed the question
  2. Retrieve relevant chunks from ChromaDB
  3. Build prompt with context
  4. Send to Ollama
  5. Display answer with source citations
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain


def setup_rag():
    """Initialize all RAG components."""
    print("[RAG] Loading embedding model...", end=" ", flush=True)
    embedder = Embedder(settings.embeddings)
    print("done")

    print(
        f"[RAG] Connecting to ChromaDB ({settings.vector_store.persist_directory})...",
        end=" ",
        flush=True,
    )
    store = ChromaStore(settings.vector_store, embedder)
    count = store.count_chunks()
    print(f"done ({count} chunks)")

    print(f"[RAG] Connecting to Ollama ({settings.llm.ollama.model})...", end=" ", flush=True)
    llm = OllamaLLM(**settings.llm.ollama.model_dump())
    print("done")

    return RAGChain(store=store, llm=llm)


def answer_question(chain: RAGChain, question: str) -> None:
    """Ask a question and print the answer with sources."""
    safe_q = question.encode("ascii", errors="replace").decode("ascii")
    print(f"\n{'=' * 60}")
    print(f"Q: {safe_q}")
    print(f"{'=' * 60}")

    try:
        result = chain.query(question)
        answer = result["answer"]
        safe_answer = answer.encode("ascii", errors="replace").decode("ascii")
        print(f"\nA: {safe_answer}")

        print("\n--- Sources ---")
        for source, pages in result["source_documents"].items():
            page_str = ", ".join(f"p{p}" for p in pages)
            print(f"  {source}: {page_str}")

        print(f"\n--- Retrieved Chunks ({len(result['chunks'])}) ---")
        for i, r in enumerate(result["chunks"], 1):
            safe = r.chunk.text[:80].encode("ascii", errors="replace").decode("ascii")
            print(
                f"  [{i}] {r.chunk.source_file} p{r.chunk.page_num} "
                f"(score={r.score:.2f}): {safe}..."
            )

        report_path = Path("data/raw/last_answer.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Q: {question}\n\nA: {answer}\n\n")
            f.write("--- Sources ---\n")
            for source, pages in result["source_documents"].items():
                page_str = ", ".join(f"p{p}" for p in pages)
                f.write(f"  {source}: {page_str}\n")
        print(f"\n(Full answer saved to {report_path})")

    except Exception as e:
        safe = str(e).encode("ascii", errors="replace").decode("ascii")
        print(f"\n[!] Error: {safe}")


def interactive_mode(chain: RAGChain) -> None:
    """Run interactive Q&A loop."""
    print("\nInteractive mode. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        answer_question(chain, question)
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot Query CLI")
    parser.add_argument("-q", "--question", type=str, help="Single question to answer")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    chain = setup_rag()

    if args.question:
        answer_question(chain, args.question)
    elif args.interactive:
        interactive_mode(chain)
    else:
        # Default: ask a sample question
        answer_question(chain, "What is this project about?")


if __name__ == "__main__":
    main()
