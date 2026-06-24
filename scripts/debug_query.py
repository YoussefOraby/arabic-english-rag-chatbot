"""Debug a single RAG query — prints every pipeline stage.

Usage:
    python scripts/debug_query.py --question "What is Naive RAG?"
    python scripts/debug_query.py --question "..." --full              # full chunk text
    python scripts/debug_query.py --question "..." --document-id <id>  # filter by doc
    python scripts/debug_query.py --question "..." --mode semantic     # disable hybrid
    python scripts/debug_query.py --question "..." --top-k 8
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain
from src.rag.prompts import format_chunks_for_context
from src.utils.helpers import is_arabic_text


def safe(s: str, maxlen: int = 120) -> str:
    return s[:maxlen].replace("\n", "\\n") if s else ""


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:60]


def save_debug_output(output: dict, question: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "evaluation" / "debug"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_q = sanitize_filename(question)[:40]
    path = base / f"{ts}_{safe_q}.md"
    lines = []
    lines.append(f"# Debug Query: {question}")
    lines.append(f"**Time:** {output['timestamp']}")
    lines.append(f"**Mode:** {output['mode']}")
    lines.append(f"**Top-K:** {output['top_k']}")
    lines.append(f"**Document ID:** {output['document_id']}")
    lines.append("")
    lines.append("## Intent Detection")
    lines.append(f"- Intent: `{output['intent']}`")
    lines.append(f"- Expanded query: `{output['expanded_query']}`")
    lines.append("")
    lines.append("## Short Document Check")
    lines.append(f"- is_short_document: `{output['is_short_document']}`")
    lines.append(f"- Total chunks in store: `{output['total_chunks']}`")
    lines.append("")
    lines.append("## Retrieved Chunks")
    for i, c in enumerate(output["chunks"], 1):
        lines.append(f"### [{i}] {c['filename']} | page: {c['page']} | score: {c['score']:.4f}")
        lines.append(f"**ID:** `{c['chunk_id']}`")
        lines.append(f"**Document ID:** `{c['document_id']}`")
        lines.append("")
        if output.get("full_chunks"):
            lines.append("```")
            lines.append(c["text"])
            lines.append("```")
        else:
            lines.append(f"```\n{c['text'][:300]}\n...\n```")
        lines.append("")
    lines.append("## Source Pages")
    lines.append(str(output["source_documents"]))
    lines.append("")
    lines.append("## Context Sent to LLM")
    lines.append("```")
    lines.append(output["context"])
    lines.append("```")
    lines.append("")
    lines.append("## System Prompt")
    lines.append("```")
    lines.append(output["system_prompt"])
    lines.append("```")
    lines.append("")
    lines.append("## Final Prompt Sent to LLM")
    lines.append("```")
    lines.append(output["prompt"])
    lines.append("```")
    lines.append("")
    lines.append("## Raw LLM Answer")
    lines.append("```")
    lines.append(output["raw_answer"])
    lines.append("```")
    lines.append("")
    lines.append("## Parsed Citations")
    for i, cit in enumerate(output.get("citations", []), 1):
        lines.append(f"- [{i}] pages={cit.get('pages')}, verified={cit.get('verified')}, text_before={safe(cit.get('text_before',''), 60)}")
    lines.append("")
    lines.append("## Final Answer (after post-processing)")
    lines.append("```")
    lines.append(output["answer"])
    lines.append("```")
    lines.append("")
    lines.append("## insufficient_data")
    lines.append(f"`{output['insufficient_data']}`")
    lines.append("")
    lines.append("## Suspected Failure Reason")
    lines.append(output.get("failure_reason", "No clear failure detected"))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def diagnose_question(
    chain: RAGChain,
    embedder: Embedder,
    question: str,
    document_id: str | None,
    full: bool,
    top_k: int,
    mode: str,
    threshold: float,
) -> dict:
    result = chain.query(question, document_id=document_id)
    intent = chain._detect_query_intent(question)
    expanded = chain._expand_query(question, intent)

    # Build the context string exactly as sent to LLM
    chunk_data = []
    for r in result.get("chunks", []):
        chunk_data.append({
            "filename": r.chunk.source_file,
            "page": r.chunk.page_num,
            "score": r.score,
            "chunk_id": r.chunk.chunk_id,
            "document_id": r.chunk.document_id,
            "text": r.chunk.text,
        })

    # Rebuild the prompt to inspect it
    chunks_list = result.get("chunks", [])
    context = format_chunks_for_context(chunks_list)
    from src.rag.prompts import get_system_prompt
    system_prompt = get_system_prompt(question, custom_template=settings.rag.system_prompt_template)
    from src.rag.prompts import format_history
    history_str = format_history([], max_pairs=settings.rag.max_history_pairs)
    prompt_parts = [system_prompt]
    if history_str:
        prompt_parts.append(history_str)
    prompt_parts.append(f"Context:\n{context}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("Answer (with citations):")
    prompt_text = "\n\n".join(prompt_parts)

    # Determine failure reason
    failure_reason = "PASS"
    answer = result.get("answer", "")
    insufficient = result.get("insufficient_data", False)
    citations = result.get("citations", [])

    if insufficient and result.get("chunks", []):
        failure_reason = "FALSE_INSUFFICIENT — chunks exist but max_score < threshold or short-doc fallback rejected"
    elif insufficient and not result.get("chunks", []):
        failure_reason = "RETRIEVAL_FAILURE — no chunks returned (all below score_threshold or store empty)"
    elif insufficient:
        failure_reason = "FALSE_INSUFFICIENT — unknown cause"
    elif not answer.strip():
        failure_reason = "LLM_IGNORED_CONTEXT — empty answer despite chunks"
    else:
        # Check if any citation is unverified
        unverified = [c for c in citations if not c.get("verified")]
        if unverified:
            failure_reason = f"CITATION_FAILURE — {len(unverified)}/{len(citations)} citations unverified"
        # Check for refusal patterns in answer
        refusal_patterns = [
            "insufficient data",
            "i don't know based on the provided documents",
            "not mentioned in the provided documents",
            "no mention of",
            "لا أعرف بناءً على المستندات",
            "غير مذكور في المستندات",
            "لا توجد معلومات",
        ]
        if any(p in answer.lower() for p in refusal_patterns):
            failure_reason = "LLM_IGNORED_CONTEXT — LLM said insufficient despite chunks containing evidence"

    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "top_k": top_k,
        "document_id": document_id,
        "intent": intent,
        "expanded_query": expanded,
        "is_short_document": chain._is_short_document(document_id) if hasattr(chain, "_is_short_document") else "N/A",
        "total_chunks": chain.store.count_chunks(),
        "chunks": chunk_data,
        "full_chunks": full,
        "source_documents": result.get("source_documents", {}),
        "context": context,
        "system_prompt": system_prompt,
        "prompt": prompt_text,
        "raw_answer": result.get("answer", ""),
        "citations": citations,
        "answer": result.get("answer", ""),
        "insufficient_data": insufficient,
        "failure_reason": failure_reason,
    }

    if full:
        output["chunks"] = chunk_data

    return output


def print_report(output: dict, full: bool) -> None:
    print("=" * 70)
    print(f"Question: {output['question']}")
    print(f"Intent:   {output['intent']}")
    print(f"Expanded: {output['expanded_query']}")
    print(f"Short doc? {output['is_short_document']}  |  Total chunks: {output['total_chunks']}")
    print("-" * 70)

    print(f"\n--- Retrieved Chunks ({len(output['chunks'])}) ---")
    for i, c in enumerate(output["chunks"], 1):
        print(f"  [{i}] {c['filename']} p{c['page']} score={c['score']:.4f} id={c['chunk_id']}")
        if full:
            print(f"       {c['text']}")
        else:
            print(f"       {safe(c['text'], 200)}")
    print(f"\nSources: {output['source_documents']}")
    print("-" * 70)

    print(f"\n--- Context ({len(output['context'])} chars) ---")
    if full:
        print(output['context'])
    else:
        print(output['context'][:500] + "\n...")
    print("-" * 70)

    print(f"\n--- Raw LLM Answer ---")
    print(output['raw_answer'])
    print(f"\n--- Final Answer ---")
    print(output['answer'])
    print(f"\n--- Citations ---")
    for i, cit in enumerate(output.get("citations", []), 1):
        verified = "✓" if cit.get("verified") else "✗"
        print(f"  [{i}] pages={cit.get('pages')} {verified} '{safe(cit.get('text_before',''), 80)}'")
    print(f"\n--- insufficient_data: {output['insufficient_data']} ---")
    print(f"\n--- Failure: {output['failure_reason']} ---")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Debug a single RAG query")
    parser.add_argument("--question", type=str, required=True, help="Question to debug")
    parser.add_argument("--full", action="store_true", help="Print full chunks and context")
    parser.add_argument("--document-id", type=str, default=None, help="Filter by document ID")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k")
    parser.add_argument("--mode", choices=["semantic", "hybrid"], default=None, help="Retrieval mode")
    parser.add_argument("--score-threshold", type=float, default=None, help="Override score threshold")
    parser.add_argument("--save", action="store_true", help="Save debug output to file")
    args = parser.parse_args()

    # Apply overrides
    if args.top_k is not None:
        settings.retrieval.top_k = args.top_k
    mode = args.mode or ("hybrid" if settings.retrieval.enable_hybrid else "semantic")
    if args.mode == "semantic":
        settings.retrieval.enable_hybrid = False
    elif args.mode == "hybrid":
        settings.retrieval.enable_hybrid = True
    if args.score_threshold is not None:
        settings.retrieval.score_threshold = args.score_threshold

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    print("Initializing...")
    embedder = Embedder(settings.embeddings)
    store = ChromaStore(settings.vector_store, embedder)
    count = store.count_chunks()
    if count == 0:
        print("ERROR: No chunks found in vector store. Ingest documents first.")
        sys.exit(1)
    print(f"Store: {count} chunks")

    print("Connecting to Ollama...")
    llm = OllamaLLM(
        model=settings.llm.ollama.model,
        base_url=settings.llm.ollama.base_url,
        temperature=settings.llm.ollama.temperature,
        num_ctx=settings.llm.ollama.num_ctx,
        timeout=settings.llm.ollama.timeout,
    )
    chain = RAGChain(store=store, llm=llm)
    chain.rebuild_index()

    print(f"\nQuerying: {args.question}\n")
    output = diagnose_question(
        chain, embedder, args.question,
        document_id=args.document_id,
        full=args.full,
        top_k=settings.retrieval.top_k,
        mode=mode,
        threshold=settings.retrieval.score_threshold,
    )
    output["question"] = args.question
    print_report(output, args.full)

    if args.save:
        path = save_debug_output(output, args.question)
        print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
