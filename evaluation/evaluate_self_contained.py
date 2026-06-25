"""Self-contained isolated evaluation runner.

Creates its own temp ChromaDB + document registry, ingests synthetic documents,
runs RAGChain queries, and evaluates results. No dependency on production data.

Usage:
    python evaluation/evaluate_self_contained.py --run-name "baseline-v2"
    python evaluation/evaluate_self_contained.py --run-name "debug" --debug-question resume-en-001
    python evaluation/evaluate_self_contained.py --run-name "paper-only" --group research_paper
"""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings, VectorStoreConfig, EmbeddingsConfig
from src.document.processor import process_document
from src.document.registry import DocumentRegistry
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.rag.chain import RAGChain

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR.parent / "tests" / "fixtures"
QUESTIONS_PATH = BASE_DIR / "golden_questions.json"
OUTPUT_DIR = BASE_DIR / "runs"
TMP_DIR = BASE_DIR / "tmp"

SYNTHETIC_DOCS: dict[str, Path] = {
    "golden_resume": FIXTURES_DIR / "golden_resume.pdf",
    "synthetic_rag_paper": FIXTURES_DIR / "synthetic_rag_paper.pdf",
    "synthetic_arabic_policy": FIXTURES_DIR / "synthetic_arabic_policy.pdf",
}

KEYWORD_THRESHOLD = 1.0


# ── question helpers ───────────────────────────────────────────────────
def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)
    for q in qs:
        q.setdefault("required_keywords", [])
        q.setdefault("forbidden_keywords", [])
        q.setdefault("expected_pages", [])
        q.setdefault("document_id", None)
    return qs


def is_insufficient_data(answer: str) -> bool:
    if not answer.strip():
        return True
    patterns = [
        "insufficient data", "cannot verify",
        "not found in the provided", "not available in the uploaded",
        "not available in the documents", "does not contain",
        "no information", "i couldn't find", "i can't find",
        "not mentioned", "not included", "not present",
        "i don't have", "unable to answer", "cannot answer",
        "the provided documents do not", "the uploaded documents do not",
        "based on the provided", "based on the uploaded",
    ]
    lower = answer.lower().strip()
    return any(p in lower for p in patterns)


def extract_citations(answer: str) -> list[int]:
    parts = re.findall(r'\[(?:page\s*:?\s*)?(\d+)(?:\s*-\s*\d+)?\]', answer, re.IGNORECASE)
    return [int(p) for p in parts]


# ── failure reason dataclass ──────────────────────────────────────────
@dataclass
class RetrievalInspection:
    document_ingested: bool = False
    chunks_retrieved: int = 0
    chunks_from_correct_doc: int = 0
    correct_doc_chunk_count: int = 0
    retrieved_pages: list[int] = field(default_factory=list)
    keywords_in_chunks: list[str] = field(default_factory=list)
    keywords_missing_from_chunks: list[str] = field(default_factory=list)
    retrieved_chunk_previews: list[dict] = field(default_factory=list)


@dataclass
class FailureDiagnosis:
    categories: list[str] = field(default_factory=list)
    raw_answer: str = ""
    final_answer: str = ""
    retrieval: RetrievalInspection = field(default_factory=RetrievalInspection)
    expected_keywords_found_in_context: bool = False
    llm_ignored_context: bool = False
    post_processing_changed: bool = False
    citation_verification_failed: bool = False
    prompt_context: str = ""


# ── evaluation logic ───────────────────────────────────────────────────
def inspect_retrieval(result: dict, expected_doc_id: str | None, required_kws: list[str]) -> RetrievalInspection:
    """Inspect what was retrieved for a query."""
    insp = RetrievalInspection()
    chunks = result.get("chunks", [])
    insp.chunks_retrieved = len(chunks)

    if expected_doc_id:
        for c in chunks:
            # c is a SearchResult with .chunk (Chunk) and .score
            chunk = c.chunk if hasattr(c, "chunk") else c.get("chunk", c)
            source = getattr(chunk, "source_file", chunk.get("source_file", "") if isinstance(chunk, dict) else "")
            if expected_doc_id in source:
                insp.chunks_from_correct_doc += 1

            page = getattr(chunk, "page_num", chunk.get("page_num", -1) if isinstance(chunk, dict) else -1)
            if page >= 0:
                insp.retrieved_pages.append(page)

            text = getattr(chunk, "text", chunk.get("text", "") if isinstance(chunk, dict) else "")
            insp.retrieved_chunk_previews.append({
                "page": page,
                "source": source,
                "text_preview": text[:200] if text else "",
            })

    # Check keywords in retrieved chunks
    all_text = " ".join(
        c.get("text_preview", "") for c in insp.retrieved_chunk_previews
    ).lower()
    for kw in required_kws:
        if kw.lower() in all_text:
            insp.keywords_in_chunks.append(kw)
        else:
            insp.keywords_missing_from_chunks.append(kw)

    return insp


def diagnose_failure(
    q: dict,
    result: dict,
    raw_answer: str,
    final_answer: str,
    chain: RAGChain | None,
) -> FailureDiagnosis:
    """Diagnose why a question failed."""
    diag = FailureDiagnosis()
    diag.raw_answer = raw_answer
    diag.final_answer = final_answer

    should = q.get("should_answer", True)
    required_kws = q.get("required_keywords", [])
    expected_doc_id = q.get("document_id")

    # Check answerability
    insufficient = is_insufficient_data(final_answer)
    if should and insufficient:
        diag.categories.append("FALSE_NEGATIVE")
    if not should and not insufficient:
        diag.categories.append("FALSE_POSITIVE")

    # Inspect retrieval
    insp = inspect_retrieval(result, expected_doc_id, required_kws)
    diag.retrieval = insp

    # Check if correct document was ingested
    if expected_doc_id and chain:
        store = chain.store
        all_chunks = store.get_all_chunks()
        insp.correct_doc_chunk_count = sum(
            1 for c in all_chunks if expected_doc_id in (
                c.source_file if hasattr(c, "source_file") else getattr(c, "source_file", str(c))
            )
        )

    # Keyword analysis
    lower_answer = final_answer.lower()
    for kw in required_kws:
        if kw.lower() not in lower_answer:
            diag.categories.append("MISSING_KEYWORD")
            break

    # Check if LLM ignored context that had the right info
    if should and required_kws and insp.keywords_in_chunks:
        # Keywords were in chunks but missing from answer → LLM ignored context
        missing_from_answer = [kw for kw in required_kws if kw.lower() not in lower_answer]
        if missing_from_answer and any(kw in insp.keywords_in_chunks for kw in missing_from_answer):
            diag.llm_ignored_context = True

    # Citation check
    forbidden = q.get("forbidden_keywords", [])
    for kw in forbidden:
        if kw.lower() in lower_answer:
            diag.categories.append("FORBIDDEN_KEYWORD")
            break

    # Citation verification (diagnostic only — no longer a primary failure)
    citations = result.get("citations", [])
    if citations:
        verified = all(c.get("verified", False) for c in citations)
        diag.citation_verification_failed = not verified

    if not diag.categories:
        diag.categories.append("PASS")

    return diag


# ── isolated environment ───────────────────────────────────────────────
def setup_isolated_environment(run_name: str) -> tuple[Path, Path]:
    """Create temporary directories for this run."""
    run_tmp = TMP_DIR / run_name
    if run_tmp.exists():
        shutil.rmtree(run_tmp)
    chroma_dir = run_tmp / "chromadb"
    db_path = run_tmp / "documents.db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir, db_path


# ── ingestion ──────────────────────────────────────────────────────────
def ingest_document(
    pdf_path: Path,
    doc_id: str,
    chroma_dir: Path,
    db_path: Path,
) -> ChromaStore:
    """Ingest a synthetic document into an isolated store. Returns the store."""
    # Override settings for isolated run
    settings.vector_store.persist_directory = str(chroma_dir)
    settings.vector_store.collection_name = f"eval_{doc_id}_{int(time.time())}"

    embedder = Embedder(settings.embeddings)
    store = ChromaStore(settings.vector_store, embedder)

    # Build chunks directly (simpler than full process_document)
    from src.pdf.ocr import extract_pages_with_fallback
    from src.pdf.chunker import chunk_pages

    # Extract
    pages = extract_pages_with_fallback(pdf_path)
    print(f"    Extracted {len(pages)} pages")

    # Chunk
    chunks = chunk_pages(
        pages,
        settings.pdf,
        source_file=pdf_path.name,
        embedder=embedder,
        pdf_path=pdf_path,
    )
    print(f"    Created {len(chunks)} chunks")

    # Tag chunks with document_id
    for c in chunks:
        c.document_id = doc_id

    # Store
    store.add_chunks(chunks)
    print(f"    Stored {len(chunks)} chunks in isolated ChromaDB ({store.count_chunks()} total)")

    return store


# ── build chain ────────────────────────────────────────────────────────
def build_chain(store: ChromaStore, retrieval_only: bool = False) -> RAGChain:
    """Build RAGChain with the given store.

    When retrieval_only is True, creates a null LLM that returns
    a placeholder so ingestion + retrieval can be tested without Ollama.
    """
    if retrieval_only:
        from src.llm.base import BaseLLM

        class NullLLM(BaseLLM):
            def invoke(self, prompt: str, **kwargs) -> str:
                return "[retrieval-only mode] No LLM call."
            def invoke_stream(self, prompt: str, **kwargs):
                yield "[retrieval-only mode] No LLM call."
            def is_available(self) -> bool: return True
            @property
            def name(self) -> str: return "NullLLM/eval"

        llm = NullLLM()
    else:
        llm = OllamaLLM(
            model=settings.llm.ollama.model,
            base_url=settings.llm.ollama.base_url,
            temperature=settings.llm.ollama.temperature,
            num_ctx=settings.llm.ollama.num_ctx,
            timeout=min(settings.llm.ollama.timeout, 60),
        )

    # Override retrieval settings for fair eval
    settings.retrieval.top_k = 4
    settings.retrieval.enable_hybrid = False
    settings.retrieval.enable_reranker = False
    settings.retrieval.score_threshold = 0.0

    chain = RAGChain(store=store, llm=llm)
    chain.rebuild_index()
    return chain


# ── query with timeout ─────────────────────────────────────────────────
def query_with_timeout(chain: RAGChain, question: str, timeout: int = 120) -> dict:
    """Execute chain.query with a hard stop on timeout."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(chain.query, question)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"answer": "TIMEOUT", "chunks": [], "citations": [], "source_documents": {}}
        except Exception as e:
            return {"answer": f"ERROR: {e}", "chunks": [], "citations": [], "source_documents": {}}


# ── debug question ─────────────────────────────────────────────────────
def debug_question(
    q: dict,
    chain: RAGChain,
    chroma_dir: Path,
) -> None:
    """Print detailed debug info for a single question."""
    qid = q["id"]
    question = q["question"]
    print(f"\n{'=' * 70}")
    print(f"  DEBUG: {qid}")
    print(f"  Question: {question}")
    print(f"{'=' * 70}")

    # Show ingested documents
    store = chain.store
    all_chunks = store.get_all_chunks()
    doc_ids = set()
    for c in all_chunks:
        doc_ids.add(getattr(c, "document_id", "?"))
    print(f"\n  Ingested document IDs: {doc_ids}")
    print(f"  Total chunks in store: {len(all_chunks)}")

    # Run query
    result = chain.query(question)
    answer = result.get("answer", "")
    chunks = result.get("chunks", [])
    citations = result.get("citations", [])
    source_docs = result.get("source_documents", {})
    insufficient = result.get("insufficient_data", False)

    print(f"\n  Insufficient data: {insufficient}")
    print(f"\n  Retrieved chunks ({len(chunks)}):")
    for i, c in enumerate(chunks):
        chunk = c.chunk if hasattr(c, "chunk") else c
        text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        source = chunk.source_file if hasattr(chunk, "source_file") else chunk.get("source_file", "?")
        page = chunk.page_num if hasattr(chunk, "page_num") else chunk.get("page_num", "?")
        score = c.score if hasattr(c, "score") else 0
        print(f"    [{i}] page={page}, score={score:.4f}, source={source}")
        print(f"        text: {text[:150]}...")

    print(f"\n  Source documents: {dict(source_docs)}")
    print(f"\n  Citations: {citations}")

    # Show expected keywords and whether they're in chunks
    required_kws = q.get("required_keywords", [])
    all_chunk_text = " ".join(
        (c.chunk.text if hasattr(c, "chunk") else "")
        for c in chunks
    ).lower()
    print(f"\n  Required keywords: {required_kws}")
    for kw in required_kws:
        in_chunks = kw.lower() in all_chunk_text
        in_answer = kw.lower() in answer.lower()
        print(f"    '{kw}': in_chunks={in_chunks}, in_answer={in_answer}")

    print(f"\n  Raw answer:")
    print(f"    {answer}")
    print(f"\n  {'=' * 70}\n")


# ── evaluation runner ──────────────────────────────────────────────────
def evaluate(
    questions: list[dict],
    chroma_dir: Path,
    db_path: Path,
    run_name: str,
    verbose: bool = False,
    debug_qid: str | None = None,
    group_filter: str | None = None,
    limit: int | None = None,
    retrieval_only: bool = False,
) -> list[dict]:
    """Run evaluation: ingest documents, query, diagnose.

    Parameters
    ----------
    questions : list[dict]
        Golden questions to evaluate.
    chroma_dir : Path
        Temp ChromaDB directory.
    db_path : Path
        Temp document registry path.
    run_name : str
        Name for this run.
    verbose : bool
        Print extra details on failures.
    debug_qid : str | None
        Single question ID for deep debug output.
    group_filter : str | None
        Only evaluate questions from this group.
    limit : int | None
        Max questions to evaluate (0 or None = all).
    retrieval_only : bool
        Skip LLM calls; only test ingestion + retrieval.
    """
    results: list[dict] = []
    chain: RAGChain | None = None
    current_doc_id: str | None = None
    questions_processed = 0

    # Group questions by document_id
    from collections import defaultdict
    by_doc: dict[str | None, list[dict]] = defaultdict(list)
    for q in questions:
        if group_filter and q.get("group") != group_filter:
            continue
        by_doc[q.get("document_id")].append(q)

    # Sort: None (unsupported) last, named docs first
    doc_order = sorted(
        by_doc.keys(),
        key=lambda x: (x is None, x or ""),
    )

    for doc_id in doc_order:
        if limit and questions_processed >= limit:
            print(f"\n  [LIMIT] Reached --limit={limit}, stopping evaluation.")
            break

        if debug_qid:
            # When debugging a specific question, only process its document
            qids_for_doc = {q["id"] for q in by_doc[doc_id]}
            if debug_qid not in qids_for_doc:
                continue
        doc_qs = by_doc[doc_id]
        if not doc_qs:
            continue

        if doc_id is None:
            # Unsupported questions — always use a fresh empty store
            print(f"\n  --- Step: Unsupported questions (no document) ---")
            print(f"    Creating empty vector store at {chroma_dir} ...", end=" ", flush=True)
            settings.vector_store.persist_directory = str(chroma_dir)
            settings.vector_store.collection_name = f"eval_empty_{int(time.time())}"
            embedder = Embedder(settings.embeddings)
            store = ChromaStore(settings.vector_store, embedder)
            chain = build_chain(store, retrieval_only=retrieval_only)
            print("done")
        else:
            # Ingest document
            pdf_path = SYNTHETIC_DOCS.get(doc_id)
            if not pdf_path or not pdf_path.exists():
                print(f"\n  [SKIP] Document '{doc_id}' not found at {pdf_path}")
                for q in doc_qs:
                    results.append({
                        "id": q["id"],
                        "question": q["question"],
                        "answer": "",
                        "passed": False,
                        "failures": ["DOCUMENT_NOT_FOUND"],
                        "diagnosis": None,
                    })
                continue

            print(f"\n  --- Step: Ingesting {pdf_path.name} (id={doc_id}) ---", flush=True)
            store = ingest_document(pdf_path, doc_id, chroma_dir, db_path)
            print(f"    Building chain (retrieval_only={retrieval_only}) ...", end=" ", flush=True)
            chain = build_chain(store, retrieval_only=retrieval_only)
            print("done")
            current_doc_id = doc_id

        print(f"  --- Step: Running {len(doc_qs)} question(s) for {doc_id or 'unsupported'} ---")

        # Run questions for this document
        for q in doc_qs:
            if limit and questions_processed >= limit:
                print(f"\n  [LIMIT] Reached --limit={limit}, stopping.")
                break

            qid = q["id"]
            question = q["question"]

            # Debug mode for a single question
            if debug_qid and qid != debug_qid:
                continue

            print(f"  [{qid}] {question[:70]}...", end=" ", flush=True)
            start = time.perf_counter()

            try:
                print("(retrieving chunks)", end=" ", flush=True)
                result = query_with_timeout(chain, question)
                elapsed = time.perf_counter() - start
                raw_answer = result.get("answer", "")
                if retrieval_only:
                    print("(retrieval-only, skipping LLM)", end=" ", flush=True)
            except Exception as e:
                elapsed = time.perf_counter() - start
                raw_answer = f"ERROR: {e}"
                result = {"answer": raw_answer, "chunks": [], "citations": [], "source_documents": {}}

            final_answer = raw_answer
            insufficient = is_insufficient_data(final_answer)

            # Diagnose
            diag = diagnose_failure(q, result, raw_answer, final_answer, chain)

            passed = diag.categories == ["PASS"] or (
                not q.get("should_answer", True) and "FALSE_POSITIVE" not in diag.categories
            )

            questions_processed += 1
            status = "PASS" if passed else "FAIL"
            fails = ", ".join(diag.categories)
            print(f"{status} ({elapsed:.1f}s)  {fails}")
            if verbose and not passed:
                print(f"    answer: {final_answer[:120]}...")

            # Debug mode — print full details
            if debug_qid and qid == debug_qid:
                debug_question(q, chain, chroma_dir)

            results.append({
                "id": qid,
                "question": question,
                "group": q.get("group"),
                "language": q.get("language"),
                "question_type": q.get("question_type"),
                "should_answer": q.get("should_answer", True),
                "expected_document": q.get("expected_document"),
                "document_id": q.get("document_id"),
                "expected_pages": q.get("expected_pages"),
                "required_keywords": q.get("required_keywords"),
                "forbidden_keywords": q.get("forbidden_keywords"),
                "answer": final_answer,
                "passed": passed,
                "failures": diag.categories,
                "latency_seconds": round(elapsed, 2),
                "insufficient_data": insufficient,
                "diagnosis": {
                    "raw_answer": diag.raw_answer,
                    "document_ingested": diag.retrieval.document_ingested,
                    "chunks_retrieved": diag.retrieval.chunks_retrieved,
                    "chunks_from_correct_doc": diag.retrieval.chunks_from_correct_doc,
                    "correct_doc_chunk_count": diag.retrieval.correct_doc_chunk_count,
                    "retrieved_pages": diag.retrieval.retrieved_pages,
                    "keywords_in_chunks": diag.retrieval.keywords_in_chunks,
                    "keywords_missing_from_chunks": diag.retrieval.keywords_missing_from_chunks,
                    "llm_ignored_context": diag.llm_ignored_context,
                    "citation_verification_failed": diag.citation_verification_failed,
                    "retrieved_chunk_previews": diag.retrieval.retrieved_chunk_previews,
                },
            })

        if debug_qid:
            break  # Only process the debug question

    return results


# ── report generation ──────────────────────────────────────────────────
def generate_report(results: list[dict], run_name: str, run_dir: Path):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rate = passed / total if total else 0.0

    # Group stats
    from collections import defaultdict
    groups: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        g = r.get("group", "unknown")
        groups[g]["total"] += 1
        if r["passed"]:
            groups[g]["passed"] += 1

    # Failure category counts
    cat_counts: dict[str, int] = defaultdict(int)
    for r in results:
        for f in r.get("failures", []):
            cat_counts[f] += 1

    # Report
    report_lines = [
        f"# Evaluation Report: {run_name}",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Total Questions:** {total}",
        f"**Passed:** {passed} / {total}  (**{rate:.1%}**)",
        "",
        "---",
        "## Failure Categories",
        "",
    ]
    if cat_counts:
        report_lines.append("| Failure | Count |")
        report_lines.append("|---------|-------|")
        for fname, fcnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"| {fname} | {fcnt} |")
    else:
        report_lines.append("No failures.")

    report_lines += [
        "",
        "---",
        "## Group Breakdown",
        "",
        "| Group | Passed / Total | Rate |",
        "|-------|---------------|------|",
    ]
    for gname in sorted(groups):
        s = groups[gname]
        report_lines.append(f"| {gname} | {s['passed']}/{s['total']} | {s['passed']/s['total']:.1%} |")

    # Deep diagnosis summary
    false_neg_count = sum(1 for r in results if "FALSE_NEGATIVE" in r.get("failures", []))
    false_pos_count = sum(1 for r in results if "FALSE_POSITIVE" in r.get("failures", []))
    missing_kw_count = sum(1 for r in results if "MISSING_KEYWORD" in r.get("failures", []))
    ignored_context = sum(
        1 for r in results
        if r.get("diagnosis", {}).get("llm_ignored_context")
    )
    citation_unverified = sum(
        1 for r in results if r.get("diagnosis", {}).get("citation_verification_failed")
    )

    report_lines += [
        "",
        "---",
        "## Deep Diagnosis",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| FALSE_NEGATIVE (said insufficient data, should have answered) | {false_neg_count} |",
        f"| FALSE_POSITIVE (answered, should have refused) | {false_pos_count} |",
        f"| MISSING_KEYWORD (correct doc retrieved but missing expected terms) | {missing_kw_count} |",
        f"| LLM ignored context (keywords in chunks but not in answer) | {ignored_context} |",
        f"| Citation verification failed | {citation_unverified} |",
    ]

    report_lines += [
        "",
        "---",
        "## Detailed Results",
        "",
        "| ID | Pass | Failures | Chunks | CorrectDoc% | KeywordsInChunks | LLMIgnored | Answer Preview |",
        "|----|------|----------|--------|-------------|------------------|------------|---------------|",
    ]
    for r in sorted(results, key=lambda x: x["id"]):
        d = r.get("diagnosis", {})
        fails = ", ".join(r.get("failures", [])) or "-"
        chunks = d.get("chunks_retrieved", 0)
        doc_pct = ""
        correct = d.get("chunks_from_correct_doc", 0)
        total_chunks = d.get("chunks_retrieved", 0)
        if total_chunks:
            doc_pct = f"{correct}/{total_chunks}"
        kw_in = ", ".join(d.get("keywords_in_chunks", []))[:40] or "-"
        ignored = "Y" if d.get("llm_ignored_context") else ""
        preview = r["answer"][:60].replace("\n", " ") + ("..." if len(r["answer"]) > 60 else "")
        status = "PASS" if r["passed"] else "FAIL"
        report_lines.append(
            f"| {r['id']} | {status} | {fails} | {chunks} | {doc_pct} | {kw_in} | {ignored} | {preview} |"
        )

    report_path = run_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report:  {report_path}")

    # CSV
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "group", "language", "type", "passed", "failures",
            "chunks_retrieved", "llm_ignored_context",
        ])
        for r in sorted(results, key=lambda x: x["id"]):
            d = r.get("diagnosis", {})
            w.writerow([
                r["id"], r.get("group"), r.get("language"), r.get("question_type"),
                "PASS" if r["passed"] else "FAIL",
                "; ".join(r.get("failures", [])),
                d.get("chunks_retrieved", 0),
                d.get("llm_ignored_context", False),
            ])
    print(f"  CSV:     {csv_path}")

    # Full results JSON
    payload = {
        "metadata": {
            "run_name": run_name,
            "date": datetime.now(timezone.utc).isoformat(),
            "total_questions": total,
            "passed": passed,
            "overall_pass_rate": round(rate, 4),
            "false_negative": false_neg_count,
            "false_positive": false_pos_count,
            "missing_keyword": missing_kw_count,
            "llm_ignored_context": ignored_context,
            "citation_unverified": citation_unverified,
        },
        "group_breakdown": {g: dict(s) for g, s in groups.items()},
        "results": results,
    }
    json_path = run_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  JSON:    {json_path}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Self-contained RAG evaluation")
    parser.add_argument("--run-name", "-n", default=None, help="Run name (default: auto)")
    parser.add_argument("--questions", "-q", default=str(QUESTIONS_PATH), help="Golden questions JSON")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--debug-question", default=None, help="Debug a specific question ID")
    parser.add_argument("--group", default=None, help="Filter to a specific group")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp files after run")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate (safety guard)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM, test ingestion+retrieval only")
    parser.add_argument("--smoke-test", action="store_true", help="Shorthand: --limit 3 --retrieval-only")
    args = parser.parse_args()

    # If --smoke-test, override limit + retrieval-only
    if args.smoke_test:
        args.limit = 3
        args.retrieval_only = True

    # Fix Windows console
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # Load questions
    questions = load_questions(Path(args.questions))
    print(f"Loaded {len(questions)} questions")

    # Setup isolated environment
    chroma_dir, db_path = setup_isolated_environment(run_name)
    print(f"Isolated stores: chroma={chroma_dir}, db={db_path}")

    # Run evaluation
    print(f"Settings: limit={args.limit}, retrieval_only={args.retrieval_only}, group={args.group}, debug={args.debug_question}")
    results = evaluate(
        questions,
        chroma_dir,
        db_path,
        run_name,
        verbose=args.verbose,
        debug_qid=args.debug_question,
        group_filter=args.group,
        limit=args.limit,
        retrieval_only=args.retrieval_only,
    )

    if not results:
        print("No questions matched. Exiting.")
        return

    # Generate reports
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    generate_report(results, run_name, run_dir)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 60}")
    print(f"  Run:        {run_name}")
    print(f"  Passed:     {passed}/{total}  ({passed/total:.1%})")
    print(f"  Failures:   {total - passed}")
    print(f"  Output:     {run_dir}")
    print(f"{'=' * 60}\n")

    # Cleanup unless requested (ignore locked files on Windows)
    if not args.keep_tmp:
        import shutil
        run_tmp = TMP_DIR / run_name
        if run_tmp.exists():
            for attempt in range(3):
                try:
                    shutil.rmtree(run_tmp)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
