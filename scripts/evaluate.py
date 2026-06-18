"""Evaluate the RAG system against a golden question set.

Usage:
    python scripts/evaluate.py                          # default mode (hybrid)
    python scripts/evaluate.py --mode semantic          # dense-only
    python scripts/evaluate.py --mode hybrid             # BM25 + dense (default)
    python scripts/evaluate.py --mode hybrid+reranker    # hybrid + cross-encoder
    python scripts/evaluate.py --max-questions 5         # quick smoke test
    python scripts/evaluate.py --semantic-threshold 0.75 # custom threshold
"""

import argparse
import csv
import json
import logging
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Keyword matching ────────────────────────────────────────────────


def keyword_match_score(answer: str, expected_summary: str) -> float:
    expected_tokens = expected_summary.lower().split()
    answer_tokens = answer.lower().split()
    expected_set = set(expected_tokens)
    answer_set = set(answer_tokens)
    if not expected_set:
        return 1.0
    if not answer_set:
        return 0.0
    intersection = expected_set & answer_set
    return len(intersection) / len(expected_set)


# ── Semantic similarity ─────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_semantic_score(answer: str, expected_summary: str, embedder: Embedder) -> float:
    vecs = embedder.embed_texts([answer, expected_summary])
    return round(cosine_similarity(vecs[0], vecs[1]), 4)


# ── Arabic-English utility ──────────────────────────────────────────

_ARABIC_DIACRITICS_TR = str.maketrans({
    '\u064b': '', '\u064c': '', '\u064d': '', '\u064e': '',
    '\u064f': '', '\u0650': '', '\u0651': '', '\u0652': '', '\u0640': '',
})


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify alef/yeh/teh."""
    text = text.translate(_ARABIC_DIACRITICS_TR)
    text = text.replace('\u0623', '\u0627').replace('\u0625', '\u0627').replace('\u0622', '\u0627')
    text = text.replace('\u0649', '\u064a')
    text = text.replace('\u0629', '\u0647')
    return text


_ARABIC_ENGLISH_EQUIVALENTS: dict[str, list[str]] = {
    "RAG": ["RAG", "راغ", "راج", "راغي"],
    "Naive RAG": ["Naive RAG", "راغ العادي", "راج العادي", "راغي العادي"],
    "Advanced RAG": ["Advanced RAG", "راغ المتقدم", "راج المتقدم", "راغي المتقدم"],
    "Modular RAG": ["Modular RAG", "راغ المعياري", "نظام راغ متكامل", "راغي المعياري"],
    "pre-retrieval": ["pre-retrieval", "ما قبل الاسترجاع"],
    "post-retrieval": ["post-retrieval", "ما بعد الاسترجاع"],
    "challenges": ["challenges", "التحديات"],
    "retrieval": ["retrieval", "استرجاع"],
    "generation": ["generation", "توليد"],
    "evaluation": ["evaluation", "تقييم"],
    "metrics": ["metrics", "مقاييس"],
    "benchmarks": ["benchmarks", "معايير"],
    "hallucination": ["hallucination", "هلوسة"],
    "missing content": ["missing content", "محتوى مفقود"],
    "dense retrieval": ["dense retrieval", "استرجاع كثيف"],
    "hybrid": ["hybrid", "هجين"],
    "reranking": ["reranking", "إعادة ترتيب"],
    "BM25": ["BM25"],
}

def _keyword_matches_text(keyword: str, normalized_answer: str) -> bool:
    """Check if a single keyword (or variant/translation) appears in the normalized answer."""
    kw = keyword.lower().strip()

    # 1. Direct substring match
    if kw in normalized_answer:
        return True

    # 2. Arabic-English equivalents
    for equiv in _ARABIC_ENGLISH_EQUIVALENTS.get(keyword, []):
        if equiv.lower() in normalized_answer:
            return True

    # 3. Morphological: -e suffix words → -ing, -ed, -al, -ion, -ation
    #    e.g. retrieve→retrieval, generate→generation
    if kw.endswith("e") and len(kw) > 3:
        base = kw[:-1]
        for ending in ["ing", "ed", "al", "er", "ion", "ation"]:
            if base + ending in normalized_answer:
                return True

    # 4. Morphological: -able/-ible → -ability/-ibility
    #    e.g. flexible→flexibility
    if kw.endswith("able") and len(kw) > 5:
        if kw[:-4] + "ability" in normalized_answer:
            return True
    if kw.endswith("ible") and len(kw) > 5:
        if kw[:-4] + "ibility" in normalized_answer:
            return True

    # 5. Simple inflectional endings
    for ending in ["ing", "ed", "s", "es"]:
        if kw + ending in normalized_answer:
            return True

    return False


def check_required_keywords(answer: str, required: list[str] | None) -> bool | None:
    if not required:
        return None
    normalized = normalize_arabic(answer.lower())
    for kw in required:
        if not _keyword_matches_text(kw, normalized):
            return False
    return True


# ── Source / citation checks ────────────────────────────────────────


def extract_returned_pages(result: dict) -> set[int]:
    pages: set[int] = set()
    for src, p_list in result.get("source_documents", {}).items():
        if isinstance(p_list, list):
            pages.update(p_list)
    for c in result.get("citations", []):
        for p in c.get("pages", []):
            pages.add(p)
    return pages


def check_source_page_accuracy(
    result: dict, expected_pages: list[int], should_answer: bool
) -> bool | None:
    if not should_answer:
        return None
    if not expected_pages:
        return None
    returned_pages = extract_returned_pages(result)
    return any(ep in returned_pages for ep in expected_pages)


_REFUSAL_PATTERNS = [
    "insufficient data",
    "i don't know based on the provided documents",
    "i don't know from the provided documents",
    "not mentioned in the provided documents",
    "no mention of",
    "لا أعرف بناءً على المستندات",
    "غير مذكور في المستندات",
    "لا توجد معلومات",
]


def check_unsupported_accuracy(result: dict) -> bool:
    if result.get("insufficient_data"):
        return True
    answer = result.get("answer", "").lower()
    for pattern in _REFUSAL_PATTERNS:
        if pattern in answer:
            citations = result.get("citations", [])
            if citations:
                return False
            return True
    return False


def compute_citation_rate(result: dict) -> float | None:
    citations = result.get("citations", [])
    if not citations:
        return None
    verified = sum(1 for c in citations if c.get("verified"))
    return verified / len(citations)


def citations_all_verified(result: dict) -> bool | None:
    """Return None if no citations (condition vacuously true), else True/False."""
    citations = result.get("citations", [])
    if not citations:
        return None
    return all(c.get("verified") for c in citations)


# ── Pass / fail ─────────────────────────────────────────────────────


def per_question_pass(
    semantic_pass: bool,
    source_accuracy: bool | None,
    citations_ok: bool | None,
    required_kw_ok: bool | None,
    should_answer: bool,
    unsupported_ok: bool,
) -> bool:
    if not should_answer:
        return unsupported_ok
    if not semantic_pass:
        return False
    if source_accuracy is not None and not source_accuracy:
        return False
    if citations_ok is not None and not citations_ok:
        return False
    if required_kw_ok is not None and not required_kw_ok:
        return False
    return True


# ── Per-question evaluation ─────────────────────────────────────────


def evaluate_question(
    q: dict, chain: RAGChain, embedder: Embedder, semantic_threshold: float
) -> dict:
    question = q["question"]
    start = time.perf_counter()
    result = chain.query(question)
    elapsed = time.perf_counter() - start

    answer = result.get("answer", "")

    kw_score = keyword_match_score(answer, q["expected_answer_summary"])
    sem_score = compute_semantic_score(answer, q["expected_answer_summary"], embedder)
    sem_pass = sem_score >= semantic_threshold

    source_accuracy = check_source_page_accuracy(result, q["expected_pages"], q["should_answer"])
    citation_rate = compute_citation_rate(result)
    citations_ok = citations_all_verified(result)
    unsupported_ok = check_unsupported_accuracy(result) if not q["should_answer"] else None
    required_kw_ok = check_required_keywords(answer, q.get("required_keywords"))

    passed = per_question_pass(
        sem_pass,
        source_accuracy,
        citations_ok,
        required_kw_ok,
        q["should_answer"],
        unsupported_ok or False,
    )

    return {
        "id": q["id"],
        "question": question,
        "language": q["language"],
        "question_type": q["question_type"],
        "should_answer": q["should_answer"],
        "answer": answer,
        "expected_answer_summary": q["expected_answer_summary"],
        "expected_pages": q["expected_pages"],
        "expected_document": q["expected_document"],
        "answer_keyword_score": round(kw_score, 4),
        "semantic_answer_score": sem_score,
        "semantic_answer_pass": sem_pass,
        "source_page_accuracy": source_accuracy,
        "citation_verified_rate": round(citation_rate, 4) if citation_rate is not None else None,
        "citations_all_verified": citations_ok,
        "required_keywords": q.get("required_keywords"),
        "required_keywords_pass": required_kw_ok,
        "unsupported_handled": unsupported_ok,
        "latency_seconds": round(elapsed, 2),
        "insufficient_data": result.get("insufficient_data", False),
        "passed": passed,
    }


# ── Aggregation ─────────────────────────────────────────────────────


def compute_aggregate(results: list[dict]) -> dict:
    n = len(results)

    passed = sum(1 for r in results if r["passed"])
    answerable = [r for r in results if r["should_answer"]]
    kw_scores = [r["answer_keyword_score"] for r in answerable]
    sem_scores = [r["semantic_answer_score"] for r in answerable]
    source_ok = [
        r["source_page_accuracy"] for r in answerable if r["source_page_accuracy"] is not None
    ]
    citation_rates = [
        r["citation_verified_rate"] for r in results if r["citation_verified_rate"] is not None
    ]
    unsupported_results = [r for r in results if not r["should_answer"]]
    unsupported_ok_count = sum(1 for r in unsupported_results if r["unsupported_handled"])
    latencies = [r["latency_seconds"] for r in results]

    return {
        "total_questions": n,
        "passed": passed,
        "overall_pass_rate": round(passed / n, 4) if n else 0.0,
        "avg_answer_keyword_match": round(sum(kw_scores) / len(kw_scores), 4)
        if kw_scores
        else None,
        "avg_semantic_similarity": round(sum(sem_scores) / len(sem_scores), 4)
        if sem_scores
        else None,
        "source_page_accuracy_pct": round(sum(source_ok) / len(source_ok), 4)
        if source_ok
        else None,
        "citation_verification_rate": round(sum(citation_rates) / len(citation_rates), 4)
        if citation_rates
        else None,
        "unsupported_accuracy": round(unsupported_ok_count / len(unsupported_results), 4)
        if unsupported_results
        else None,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


# ── Output writers ──────────────────────────────────────────────────


def write_report(
    aggregate: dict,
    results: list[dict],
    mode: str,
    threshold: float,
    report_path: Path,
) -> None:
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Mode:** {mode}")
    lines.append(f"**Semantic threshold:** {threshold}")
    lines.append(f"**Model:** {settings.llm.ollama.model}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total questions | {aggregate['total_questions']} |")
    lines.append(f"| Passed | {aggregate['passed']} |")
    lines.append(f"| Overall pass rate | {aggregate['overall_pass_rate']:.0%} |")
    lines.append(f"| Avg keyword match | {aggregate['avg_answer_keyword_match'] or 'N/A'} |")
    lines.append(f"| Avg semantic similarity | {aggregate['avg_semantic_similarity'] or 'N/A'} |")
    lines.append(f"| Source page accuracy | {aggregate['source_page_accuracy_pct'] or 'N/A'} |")
    lines.append(
        f"| Citation verification rate | {aggregate['citation_verification_rate'] or 'N/A'} |"
    )
    lines.append(f"| Unsupported accuracy | {aggregate['unsupported_accuracy'] or 'N/A'} |")
    lines.append(f"| Avg latency | {aggregate['avg_latency_seconds']}s |")
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    cols = ["ID", "Type", "Lang", "Pass", "Sem", "KW", "Pages", "Cit", "ReqKW", "Latency"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in results:
        sem_str = f"{r['semantic_answer_score']:.2f}" if r["should_answer"] else "—"
        kw_str = f"{r['answer_keyword_score']:.2f}" if r["should_answer"] else "—"
        if r["source_page_accuracy"] is True:
            pages_str = "Y"
        elif r["source_page_accuracy"] is None:
            pages_str = "—"
        else:
            pages_str = "N"
        cit_str = (
            f"{r['citation_verified_rate']:.2f}" if r["citation_verified_rate"] is not None else "—"
        )
        rk_str: str = "—"
        if r.get("required_keywords_pass") is not None:
            rk_str = "Y" if r["required_keywords_pass"] else "N"
        pass_char = "P" if r["passed"] else "F"
        lines.append(
            f"| {r['id']} | {r['question_type']} | {r['language']} "
            f"| {pass_char} | {sem_str} | {kw_str} "
            f"| {pages_str} | {cit_str} | {rk_str} | {r['latency_seconds']}s |"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "id",
        "question",
        "language",
        "question_type",
        "should_answer",
        "answer",
        "expected_answer_summary",
        "expected_pages",
        "expected_document",
        "answer_keyword_score",
        "semantic_answer_score",
        "semantic_answer_pass",
        "source_page_accuracy",
        "citation_verified_rate",
        "citations_all_verified",
        "required_keywords",
        "required_keywords_pass",
        "unsupported_handled",
        "latency_seconds",
        "insufficient_data",
        "passed",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG system against a golden question set."
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "hybrid", "hybrid+reranker"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for quick smoke test)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.70,
        help="Semantic similarity threshold for pass (default: 0.70)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    questions_path = base_dir / "evaluation" / "questions.json"
    results_path = base_dir / "evaluation" / "results.json"
    csv_path = base_dir / "evaluation" / "results.csv"
    report_path = base_dir / "evaluation" / "report.md"

    if not questions_path.exists():
        logger.error("Questions file not found: %s", questions_path)
        sys.exit(1)

    questions = load_questions(questions_path)
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"Evaluation mode: {args.mode}")
    print(f"Semantic threshold: {args.semantic_threshold}")
    print(f"Questions: {len(questions)}")

    settings.retrieval.enable_hybrid = args.mode in ("hybrid", "hybrid+reranker")
    settings.retrieval.enable_reranker = args.mode == "hybrid+reranker"

    print("Initializing embedder...")
    embedder = Embedder(settings.embeddings)
    print("Initializing vector store...")
    store = ChromaStore(settings.vector_store, embedder)

    chunk_count = store.count_chunks()
    if chunk_count == 0:
        logger.error("No chunks found in vector store. Ingest documents first via the UI or API.")
        sys.exit(1)
    print(f"Chunks in store: {chunk_count}")

    print("Checking Ollama...")
    llm = OllamaLLM(
        model=settings.llm.ollama.model,
        base_url=settings.llm.ollama.base_url,
        temperature=settings.llm.ollama.temperature,
        num_ctx=settings.llm.ollama.num_ctx,
        timeout=settings.llm.ollama.timeout,
    )
    if not llm.is_available():
        logger.error("Ollama is not reachable at %s", settings.llm.ollama.base_url)
        sys.exit(1)
    print(f"LLM: {llm.name}")

    print("Building RAG chain...")
    chain = RAGChain(store=store, llm=llm)
    chain.rebuild_index()

    print("\nRunning evaluation...\n")
    results = []
    for i, q in enumerate(questions, 1):
        try:
            r = evaluate_question(q, chain, embedder, args.semantic_threshold)
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{i}/{len(questions)}] {status} {r['id']} ({r['latency_seconds']}s)")
        except Exception as e:
            logger.error("Failed on %s: %s", q["id"], e)
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "language": q["language"],
                    "question_type": q["question_type"],
                    "should_answer": q["should_answer"],
                    "answer": "",
                    "expected_answer_summary": q["expected_answer_summary"],
                    "expected_pages": q["expected_pages"],
                    "expected_document": q["expected_document"],
                    "answer_keyword_score": 0.0,
                    "semantic_answer_score": 0.0,
                    "semantic_answer_pass": False,
                    "source_page_accuracy": None,
                    "citation_verified_rate": None,
                    "citations_all_verified": None,
                    "required_keywords": q.get("required_keywords"),
                    "required_keywords_pass": None,
                    "unsupported_handled": None,
                    "latency_seconds": 0.0,
                    "insufficient_data": False,
                    "passed": False,
                    "error": str(e),
                }
            )
            print(f"  [{i}/{len(questions)}] FAIL {r['id']} ERROR: {e}")

    aggregate = compute_aggregate(results)

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "mode": args.mode,
            "semantic_threshold": args.semantic_threshold,
            "model": settings.llm.ollama.model,
            "total_questions": len(questions),
        },
        "aggregate": aggregate,
        "results": results,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    write_csv(results, csv_path)
    write_report(aggregate, results, args.mode, args.semantic_threshold, report_path)

    print(f"\n{'=' * 50}")
    print(
        f"Pass rate:   {aggregate['overall_pass_rate']:.0%} ({aggregate['passed']}/{aggregate['total_questions']})"
    )
    print(f"Avg semantic: {aggregate['avg_semantic_similarity'] or 'N/A'}")
    print(f"Avg keyword:  {aggregate['avg_answer_keyword_match'] or 'N/A'}")
    print(f"Avg latency:  {aggregate['avg_latency_seconds']}s")
    print(f"{'=' * 50}")
    print("\nResults written to:")
    print(f"  {results_path}")
    print(f"  {csv_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
