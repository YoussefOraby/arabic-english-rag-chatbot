"""Phase 3: Comprehensive RAG Evaluation Engine.

Usage:
    python evaluation/evaluate.py --run-name "baseline-v1" [--verbose]

Options:
    --questions PATH   Path to golden questions JSON (default: golden_questions.json)
    --run-name NAME    Name for this evaluation run (default: auto-timestamp)
    --output DIR       Output directory (default: evaluation/runs/)
    --chat-url URL     Chatbot API URL (default: http://localhost:8000/chat)
    --timeout SEC      Timeout per question in seconds (default: 120)
    --verbose          Print per-question results to stdout

Output:
    runs/{run-name}/results.json   — full results
    runs/{run-name}/report.md      — human-readable report
    runs/{run-name}/summary.csv    — CSV summary
    runs/latest -> symlink to latest run
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



# ── config ──────────────────────────────────────────────────────────────
DEFAULT_QUESTIONS = Path(__file__).parent / "golden_questions.json"
DEFAULT_OUTPUT = Path(__file__).parent / "runs"
DEFAULT_CHAT_URL = "http://localhost:8000/query"
DEFAULT_TIMEOUT = 120
SEMANTIC_THRESHOLD = 0.6
KEYWORD_THRESHOLD = 1.0  # fraction of required keywords that must be present


# ── question loading ────────────────────────────────────────────────────
def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)
    for q in qs:
        q.setdefault("required_keywords", [])
        q.setdefault("forbidden_keywords", [])
        q.setdefault("expected_pages", [])
        q.setdefault("expected_document", None)
    return qs


# ── API call ────────────────────────────────────────────────────────────
def ask_chatbot(question: str, url: str, timeout: int) -> dict:
    """Call the chatbot API and return parsed JSON response."""
    payload = json.dumps({
        "question": question,
        "top_k": 4,
        "history": [],
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode()
        return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        return {"error": f"HTTP {e.code}: {body}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ── answer helpers ──────────────────────────────────────────────────────
def extract_answer_text(response: dict) -> str:
    """Extract the answer text from chatbot response."""
    if "error" in response:
        return response["error"]
    answer = response.get("answer") or response.get("response") or response.get("message") or ""
    return str(answer).strip()


def extract_citations(answer: str) -> list[int]:
    """Extract page citations like [page 3], [page:3], [3], [page 3-5]."""
    parts = re.findall(r'\[(?:page\s*:?\s*)?(\d+)(?:\s*-\s*\d+)?\]', answer, re.IGNORECASE)
    return [int(p) for p in parts]


def is_insufficient_data(answer: str) -> bool:
    """Check if the answer signals insufficient data."""
    if not answer.strip():
        return True
    patterns = [
        "insufficient data",
        "cannot verify",
        "not found in the provided",
        "not available in the uploaded",
        "not available in the documents",
        "does not contain",
        "no information",
        "i couldn't find",
        "i can't find",
        "not mentioned",
        "not included",
        "not present",
        "i don't have",
        "unable to answer",
        "cannot answer",
        "the provided documents do not",
        "the uploaded documents do not",
        "based on the provided",
        "based on the uploaded",
    ]
    lower = answer.lower().strip()
    return any(p in lower for p in patterns)


# ── evaluation logic ────────────────────────────────────────────────────
def evaluate_answer(
    question_id: str,
    question_text: str,
    answer_text: str,
    q: dict,
) -> dict:
    """Evaluate a single answer against expected criteria."""
    result: dict[str, Any] = {
        "id": question_id,
        "question": question_text,
        "answer": answer_text,
        "failures": [],
        "failure_category": None,
        "keyword_score": 0.0,
        "citations": extract_citations(answer_text),
        "insufficient_data": is_insufficient_data(answer_text),
        "language": q.get("language"),
        "question_type": q.get("question_type"),
        "group": q.get("group"),
        "should_answer": q.get("should_answer", True),
    }

    should_answer = q.get("should_answer", True)
    required = q.get("required_keywords", [])
    forbidden = q.get("forbidden_keywords", [])
    expected_pages = q.get("expected_pages", [])
    expected_doc = q.get("expected_document", None)
    expected_summary = q.get("expected_answer_summary", "")

    # ── check answerability ──────────────────────────────────────────
    answered = not result["insufficient_data"]

    if should_answer and not answered:
        result["failures"].append("FALSE_NEGATIVE")
        result["failure_category"] = "FALSE_NEGATIVE"

    if not should_answer and answered:
        result["failures"].append("FALSE_POSITIVE")
        result["failure_category"] = "FALSE_POSITIVE"

    # ── keyword evaluation ──────────────────────────────────────────
    if should_answer and required:
        lower_answer = answer_text.lower()
        matched = sum(1 for kw in required if kw.lower() in lower_answer)
        score = matched / len(required)
        result["keyword_score"] = round(score, 4)
        if score < KEYWORD_THRESHOLD:
            result["failures"].append("MISSING_KEYWORD")
            result.setdefault("missing_keywords", [])
            for kw in required:
                if kw.lower() not in lower_answer:
                    result["missing_keywords"].append(kw)
    elif should_answer:
        result["keyword_score"] = 1.0

    # ── forbidden keywords ──────────────────────────────────────────
    if should_answer and forbidden:
        lower_answer = answer_text.lower()
        found_forbidden = [kw for kw in forbidden if kw.lower() in lower_answer]
        if found_forbidden:
            result["failures"].append("FORBIDDEN_KEYWORD")
            result["forbidden_found"] = found_forbidden

    # ── citation / page verification (diagnostic only) ──────────────
    if should_answer and answered and expected_pages and result["citations"]:
        cited = result["citations"]
        overlap = set(cited) & set(expected_pages)
        result["citation_detail"] = {
            "expected_pages": expected_pages,
            "actual_citations": cited,
            "page_match": bool(overlap),
        }

    # ── deterministic keyword OR ────────────────────────────────────
    if "expected_keywords" in q and should_answer and answered:
        lower_answer = answer_text.lower()
        expected_kws = q["expected_keywords"]
        matched_any = any(kw.lower() in lower_answer for kw in expected_kws)
        if not matched_any:
            result["failures"].append("MISSING_EXPECTED_KEYWORD")
            if result["failure_category"] is None:
                result["failure_category"] = "MISSING_KEYWORD"

    # ── category fallback ───────────────────────────────────────────
    if result["failure_category"] is None and result["failures"]:
        result["failure_category"] = result["failures"][0]

    # ── overall pass ────────────────────────────────────────────────
    result["passed"] = len(result["failures"]) == 0

    # ── latency placeholder (filled by runner) ──────────────────────
    result.setdefault("latency_seconds", None)
    result.setdefault("source_page_accuracy", None)
    result.setdefault("citations_all_verified", None)

    return result


# ── grouping helpers ────────────────────────────────────────────────────
def group_key(q: dict) -> str:
    return q.get("group", q.get("question_type", "other"))


def language_key(q: dict) -> str:
    return q.get("language", "unknown")


# ── report generation ───────────────────────────────────────────────────
def generate_report(results: list[dict], questions: list[dict], run_name: str, run_dir: Path):
    """Write report.md and summary.csv to run_dir."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rate = passed / total if total else 0.0

    # failure categories
    cat_counts: dict[str, int] = {}
    for r in results:
        for f in r.get("failures", []):
            cat_counts[f] = cat_counts.get(f, 0) + 1

    # group breakdown
    groups: dict[str, list[dict]] = {}
    for q in questions:
        g = group_key(q)
        groups.setdefault(g, []).append(q)

    group_stats: dict[str, dict] = {}
    for gname, gqs in groups.items():
        gids = {q["id"] for q in gqs}
        gres = [r for r in results if r["id"] in gids]
        gt = len(gres)
        gp = sum(1 for r in gres if r["passed"])
        group_stats[gname] = {"total": gt, "passed": gp, "rate": round(gp / gt, 3) if gt else 0}

    # language breakdown
    langs: dict[str, list[str]] = {}
    for q in questions:
        key = q.get("language", "unknown")
        langs.setdefault(key, []).append(q["id"])

    lang_stats: dict[str, dict] = {}
    for lname, lids in langs.items():
        lset = set(lids)
        lres = [r for r in results if r["id"] in lset]
        lt = len(lres)
        lp = sum(1 for r in lres if r["passed"])
        lang_stats[lname] = {"total": lt, "passed": lp, "rate": round(lp / lt, 3) if lt else 0}

    # type breakdown
    types: dict[str, list[str]] = {}
    for q in questions:
        key = q.get("question_type", "other")
        types.setdefault(key, []).append(q["id"])

    type_stats: dict[str, dict] = {}
    for tname, tids in types.items():
        tset = set(tids)
        tres = [r for r in results if r["id"] in tset]
        tt = len(tres)
        tp = sum(1 for r in tres if r["passed"])
        type_stats[tname] = {"total": tt, "passed": tp, "rate": round(tp / tt, 3) if tt else 0}

    # ── write report.md ──
    report_path = run_dir / "report.md"
    lines = [
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
        lines.append("| Failure | Count |")
        lines.append("|---------|-------|")
        for fname, fcnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {fname} | {fcnt} |")
    else:
        lines.append("No failures.")

    lines += [
        "",
        "---",
        "## Breakdown by Group",
        "",
        "| Group | Passed / Total | Rate |",
        "|-------|---------------|------|",
    ]
    for gname in sorted(group_stats):
        s = group_stats[gname]
        lines.append(f"| {gname} | {s['passed']}/{s['total']} | {s['rate']:.1%} |")

    lines += [
        "",
        "---",
        "## Breakdown by Language",
        "",
        "| Language | Passed / Total | Rate |",
        "|----------|---------------|------|",
    ]
    for lname in sorted(lang_stats):
        s = lang_stats[lname]
        lines.append(f"| {lname} | {s['passed']}/{s['total']} | {s['rate']:.1%} |")

    lines += [
        "",
        "---",
        "## Breakdown by Question Type",
        "",
        "| Type | Passed / Total | Rate |",
        "|------|---------------|------|",
    ]
    for tname in sorted(type_stats):
        s = type_stats[tname]
        lines.append(f"| {tname} | {s['passed']}/{s['total']} | {s['rate']:.1%} |")

    lines += [
        "",
        "---",
        "## Detailed Results",
        "",
        "| ID | Pass | Failures | Answer Preview | Latency (s) |",
        "|----|------|----------|---------------|-------------|",
    ]
    for r in sorted(results, key=lambda x: x["id"]):
        fails = ", ".join(r.get("failures", [])) or "-"
        preview = r["answer"][:80].replace("\n", " ") + ("..." if len(r["answer"]) > 80 else "")
        latency = r.get("latency_seconds", "N/A")
        latency_str = f"{latency:.1f}" if isinstance(latency, (int, float)) else str(latency)
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"| {r['id']} | {status} | {fails} | {preview} | {latency_str} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report:  {report_path}")

    # ── write summary.csv ──
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "passed", "failures", "failure_category",
            "keyword_score", "latency_seconds", "insufficient_data",
            "answer_preview",
        ])
        for r in sorted(results, key=lambda x: x["id"]):
            writer.writerow([
                r["id"],
                "PASS" if r["passed"] else "FAIL",
                "; ".join(r.get("failures", [])),
                r.get("failure_category", ""),
                r.get("keyword_score", ""),
                r.get("latency_seconds", ""),
                r.get("insufficient_data", ""),
                r["answer"][:120].replace("\n", " "),
            ])
    print(f"  CSV:     {csv_path}")

    return group_stats, lang_stats, type_stats


def write_results_json(results: list[dict], questions: list[dict], run_name: str, run_dir: Path, metadata: dict):
    """Write full results.json."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    # aggregate group stats
    groups: dict[str, dict] = {}
    for q in questions:
        g = group_key(q)
        gid = q["id"]
        r = next((x for x in results if x["id"] == gid), None)
        if r is None:
            continue
        groups.setdefault(g, {"total": 0, "passed": 0})
        groups[g]["total"] += 1
        if r["passed"]:
            groups[g]["passed"] += 1

    payload = {
        "metadata": {
            "run_name": run_name,
            "date": datetime.now(timezone.utc).isoformat(),
            "total_questions": total,
            "passed": passed,
            "overall_pass_rate": round(passed / total, 4) if total else 0,
            **metadata,
        },
        "group_breakdown": groups,
        "results": results,
    }
    path = run_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Results: {path}")


# ── comparison with previous runs ───────────────────────────────────────
def compare_with_previous(run_dir: Path, results: list[dict]):
    """Compare with the most recent previous run in the same output directory."""
    parent = run_dir.parent
    all_runs = sorted(
        [d for d in parent.iterdir() if d.is_dir() and d.name != run_dir.name],
        key=lambda d: d.stat().st_mtime,
    )
    if not all_runs:
        return

    prev_dir = all_runs[-1]
    prev_path = prev_dir / "results.json"
    if not prev_path.exists():
        return

    with open(prev_path, encoding="utf-8") as f:
        prev_data = json.load(f)

    prev_results = {r["id"]: r for r in prev_data.get("results", [])}
    curr_results = {r["id"]: r for r in results}

    deltas = []
    for qid, curr in curr_results.items():
        prev = prev_results.get(qid)
        if prev is None:
            continue
        curr_pass = curr["passed"]
        prev_pass = prev.get("passed", False)
        if curr_pass and not prev_pass:
            deltas.append((qid, "IMPROVED"))
        elif not curr_pass and prev_pass:
            deltas.append((qid, "REGRESSED"))

    if deltas:
        delta_path = run_dir / "deltas.md"
        lines = [
            f"# Delta vs {prev_dir.name}",
            "",
            f"**Improved:** {sum(1 for _, s in deltas if s == 'IMPROVED')}",
            f"**Regressed:** {sum(1 for _, s in deltas if s == 'REGRESSED')}",
            "",
            "| Question ID | Change |",
            "|-------------|--------|",
        ]
        for qid, change in sorted(deltas):
            lines.append(f"| {qid} | {change} |")
        with open(delta_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Deltas:  {delta_path}")


# ── main ────────────────────────────────────────────────────────────────
def main():
    # fix Windows console for Unicode (only in CLI mode)
    if sys.platform == "win32" and not any("pytest" in a for a in sys.argv):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phase 3: RAG Evaluation Engine")
    parser.add_argument("--questions", "-q", default=str(DEFAULT_QUESTIONS), help="Golden questions JSON")
    parser.add_argument("--run-name", "-n", default=None, help="Name for this run (default: auto)")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL, help="Chatbot API URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per question (seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-question results")
    args = parser.parse_args()

    # resolve paths
    questions_path = Path(args.questions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # run name
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # load questions
    questions = load_questions(questions_path)
    print(f"\nLoaded {len(questions)} questions from {questions_path}")

    metadata = {
        "questions_file": str(questions_path),
        "chat_url": args.chat_url,
        "timeout": args.timeout,
    }

    # evaluate each question
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question_text[:70]}...", end=" ")

        start = time.time()
        response = ask_chatbot(question_text, args.chat_url, args.timeout)
        elapsed = time.time() - start

        answer_text = extract_answer_text(response)
        result = evaluate_answer(qid, question_text, answer_text, q)
        result["latency_seconds"] = round(elapsed, 2)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        fails = ", ".join(result.get("failures", []))
        print(f"{status}  ({elapsed:.1f}s)")
        if args.verbose and not result["passed"]:
            print(f"    Answer: {answer_text[:120]}...")
            print(f"    Failures: {fails}")

    # reports
    generate_report(results, questions, run_name, run_dir)
    write_results_json(results, questions, run_name, run_dir, metadata)

    # comparison
    compare_with_previous(run_dir, results)

    # summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"  Run:        {run_name}")
    print(f"  Passed:     {passed}/{total}  ({passed/total:.1%})")
    print(f"  Failures:   {total - passed}")
    print(f"  Output:     {run_dir}")
    print(f"{'='*60}\n")

    # exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
