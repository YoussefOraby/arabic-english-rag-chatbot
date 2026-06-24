"""Compare multiple evaluation runs and show trends.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --dir evaluation/runs --runs baseline-v1 baseline-v2
    python scripts/compare_runs.py --latest 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_run(run_dir: Path) -> dict | None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def get_question_result(run_data: dict, qid: str) -> dict | None:
    for r in run_data.get("results", []):
        if r["id"] == qid:
            return r
    return None


def format_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str[:19]


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("--dir", default="evaluation/runs", help="Runs directory")
    parser.add_argument("--runs", nargs="*", default=None, help="Specific run names to compare")
    parser.add_argument("--latest", type=int, default=0, help="Compare N most recent runs")
    parser.add_argument("--questions", nargs="*", default=None, help="Filter to specific question IDs")
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"Directory not found: {base}")
        return

    # collect runs
    all_runs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda d: d.stat().st_mtime,
    )

    if args.runs:
        selected = [base / name for name in args.runs if (base / name).is_dir()]
    elif args.latest:
        selected = all_runs[-args.latest:]
    else:
        selected = all_runs

    if not selected:
        print("No runs found.")
        return

    # load data
    run_data = []
    for run_dir in selected:
        data = load_run(run_dir)
        if data:
            run_data.append((run_dir.name, data))

    if not run_data:
        print("No valid run data found.")
        return

    # collect all question IDs across all runs
    all_qids: set[str] = set()
    for _, data in run_data:
        for r in data.get("results", []):
            all_qids.add(r["id"])

    if args.questions:
        all_qids = {q for q in all_qids if q in args.questions}
    all_qids = sorted(all_qids)

    # print header
    print(f"\n{'=' * 100}")
    print("  RUN COMPARISON DASHBOARD")
    print(f"{'=' * 100}\n")

    # summary table
    print(f"{'Run':<25} {'Date':<18} {'Passed':<8} {'Total':<8} {'Rate':<8} {'Avg Latency':<12}")
    print("-" * 80)
    for name, data in run_data:
        meta = data.get("metadata", {})
        date_str = format_date(meta.get("date", ""))
        passed = meta.get("passed", "?")
        total = meta.get("total_questions", "?")
        rate = f"{passed / total:.1%}" if isinstance(passed, int) and isinstance(total, int) and total else "?"
        avg_lat = meta.get("avg_latency", "")
        print(f"{name:<25} {date_str:<18} {str(passed):<8} {str(total):<8} {rate:<8} {str(avg_lat):<12}")

    # per-question comparison
    print(f"\n{'-' * 100}")
    print("  PER-QUESTION COMPARISON  (P = Pass, F = Fail, - = missing)")
    print(f"{'-' * 100}\n")

    header = f"{'Question':<28}"
    for name, _ in run_data:
        header += f" {name:<18}"
    print(header)
    print("-" * (28 + 19 * len(run_data)))

    for qid in all_qids:
        row = f"{qid:<28}"
        for _, data in run_data:
            r = get_question_result(data, qid)
            if r is None:
                row += f" {'-':<18}"
            else:
                status = "P" if r.get("passed") else "F"
                failures = ";".join(r.get("failures", []))
                label = f"{status}"
                if failures:
                    label += f"({failures})"
                row += f" {label:<18}"
        print(row)

    # failure trends
    print(f"\n{'-' * 100}")
    print("  FAILURE TRENDS  (number of questions hitting each failure category)")
    print(f"{'-' * 100}\n")

    all_cats: set[str] = set()
    for _, data in run_data:
        for r in data.get("results", []):
            for f in r.get("failures", []):
                all_cats.add(f)
    all_cats = sorted(all_cats)

    header = f"{'Category':<25}"
    for name, _ in run_data:
        header += f" {name:<12}"
    print(header)
    print("-" * (25 + 13 * len(run_data)))

    for cat in all_cats:
        row = f"{cat:<25}"
        for _, data in run_data:
            count = sum(1 for r in data.get("results", []) if cat in r.get("failures", []))
            row += f" {str(count):<12}"
        print(row)

    # group breakdown trends
    print(f"\n{'-' * 100}")
    print("  GROUP BREAKDOWN TRENDS  (pass rate per group)")
    print(f"{'-' * 100}\n")

    all_groups: set[str] = set()
    for _, data in run_data:
        gb = data.get("group_breakdown", {})
        all_groups.update(gb.keys())
    all_groups = sorted(all_groups)

    header = f"{'Group':<25}"
    for name, _ in run_data:
        header += f" {name:<14}"
    print(header)
    print("-" * (25 + 15 * len(run_data)))

    for grp in all_groups:
        row = f"{grp:<25}"
        for _, data in run_data:
            gb = data.get("group_breakdown", {})
            info = gb.get(grp, {})
            p = info.get("passed", 0)
            t = info.get("total", 0)
            rate = f"{p}/{t}" if t else "-"
            row += f" {rate:<14}"
        print(row)

    # question type breakdown trends
    print(f"\n{'-' * 100}")
    print("  QUESTION TYPE BREAKDOWN TRENDS")
    print(f"{'-' * 100}\n")

    # manually compute from results
    all_types: set[str] = set()
    for _, data in run_data:
        for r in data.get("results", []):
            qt = r.get("question_type", "unknown")
            all_types.add(qt)
    all_types = sorted(all_types)

    header = f"{'Type':<20}"
    for name, _ in run_data:
        header += f" {name:<14}"
    print(header)
    print("-" * (20 + 15 * len(run_data)))

    for qtype in all_types:
        row = f"{qtype:<20}"
        for _, data in run_data:
            type_results = [r for r in data.get("results", []) if r.get("question_type") == qtype]
            total = len(type_results)
            passed = sum(1 for r in type_results if r.get("passed"))
            rate = f"{passed}/{total}" if total else "-"
            row += f" {rate:<14}"
        print(row)

    print(f"\n{'=' * 100}")
    print(f"  {len(run_data)} runs, {len(all_qids)} unique questions tracked")
    print(f"  Run dirs: {', '.join(name for name, _ in run_data)}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
