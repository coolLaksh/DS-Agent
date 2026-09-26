"""Recompute and check the arithmetic behind results/ablation.csv.

This script exists so a reader does not have to trust the numbers in
ablation.csv on faith. For each run it does two independent checks:

1. It recomputes easy_pct and hard_pct from the raw correct/total counts
   (100 * correct / total) and compares them against the stored
   percentages. This catches typos or stale percentages that no longer
   match the counts.
2. It recomputes overall_pct from the stored easy_pct and hard_pct using
   the task-count weighting stated in the project brief:
   overall = (72 * easy_pct + 378 * hard_pct) / 450.
   This catches arithmetic errors in the overall column.

A row passes only if both checks are within a 0.01 tolerance (rounding
in the source percentages). The script exits with code 1 if any row
fails, so it can be used as a simple CI-style guard on the CSV.
"""

import csv
import sys
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "ablation.csv"
TOLERANCE = 0.01

EASY_TOTAL_WEIGHT = 72
HARD_TOTAL_WEIGHT = 378
TOTAL_WEIGHT = EASY_TOTAL_WEIGHT + HARD_TOTAL_WEIGHT  # 450


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_row(row: dict) -> dict:
    easy_pct = float(row["easy_pct"])
    hard_pct = float(row["hard_pct"])
    easy_correct = int(row["easy_correct"])
    hard_correct = int(row["hard_correct"])
    easy_total = int(row["easy_total"])
    hard_total = int(row["hard_total"])
    overall_pct = float(row["overall_pct"])

    recomputed_easy_pct = 100 * easy_correct / easy_total
    recomputed_hard_pct = 100 * hard_correct / hard_total
    recomputed_overall = (
        EASY_TOTAL_WEIGHT * easy_pct + HARD_TOTAL_WEIGHT * hard_pct
    ) / TOTAL_WEIGHT

    easy_ok = abs(recomputed_easy_pct - easy_pct) <= TOLERANCE
    hard_ok = abs(recomputed_hard_pct - hard_pct) <= TOLERANCE
    overall_ok = abs(recomputed_overall - overall_pct) <= TOLERANCE

    return {
        "run": row["run"],
        "easy_pct": easy_pct,
        "recomputed_easy_pct": recomputed_easy_pct,
        "easy_ok": easy_ok,
        "hard_pct": hard_pct,
        "recomputed_hard_pct": recomputed_hard_pct,
        "hard_ok": hard_ok,
        "overall_pct": overall_pct,
        "recomputed_overall": recomputed_overall,
        "overall_ok": overall_ok,
        "pass": easy_ok and hard_ok and overall_ok,
    }


def main() -> int:
    rows = load_rows(CSV_PATH)
    results = [check_row(row) for row in rows]

    header = (
        f"{'run':<24}{'easy%':>8}{'->':>4}{'recomp':>8}"
        f"{'hard%':>8}{'->':>4}{'recomp':>8}"
        f"{'overall%':>10}{'->':>4}{'recomp':>8}  {'result'}"
    )
    print(header)
    print("-" * len(header))

    any_failed = False
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            any_failed = True
        print(
            f"{r['run']:<24}"
            f"{r['easy_pct']:>8.2f}{'->':>4}{r['recomputed_easy_pct']:>8.2f}"
            f"{r['hard_pct']:>8.2f}{'->':>4}{r['recomputed_hard_pct']:>8.2f}"
            f"{r['overall_pct']:>10.2f}{'->':>4}{r['recomputed_overall']:>8.2f}"
            f"  {status}"
        )

    print()
    if any_failed:
        print("One or more rows did not reconcile within tolerance "
              f"({TOLERANCE}). See FAIL rows above.")
        return 1

    print(f"All {len(results)} rows reconcile within tolerance ({TOLERANCE}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
