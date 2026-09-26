"""Plot Hard and Overall accuracy by run from results/ablation.csv.

Produces two plain bar charts in results/figures/: ablation_hard.png and
ablation_overall.png. Both use the CSV's own row order, which is
chronological. No styling beyond default matplotlib: labeled axes, a
title, a y-axis starting at zero, and the value printed on top of each
bar.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "ablation.csv"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def make_bar_chart(runs, values, ylabel, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(runs, values)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    rows = load_rows(CSV_PATH)
    runs = [row["run"] for row in rows]
    hard_pct = [float(row["hard_pct"]) for row in rows]
    overall_pct = [float(row["overall_pct"]) for row in rows]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_bar_chart(
        runs,
        hard_pct,
        ylabel="Hard accuracy (%)",
        title="Hard accuracy by run",
        out_path=FIGURES_DIR / "ablation_hard.png",
    )

    make_bar_chart(
        runs,
        overall_pct,
        ylabel="Overall accuracy (%)",
        title="Overall accuracy by run",
        out_path=FIGURES_DIR / "ablation_overall.png",
    )

    print(f"Wrote {FIGURES_DIR / 'ablation_hard.png'}")
    print(f"Wrote {FIGURES_DIR / 'ablation_overall.png'}")


if __name__ == "__main__":
    main()
