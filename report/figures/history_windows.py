"""Draw the headline figure: quality and training time against months of history.

Run from the repository root: `uv run python report/figures/history_windows.py`.
It reads only committed result files, so anyone can redraw it.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

RESULTS = Path("experiments/results")
OUTPUT = Path("report/figures/history_windows.png")
WINDOWS = ["1", "3", "6", "12", "24", "all"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed order: each model keeps its colour in both panels.
MODELS = {
    "history-two-tower": ("Two-Tower", "one Tesla T4 GPU", "#2a78d6"),
    "history-spark-als": ("Spark ALS", "4-core CPU, Spark local mode", "#eb6834"),
    "history-popularity": ("Popularity", "not timed", "#1baf7a"),
}


def load_rows(name: str) -> dict[str, dict[str, Any]]:
    summary = json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8"))
    return {row["window"]: row for row in summary["results"]}


def style(axes: Axes, title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=11)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY_INK, length=0)


def draw_series(
    axes: Axes, rows: dict[str, dict[str, Any]], key: str, label: str, colour: str
) -> None:
    positions = range(len(WINDOWS))
    means = [rows[w][f"{key}_mean"] for w in WINDOWS]
    axes.fill_between(
        positions,
        [rows[w][f"{key}_min"] for w in WINDOWS],
        [rows[w][f"{key}_max"] for w in WINDOWS],
        color=colour,
        alpha=0.15,
        linewidth=0,
    )
    axes.plot(positions, means, color=colour, linewidth=2, marker="o", markersize=6, label=label)
    axes.annotate(
        label.split(" (")[0],
        (len(WINDOWS) - 1, means[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=SECONDARY_INK,
        fontsize=9,
    )


def main() -> None:
    rows = {name: load_rows(name) for name in MODELS}
    figure, (quality, cost) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    for name, (label, _hardware, colour) in MODELS.items():
        draw_series(quality, rows[name], "ndcg", label, colour)
    for name, (label, hardware, colour) in MODELS.items():
        if name != "history-popularity":
            draw_series(cost, rows[name], "fit_seconds", f"{label} ({hardware})", colour)

    any_rows = rows["history-spark-als"]
    ticks = [f"{w}\n{any_rows[w]['train_rows_mean'] / 1e6:.0f}M rows" for w in WINDOWS]
    for axes, title, legend_corner in (
        (quality, "Quality: NDCG@10 on the following month", "lower left"),
        (cost, "Cost: training seconds (hardware differs, compare within a model)", "upper left"),
    ):
        style(axes, title)
        axes.set_xticks(range(len(WINDOWS)), ticks)
        axes.set_xlabel("Months of history before the cutoff", color=SECONDARY_INK)
        axes.set_xlim(-0.3, len(WINDOWS) - 0.3)
        axes.set_ylim(bottom=0)
        axes.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc=legend_corner)

    figure.text(
        0.01,
        0.01,
        "Netflix Prize, test months August to October 2005. Lines: mean over 3 test months and "
        "2 seeds (popularity has no seed). Bands: lowest to highest run, wide mainly because some\n"
        "test months are easier than others. Within each test month, Two-Tower scores above both "
        "other models at every window.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
