"""Draw the scaling figure: Spark ALS speedup, and its time against single-threaded NumPy.

Run from the repository root: `uv run python report/figures/scaling.py`.
It reads only committed result files, so anyone can redraw it.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SUMMARY = Path("experiments/results/scaling/summary.json")
OUTPUT = Path("report/figures/scaling.png")
# From lscpu on the Kaggle notebook: 2 physical cores with 2 threads each.
PHYSICAL_CORES = 2

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f0efea"
SPARK_COLOUR = "#eb6834"


def style(axes: Axes, title: str, cores: list[int]) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=11)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY_INK, length=0)
    axes.set_xticks(cores)
    axes.set_xlabel("Spark cores (local mode, one machine)", color=SECONDARY_INK)
    axes.set_xlim(0.7, max(cores) + 0.5)
    axes.axvspan(PHYSICAL_CORES, max(cores) + 0.5, color=SHADE, linewidth=0, zorder=0)
    axes.text(
        PHYSICAL_CORES + 0.08,
        0.02,
        "beyond 2 physical cores:\nthreads share a core",
        transform=axes.get_xaxis_transform(),
        color=MUTED,
        fontsize=8,
    )


def label_end(axes: Axes, x: float, y: float, text: str) -> None:
    axes.annotate(
        text,
        (x, y),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=SECONDARY_INK,
        fontsize=9,
    )


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    spark: list[dict[str, Any]] = [r for r in summary["results"] if r["engine"] == "spark"]
    (numpy,) = [r for r in summary["results"] if r["engine"] == "numpy"]
    cores = [r["cores"] for r in spark]
    figure, (speed, seconds) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    speedups = [r["speedup"] for r in spark]
    speed.plot(cores, cores, color=MUTED, linewidth=1.5, linestyle="--")
    label_end(speed, cores[-1], cores[-1], "perfect scaling")
    speed.plot(cores, speedups, color=SPARK_COLOUR, linewidth=2, marker="o", markersize=7)
    for x, y in zip(cores, speedups, strict=True):
        speed.annotate(
            f"{y:.2f}x",
            (x, y),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
    label_end(speed, cores[-1], speedups[-1], "Spark ALS")
    style(speed, "Speedup over Spark on 1 core", cores)
    speed.set_ylim(0, max(cores) + 0.3)

    medians = [r["total_seconds_median"] for r in spark]
    low = [m - r["total_seconds_min"] for m, r in zip(medians, spark, strict=True)]
    high = [r["total_seconds_max"] - m for m, r in zip(medians, spark, strict=True)]
    seconds.axhspan(
        numpy["total_seconds_min"], numpy["total_seconds_max"], color=MUTED, alpha=0.2, linewidth=0
    )
    seconds.axhline(
        numpy["total_seconds_median"], color=SECONDARY_INK, linewidth=1.5, linestyle="--"
    )
    seconds.annotate(
        "NumPy ALS, 1 thread",
        (1.0, numpy["total_seconds_median"]),
        xytext=(0, -8),
        textcoords="offset points",
        va="top",
        color=SECONDARY_INK,
        fontsize=9,
    )
    seconds.errorbar(
        cores,
        medians,
        yerr=[low, high],
        color=SPARK_COLOUR,
        linewidth=2,
        marker="o",
        markersize=7,
        capsize=4,
    )
    label_end(seconds, cores[-1], medians[-1], "Spark ALS")
    style(seconds, "Whole-job seconds: read, fit and copy factors", cores)
    seconds.set_ylim(0, max(r["total_seconds_max"] for r in spark) * 1.1)

    figure.text(
        0.01,
        0.01,
        f"Netflix Prize, 12 months before August 2005, {summary['rows'] / 1e6:.1f}M ratings. "
        "Kaggle CPU notebook: AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB.\n"
        "Points: median of 3 timed runs after a warm-up. Bars and band: fastest to slowest run. "
        f"Spark beats NumPy only on 4 threads, so COST is {summary['cost']}.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
