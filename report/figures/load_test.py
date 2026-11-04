"""Draw the load test figure: latency and throughput against simultaneous users.

Run from the repository root: `uv run python report/figures/load_test.py`.
It reads only committed result files, so anyone can redraw it.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

SUMMARY = Path("experiments/results/load-test/summary.json")
OUTPUT = Path("report/figures/load_test.png")
P95_TARGET_MS = 100
THROUGHPUT_CEILING = 1000

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COPIES = {1: ("1 copy", "#2a78d6"), 2: ("2 copies", "#eb6834")}


def style(axes: Axes, title: str, users: list[int]) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=11)
    axes.grid(axis="y", color=GRID, linewidth=0.8, which="major")
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY_INK, length=0, which="both")
    # Steps are spaced evenly, not to scale, so the crowded high steps stay readable.
    axes.set_xticks(range(len(users)), [str(u) for u in users])
    axes.set_xlabel(
        "Simultaneous users, each clicking about once a second (steps, not to scale)",
        color=SECONDARY_INK,
    )


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = summary["steps"]
    users = sorted({step["users"] for step in steps})
    position = {u: index for index, u in enumerate(users)}
    figure, (latency, throughput) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    latency.axhline(P95_TARGET_MS, color=SECONDARY_INK, linewidth=1.2, linestyle="--")
    latency.annotate(
        "target: 100 ms",
        (0, P95_TARGET_MS),
        xytext=(0, 5),
        textcoords="offset points",
        color=SECONDARY_INK,
        fontsize=9,
    )
    throughput.plot(range(len(users)), users, color=MUTED, linewidth=1.2, linestyle="--")
    shown = [u for u in users if u <= THROUGHPUT_CEILING][-1]
    throughput.annotate(
        "if never slowed",
        (position[shown], shown),
        xytext=(6, -4),
        textcoords="offset points",
        va="top",
        color=SECONDARY_INK,
        fontsize=9,
    )

    for copies, (label, colour) in COPIES.items():
        own = sorted((s for s in steps if s["copies"] == copies), key=lambda s: s["users"])
        x = [position[s["users"]] for s in own]
        for axes, key in ((latency, "p95_ms"), (throughput, "requests_per_second")):
            axes.fill_between(
                x,
                [s[f"{key}_min"] for s in own],
                [s[f"{key}_max"] for s in own],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            axes.plot(
                x,
                [s[f"{key}_median"] for s in own],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=6,
                label=label,
            )

    style(latency, "p95 latency, milliseconds (log scale)", users)
    latency.set_yscale("log")
    latency.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style(throughput, "Requests answered per second", users)
    throughput.set_ylim(0, THROUGHPUT_CEILING)
    for axes in (latency, throughput):
        axes.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc="upper left")

    failure = {row["copies"]: row["failure_point_users"] for row in summary["by_copies"]}
    figure.text(
        0.01,
        0.01,
        "One laptop (Intel Core i7-1355U): a three-node kind cluster on Docker CPUs 0-7, k6 on "
        "CPUs 8-11. Points: median of 3 passes of 60 s. Bands: lowest to highest pass.\n"
        f"First step missing the target: {failure.get(1) or 'none'} users with 1 copy, "
        f"{failure.get(2) or 'none'} users with 2 copies. Requests time out at 5 s, so p95 "
        "stops at 5000 ms. Not a production cluster.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
