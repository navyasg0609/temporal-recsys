"""Measure how much storage layout changes the time to train a model.

The read benchmark timed reads alone. This times the whole training job, because a
saving on the read only matters in proportion to the read's share of the job.
"""

import itertools
import json
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import FLAT_DIR
from temporal_recsys.evaluation.split import window_before
from temporal_recsys.experiments.baseline import SUMMARY_FILE, load_finished
from temporal_recsys.experiments.spark_als import train_and_time
from temporal_recsys.models.spark_als import SparkAlsConfig

LOGGER = logging.getLogger(__name__)

MEASUREMENTS_FILE = "measurements.jsonl"


@dataclass(frozen=True)
class LatencyPlan:
    id: str
    cutoff: date
    window_months: tuple[int, ...]
    layouts: tuple[str, ...]
    test_months: int
    repetitions: int
    warmup_passes: int
    order_seed: int
    n_films: int
    model: SparkAlsConfig

    def __post_init__(self) -> None:
        if FLAT_DIR not in self.layouts:
            raise ValueError("Savings are measured against the flat layout")


def load_plan(path: Path) -> LatencyPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design = plan["design"]
    parameters = design["fixed_parameters"]
    return LatencyPlan(
        id=plan["id"],
        cutoff=date.fromisoformat(design["cutoff"]),
        window_months=tuple(design["window_months"]),
        layouts=tuple(design["layouts"]),
        test_months=design["test_months"],
        repetitions=design["repetitions"],
        warmup_passes=design["warmup_passes"],
        order_seed=design["order_seed"],
        n_films=design["n_films"],
        model=SparkAlsConfig(
            parameters["rank"],
            parameters["regularization"],
            parameters["confidence"],
            parameters["iterations"],
            parameters["seed"],
        ),
    )


def run_training_latency(
    spark: SparkSession, data_dir: Path, plan: LatencyPlan, output_dir: Path, code_commit: str
) -> dict[str, Any]:
    """Run every arm in a freshly shuffled order on each pass.

    The order is fixed by the seed, so a restart skips finished arms and continues.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MEASUREMENTS_FILE
    finished = load_finished(path)
    arms = list(itertools.product(plan.layouts, plan.window_months))
    rng = np.random.default_rng(plan.order_seed)

    for pass_number in range(plan.warmup_passes + plan.repetitions):
        for arm_index in rng.permutation(len(arms)).tolist():
            layout, months = arms[arm_index]
            arm_id = f"{pass_number}/{layout}/{months}"
            if arm_id in finished:
                continue
            split = window_before(plan.cutoff, months, plan.test_months)
            timed = train_and_time(spark, data_dir, layout, split, plan.model, plan.n_films)
            record = {
                "arm": arm_id,
                "pass": pass_number,
                "is_warmup": pass_number < plan.warmup_passes,
                "layout": layout,
                "window_months": months,
                "rows": timed.rows,
                "read_seconds": timed.read_seconds,
                "fit_seconds": timed.fit_seconds,
                "collect_seconds": timed.collect_seconds,
            }
            with path.open("a", encoding="utf-8") as measurements:
                measurements.write(json.dumps(record) + "\n")
            finished[arm_id] = record
            LOGGER.info(
                "%s: read %.1fs, fit %.1fs, collect %.1fs",
                arm_id,
                timed.read_seconds,
                timed.fit_seconds,
                timed.collect_seconds,
            )

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": {
            "spark": spark.version,
            "master": spark.sparkContext.master,
            "driver_memory": spark.conf.get("spark.driver.memory"),
        },
        "results": summarize_latency(list(finished.values())),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_latency(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report median times per arm, and the whole-job saving against flat storage.

    Warm-up passes are left out. Layouts must have read the same rows, or the times
    would not be comparable.
    """
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        if not record["is_warmup"]:
            key = (record["window_months"], record["layout"])
            grouped.setdefault(key, []).append(record)

    rows_by_window: dict[int, set[int]] = {}
    for (months, _), items in grouped.items():
        rows_by_window.setdefault(months, set()).update(item["rows"] for item in items)
    for months, rows in rows_by_window.items():
        if len(rows) != 1:
            raise ValueError(f"Layouts read different rows for {months} months: {rows}")

    medians = {key: _median_seconds(items) for key, items in grouped.items()}
    results = []
    for (months, layout), seconds in sorted(medians.items()):
        flat_total = medians[(months, FLAT_DIR)]["total"]
        results.append(
            {
                "window_months": months,
                "layout": layout,
                "rows": next(iter(rows_by_window[months])),
                "passes": len(grouped[(months, layout)]),
                "read_seconds_median": seconds["read"],
                "fit_seconds_median": seconds["fit"],
                "collect_seconds_median": seconds["collect"],
                "total_seconds_median": seconds["total"],
                "total_seconds_min": min(_total(item) for item in grouped[(months, layout)]),
                "total_seconds_max": max(_total(item) for item in grouped[(months, layout)]),
                "saving_against_flat": 1 - seconds["total"] / flat_total,
            }
        )
    return results


def _median_seconds(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "read": statistics.median(item["read_seconds"] for item in items),
        "fit": statistics.median(item["fit_seconds"] for item in items),
        "collect": statistics.median(item["collect_seconds"] for item in items),
        "total": statistics.median(_total(item) for item in items),
    }


def _total(record: dict[str, Any]) -> float:
    total: float = record["read_seconds"] + record["fit_seconds"] + record["collect_seconds"]
    return total
