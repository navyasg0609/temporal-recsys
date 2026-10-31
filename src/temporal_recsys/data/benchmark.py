"""Compare how quickly each storage layout serves a recent window of ratings."""

import itertools
import logging
import os
import platform
import shutil
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from temporal_recsys.data.partition import MONTH_COLUMN
from temporal_recsys.data.pipeline import FLAT_DIR, MONTHLY_DIR, SPARK_SUCCESS_MARKER

LOGGER = logging.getLogger(__name__)

DATE_SORTED_DIR = "date_sorted"
LAYOUTS = (FLAT_DIR, DATE_SORTED_DIR, MONTHLY_DIR)
RATING_COLUMNS = ("film_id", "user_id", "stars", "rated_on")
READ_TASK = "read"
AGGREGATE_TASK = "aggregate"
TASKS = (READ_TASK, AGGREGATE_TASK)


@dataclass(frozen=True)
class BenchmarkPlan:
    id: str
    last_month: str
    window_months: tuple[int, ...]
    repetitions: int
    warmup_passes: int
    seed: int


@dataclass(frozen=True)
class Measurement:
    run: int
    layout: str
    window_months: int
    task: str
    seconds: float
    is_warmup: bool


@dataclass(frozen=True)
class Summary:
    layout: str
    window_months: int
    task: str
    minimum_seconds: float
    median_seconds: float
    maximum_seconds: float


def load_plan(path: Path) -> BenchmarkPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design = plan["design"]
    return BenchmarkPlan(
        id=plan["id"],
        last_month=design["windows"]["last_month"],
        window_months=tuple(design["windows"]["months"]),
        repetitions=design["repetitions"],
        warmup_passes=design["warmup_passes"],
        seed=design["seed"],
    )


def window_bounds(last_month: str, months: int) -> tuple[date, date]:
    """Return the first day of the window and the first day after it."""
    year, month = map(int, last_month.split("-"))
    last_index = year * 12 + month - 1
    first_index = last_index - months + 1
    return (
        date(first_index // 12, first_index % 12 + 1, 1),
        date((last_index + 1) // 12, (last_index + 1) % 12 + 1, 1),
    )


def build_date_sorted(spark: SparkSession, data_dir: Path) -> None:
    """Write the date-sorted layout unless a finished copy exists."""
    target = data_dir / DATE_SORTED_DIR
    if (target / SPARK_SUCCESS_MARKER).exists():
        return
    if target.exists():
        shutil.rmtree(target)
    spark.read.parquet(str(data_dir / FLAT_DIR)).sort("rated_on").write.parquet(str(target))


def select_window(
    spark: SparkSession, data_dir: Path, layout: str, first_day: date, end_day: date
) -> DataFrame:
    """Select a window using the filter each layout can use to skip data."""
    ratings = spark.read.parquet(str(data_dir / layout))
    if layout == MONTHLY_DIR:
        months_filter = (F.col(MONTH_COLUMN) >= first_day.strftime("%Y-%m")) & (
            F.col(MONTH_COLUMN) < end_day.strftime("%Y-%m")
        )
        return ratings.filter(months_filter).select(*RATING_COLUMNS)
    days_filter = (F.col("rated_on") >= first_day) & (F.col("rated_on") < end_day)
    return ratings.filter(days_filter).select(*RATING_COLUMNS)


def eligible_bytes(data_dir: Path, layout: str, first_day: date, end_day: date) -> int:
    """Return the size of the files a layout must consider for a window."""
    files = (data_dir / layout).rglob("*.parquet")
    if layout == MONTHLY_DIR:
        first_month, end_month = first_day.strftime("%Y-%m"), end_day.strftime("%Y-%m")
        files = (
            path for path in files if first_month <= path.parent.name.split("=")[1] < end_month
        )
    return sum(path.stat().st_size for path in files)


def run_task(window: DataFrame, task: str) -> None:
    if task == AGGREGATE_TASK:
        window = window.groupBy("film_id").agg(F.count("*"), F.avg("stars"))
    window.write.format("noop").mode("overwrite").save()


def measure_windows(
    spark: SparkSession, data_dir: Path, plan: BenchmarkPlan
) -> list[dict[str, int | str]]:
    """Record each window's size, and fail if the layouts disagree on its rows."""
    rows: list[dict[str, int | str]] = []
    for months in plan.window_months:
        first_day, end_day = window_bounds(plan.last_month, months)
        counts = {
            layout: select_window(spark, data_dir, layout, first_day, end_day).count()
            for layout in LAYOUTS
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"Layouts matched different rows for {months} months: {counts}")
        for layout in LAYOUTS:
            rows.append(
                {
                    "layout": layout,
                    "window_months": months,
                    "rows_matched": counts[layout],
                    "eligible_bytes": eligible_bytes(data_dir, layout, first_day, end_day),
                }
            )
    return rows


def run_passes(spark: SparkSession, data_dir: Path, plan: BenchmarkPlan) -> list[Measurement]:
    """Time every arm in a freshly shuffled order on each pass."""
    arms = list(itertools.product(LAYOUTS, plan.window_months, TASKS))
    rng = np.random.default_rng(plan.seed)
    measurements: list[Measurement] = []
    for run in range(plan.warmup_passes + plan.repetitions):
        for arm_index in rng.permutation(len(arms)).tolist():
            layout, months, task = arms[arm_index]
            first_day, end_day = window_bounds(plan.last_month, months)
            window = select_window(spark, data_dir, layout, first_day, end_day)
            start = time.perf_counter()
            run_task(window, task)
            seconds = time.perf_counter() - start
            LOGGER.info(
                "run %d %s %d %s: %.2fs",
                run,
                layout,
                months,
                task,
                seconds,
            )
            measurements.append(
                Measurement(run, layout, months, task, seconds, run < plan.warmup_passes)
            )
    return measurements


def summarize(measurements: Sequence[Measurement]) -> list[Summary]:
    """Summarize measured passes, leaving out warm-up passes."""
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for item in measurements:
        if not item.is_warmup:
            key = (item.layout, item.window_months, item.task)
            grouped.setdefault(key, []).append(item.seconds)
    return [
        Summary(
            layout,
            months,
            task,
            min(seconds),
            statistics.median(seconds),
            max(seconds),
        )
        for (layout, months, task), seconds in sorted(grouped.items())
    ]


def run_benchmark(
    spark: SparkSession, data_dir: Path, plan: BenchmarkPlan, code_commit: str
) -> dict[str, Any]:
    build_date_sorted(spark, data_dir)
    windows = measure_windows(spark, data_dir, plan)
    measurements = run_passes(spark, data_dir, plan)
    return {
        "plan": asdict(plan),
        "code_commit": code_commit,
        "machine": {
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "spark": spark.version,
        },
        "windows": windows,
        "summary": [asdict(item) for item in summarize(measurements)],
        "measurements": [asdict(item) for item in measurements],
    }
