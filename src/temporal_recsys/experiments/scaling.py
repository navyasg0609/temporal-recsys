"""Measure how Spark ALS training time changes with the number of cores.

Speedup compares Spark with itself on one core. COST (McSherry et al., 2015) asks how
many cores Spark needs to beat a single-threaded program doing the same job, here the
NumPy ALS limited to one thread.

Each measurement runs in a fresh process. A Spark session's core count is fixed when it
starts, and a fresh process also stops one measurement from warming caches for the next.
"""

import json
import logging
import os
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyspark
import yaml
from threadpoolctl import threadpool_limits

from temporal_recsys.data.arrays import load_ratings
from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.data.spark import create_spark_session
from temporal_recsys.evaluation.split import TemporalSplit, window_before
from temporal_recsys.experiments.baseline import SUMMARY_FILE, load_finished
from temporal_recsys.experiments.spark_als import train_and_time
from temporal_recsys.models.als import AlsConfig, AlsModel, AlsVariant
from temporal_recsys.models.spark_als import SparkAlsConfig

LOGGER = logging.getLogger(__name__)

MEASUREMENTS_FILE = "measurements.jsonl"
SPARK = "spark"
NUMPY = "numpy"
# Nothing is evaluated, but a split needs a test period after the cutoff.
UNUSED_TEST_MONTHS = 1


@dataclass(frozen=True)
class ScalingPlan:
    id: str
    cutoff: date
    window_months: int
    core_counts: tuple[int, ...]
    repetitions: int
    warmup_passes: int
    order_seed: int
    n_films: int
    driver_memory: str
    spark_model: SparkAlsConfig
    numpy_model: AlsConfig

    def __post_init__(self) -> None:
        if 1 not in self.core_counts:
            raise ValueError("Speedup is measured against Spark on one core")

    @property
    def split(self) -> TemporalSplit:
        return window_before(self.cutoff, self.window_months, UNUSED_TEST_MONTHS)

    @property
    def arms(self) -> list[tuple[str, int]]:
        return [(SPARK, cores) for cores in self.core_counts] + [(NUMPY, 1)]


def load_plan(path: Path) -> ScalingPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design = plan["design"]
    spark, numpy = design["spark_als"], design["numpy_als"]
    return ScalingPlan(
        id=plan["id"],
        cutoff=date.fromisoformat(design["cutoff"]),
        window_months=design["window_months"],
        core_counts=tuple(design["core_counts"]),
        repetitions=design["repetitions"],
        warmup_passes=design["warmup_passes"],
        order_seed=design["order_seed"],
        n_films=design["n_films"],
        driver_memory=design["driver_memory"],
        spark_model=SparkAlsConfig(
            spark["rank"],
            spark["regularization"],
            spark["confidence"],
            spark["iterations"],
            spark["seed"],
        ),
        numpy_model=AlsConfig(
            AlsVariant.PREFERENCE,
            numpy["rank"],
            numpy["regularization"],
            numpy["confidence"],
            numpy["iterations"],
            numpy["seed"],
        ),
    )


def time_spark(
    data_dir: Path,
    split: TemporalSplit,
    config: SparkAlsConfig,
    n_films: int,
    cores: int,
    driver_memory: str,
) -> dict[str, Any]:
    spark = create_spark_session("scaling", driver_memory=driver_memory, cores=cores)
    try:
        timed = train_and_time(spark, data_dir, MONTHLY_DIR, split, config, n_films)
        master = spark.sparkContext.master
    finally:
        spark.stop()
    return {
        "master": master,
        "rows": timed.rows,
        "read_seconds": timed.read_seconds,
        "fit_seconds": timed.fit_seconds,
        "collect_seconds": timed.collect_seconds,
    }


def time_numpy(
    data_dir: Path, split: TemporalSplit, config: AlsConfig, n_films: int
) -> dict[str, Any]:
    # The fit is a Python loop, but the matrix products could use several BLAS threads.
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        train = load_ratings(data_dir / MONTHLY_DIR, split.train_start, split.cutoff)
        read_done = time.perf_counter()
        AlsModel(config).fit(train, n_films)
        fit_done = time.perf_counter()
    return {
        "master": "numpy, 1 thread",
        "rows": len(train),
        "read_seconds": read_done - start,
        "fit_seconds": fit_done - read_done,
        "collect_seconds": 0.0,
    }


def in_fresh_process(function: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(function, *args).result()


def run_scaling(
    data_dir: Path, plan: ScalingPlan, output_dir: Path, code_commit: str
) -> dict[str, Any]:
    """Run every arm in a freshly shuffled order on each pass.

    The order is fixed by the seed, so a restart skips finished arms and continues.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MEASUREMENTS_FILE
    finished = load_finished(path)
    arms = plan.arms
    rng = np.random.default_rng(plan.order_seed)

    for pass_number in range(plan.warmup_passes + plan.repetitions):
        for arm_index in rng.permutation(len(arms)).tolist():
            engine, cores = arms[arm_index]
            arm_id = f"{pass_number}/{engine}/{cores}"
            if arm_id in finished:
                continue
            if engine == SPARK:
                timed = in_fresh_process(
                    time_spark,
                    data_dir,
                    plan.split,
                    plan.spark_model,
                    plan.n_films,
                    cores,
                    plan.driver_memory,
                )
            else:
                timed = in_fresh_process(
                    time_numpy, data_dir, plan.split, plan.numpy_model, plan.n_films
                )
            record = {
                "arm": arm_id,
                "pass": pass_number,
                "is_warmup": pass_number < plan.warmup_passes,
                "engine": engine,
                "cores": cores,
            } | timed
            with path.open("a", encoding="utf-8") as measurements:
                measurements.write(json.dumps(record) + "\n")
            finished[arm_id] = record
            LOGGER.info("%s: %.1fs in total", arm_id, _total(record))

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": {
            "cpu_count": os.cpu_count(),
            "spark": pyspark.__version__,
            "driver_memory": plan.driver_memory,
        },
    } | summarize_scaling(list(finished.values()))
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_scaling(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report median job times, Spark's speedup over one core, and its COST.

    Warm-up passes are left out. COST is the fewest cores at which Spark's slowest run
    beats NumPy's fastest, so the two ranges do not overlap. It is None if no core
    count does. Every arm must have read the same rows, or the times would not compare.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if not record["is_warmup"]:
            grouped.setdefault((record["engine"], record["cores"]), []).append(record)

    rows = {record["rows"] for items in grouped.values() for record in items}
    if len(rows) != 1:
        raise ValueError(f"Arms read different rows: {sorted(rows)}")
    if (SPARK, 1) not in grouped or (NUMPY, 1) not in grouped:
        raise ValueError("Need Spark on one core and NumPy to compare against")

    def median(key: tuple[str, int]) -> float:
        return statistics.median(_total(item) for item in grouped[key])

    spark_one_core = median((SPARK, 1))
    numpy_fastest = min(_total(item) for item in grouped[(NUMPY, 1)])
    results = []
    for (engine, cores), items in sorted(grouped.items()):
        totals = [_total(item) for item in items]
        result: dict[str, Any] = {
            "engine": engine,
            "cores": cores,
            "passes": len(items),
            "read_seconds_median": statistics.median(item["read_seconds"] for item in items),
            "fit_seconds_median": statistics.median(item["fit_seconds"] for item in items),
            "total_seconds_median": statistics.median(totals),
            "total_seconds_min": min(totals),
            "total_seconds_max": max(totals),
        }
        if engine == SPARK:
            result["speedup"] = spark_one_core / statistics.median(totals)
            result["beats_numpy"] = max(totals) < numpy_fastest
        results.append(result)

    beating = [r["cores"] for r in results if r["engine"] == SPARK and r["beats_numpy"]]
    return {"rows": rows.pop(), "cost": min(beating, default=None), "results": results}


def _total(record: dict[str, Any]) -> float:
    total: float = record["read_seconds"] + record["fit_seconds"] + record["collect_seconds"]
    return total
