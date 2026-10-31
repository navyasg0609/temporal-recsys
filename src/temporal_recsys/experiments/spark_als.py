"""Train ALS with Spark on a registered plan, timing each part of the job.

Settings are chosen on the tuning cutoff and then frozen for the test cutoffs. Each
finished arm is saved at once, so a run that stops partway can simply be restarted.
"""

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import SparkSession

from temporal_recsys.data.arrays import load_ratings
from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.evaluation.harness import EvaluationConfig, UserGroup, evaluate
from temporal_recsys.evaluation.split import TemporalSplit, split_by_time, window_before
from temporal_recsys.experiments.baseline import (
    ARMS_FILE,
    SUMMARY_FILE,
    load_finished,
    make_record,
    summarize,
)
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.models.spark_als import (
    SparkAlsConfig,
    collect_factors,
    fit_spark_als,
    training_window,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "spark"


@dataclass(frozen=True)
class SparkAlsPlan:
    id: str
    window_months: int
    tuning_cutoff: date
    test_cutoffs: tuple[date, ...]
    test_months: int
    seeds: tuple[int, ...]
    n_films: int
    rank: int
    iterations: int
    regularization: tuple[float, ...]
    confidence: tuple[float, ...]
    evaluation: EvaluationConfig

    def split_at(self, cutoff: date) -> TemporalSplit:
        return window_before(cutoff, self.window_months, self.test_months)


@dataclass(frozen=True)
class TimedTraining:
    model: FactorModel
    rows: int
    read_seconds: float
    fit_seconds: float
    collect_seconds: float


def load_plan(path: Path) -> SparkAlsPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement = plan["design"], plan["measurement"]
    return SparkAlsPlan(
        id=plan["id"],
        window_months=design["window_months"],
        tuning_cutoff=date.fromisoformat(design["tuning_cutoff"]),
        test_cutoffs=tuple(date.fromisoformat(day) for day in design["test_cutoffs"]),
        test_months=design["test_months"],
        seeds=tuple(design["seeds"]),
        n_films=design["n_films"],
        rank=design["fixed_parameters"]["rank"],
        iterations=design["fixed_parameters"]["iterations"],
        regularization=tuple(design["grid"]["regularization"]),
        confidence=tuple(design["grid"]["confidence"]),
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            max_users=measurement["max_users"],
            seed=measurement["evaluation_seed"],
        ),
    )


def train_and_time(
    spark: SparkSession,
    data_dir: Path,
    layout: str,
    split: TemporalSplit,
    config: SparkAlsConfig,
    n_films: int,
) -> TimedTraining:
    """Time reading, fitting and copying the factors as three separate parts.

    The data is cached and counted first. Spark reads lazily, so otherwise the read
    would be hidden inside the fit.
    """
    start = time.perf_counter()
    window = training_window(spark, data_dir, layout, split).select("user_id", "film_id").persist()
    try:
        rows = window.count()
        read_done = time.perf_counter()
        fitted = fit_spark_als(window, config)
        fit_done = time.perf_counter()
        model = collect_factors(fitted, n_films)
        collect_done = time.perf_counter()
    finally:
        window.unpersist()
    return TimedTraining(
        model, rows, read_done - start, fit_done - read_done, collect_done - fit_done
    )


def run_spark_als(
    spark: SparkSession, data_dir: Path, plan: SparkAlsPlan, output_dir: Path, code_commit: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_path = output_dir / ARMS_FILE
    finished = load_finished(arms_path)
    splits = [plan.split_at(day) for day in (plan.tuning_cutoff, *plan.test_cutoffs)]
    ratings = load_ratings(
        data_dir / MONTHLY_DIR,
        min(split.train_start for split in splits),
        max(split.test_end for split in splits),
    )

    def run_arm(arm_id: str, stage: str, name: str, config: SparkAlsConfig, cutoff: date) -> float:
        if arm_id in finished:
            LOGGER.info("Keeping finished arm %s", arm_id)
            ndcg: float = finished[arm_id]["ndcg"]
            return ndcg
        split = plan.split_at(cutoff)
        train, test = split_by_time(ratings, split)
        timed = train_and_time(spark, data_dir, MONTHLY_DIR, split, config, plan.n_films)
        if timed.rows != len(train):
            raise ValueError(
                f"Spark trained on {timed.rows} rows but evaluation used {len(train)} for {arm_id}"
            )
        result = evaluate(timed.model, train, test, plan.n_films, plan.evaluation)
        record = make_record(
            arm_id, stage, name, result, timed.fit_seconds, train, test, UserGroup.KNOWN
        ) | {"read_seconds": timed.read_seconds, "collect_seconds": timed.collect_seconds}
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info(
            "%s: NDCG@10 %.4f, read %.1fs, fit %.1fs, collect %.1fs",
            arm_id,
            result.ndcg,
            timed.read_seconds,
            timed.fit_seconds,
            timed.collect_seconds,
        )
        return result.ndcg

    best: SparkAlsConfig | None = None
    best_ndcg = -1.0
    for regularization, confidence in itertools.product(plan.regularization, plan.confidence):
        config = SparkAlsConfig(
            plan.rank, regularization, confidence, plan.iterations, plan.seeds[0]
        )
        name = _config_name(config)
        ndcg = run_arm(f"tune/{name}", "tune", name, config, plan.tuning_cutoff)
        if ndcg > best_ndcg:
            best, best_ndcg = config, ndcg
    if best is None:
        raise ValueError("The plan's grid is empty")

    for cutoff, seed in itertools.product(plan.test_cutoffs, plan.seeds):
        config = SparkAlsConfig(
            plan.rank, best.regularization, best.confidence, plan.iterations, seed
        )
        run_arm(f"test/{cutoff}/{MODEL_NAME}/seed={seed}", "test", MODEL_NAME, config, cutoff)

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": {
            "cpu_count": os.cpu_count(),
            "spark": spark.version,
            "master": spark.sparkContext.master,
            "driver_memory": spark.conf.get("spark.driver.memory"),
        },
        "chosen_settings": {MODEL_NAME: _config_name(best)},
        "results": summarize(list(finished.values())),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _config_name(config: SparkAlsConfig) -> str:
    return f"{MODEL_NAME}/reg={config.regularization}/confidence={config.confidence}"
