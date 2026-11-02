"""Measure how quality and cost change with the amount of training history.

Every window is scored the same way so that scores can be compared. All windows at a cutoff
share the users who rated anything before it, and those users' earlier films are removed from
their lists, however short the training window. Users absent from a window get that window's
popular films, as a real service would give them.
"""

import itertools
import json
import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyspark.sql import SparkSession

from temporal_recsys.data.arrays import RatingArrays, load_ratings
from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.evaluation.harness import EvaluationConfig, Scorer, UserGroup, evaluate
from temporal_recsys.evaluation.split import TemporalSplit, add_months, split_by_time
from temporal_recsys.experiments.baseline import (
    ARMS_FILE,
    SUMMARY_FILE,
    load_finished,
    make_record,
)
from temporal_recsys.experiments.spark_als import train_and_time
from temporal_recsys.models.fallback import PopularityFallback
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.models.spark_als import SparkAlsConfig

LOGGER = logging.getLogger(__name__)

ALL_HISTORY = "all"
NO_SEED = 0
POPULARITY = "popularity"
SPARK_ALS = "spark-als"
TWO_TOWER = "two-tower"


@dataclass(frozen=True)
class HistoryPlan:
    id: str
    first_day: date
    window_months: tuple[int | None, ...]
    test_cutoffs: tuple[date, ...]
    test_months: int
    seeds: tuple[int, ...]
    n_films: int
    als_rank: int
    als_regularization: float
    als_confidence: float
    als_iterations: int
    tower_dimensions: int
    tower_max_history: int
    tower_batch_size: int
    tower_epochs: int
    tower_learning_rate: float
    tower_temperature: float
    evaluation: EvaluationConfig

    def history_split(self, cutoff: date) -> TemporalSplit:
        return TemporalSplit(self.first_day, cutoff, add_months(cutoff, self.test_months))

    def training_split(self, cutoff: date, months: int | None) -> TemporalSplit:
        """Return the window's split. `None` means all history, as does a window past the start."""
        history = self.history_split(cutoff)
        if months is None:
            return history
        start = max(add_months(cutoff, -months), self.first_day)
        return TemporalSplit(start, cutoff, history.test_end)


@dataclass(frozen=True)
class Trained:
    scorer: Scorer
    fit_seconds: float
    extras: dict[str, float]


Trainer = Callable[[str, TemporalSplit, RatingArrays, int], Trained]


def load_plan(path: Path) -> HistoryPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement = plan["design"], plan["measurement"]
    als, tower = design["spark_als"], design["two_tower"]
    return HistoryPlan(
        id=plan["id"],
        first_day=date.fromisoformat(design["first_day"]),
        window_months=tuple(design["window_months"]),
        test_cutoffs=tuple(date.fromisoformat(day) for day in design["test_cutoffs"]),
        test_months=design["test_months"],
        seeds=tuple(design["seeds"]),
        n_films=design["n_films"],
        als_rank=als["rank"],
        als_regularization=als["regularization"],
        als_confidence=als["confidence"],
        als_iterations=als["iterations"],
        tower_dimensions=tower["dimensions"],
        tower_max_history=tower["max_history"],
        tower_batch_size=tower["batch_size"],
        tower_epochs=tower["epochs"],
        tower_learning_rate=tower["learning_rate"],
        tower_temperature=tower["temperature"],
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            max_users=measurement["max_users"],
            seed=measurement["evaluation_seed"],
        ),
    )


def popularity_trainer(plan: HistoryPlan) -> Trainer:
    def train(_arm_id: str, _split: TemporalSplit, window: RatingArrays, _seed: int) -> Trained:
        model = PopularityModel()
        model.fit(window, plan.n_films)
        return Trained(model, 0.0, {})

    return train


def spark_als_trainer(spark: SparkSession, data_dir: Path, plan: HistoryPlan) -> Trainer:
    def train(_arm_id: str, split: TemporalSplit, window: RatingArrays, seed: int) -> Trained:
        config = SparkAlsConfig(
            plan.als_rank,
            plan.als_regularization,
            plan.als_confidence,
            plan.als_iterations,
            seed,
        )
        timed = train_and_time(spark, data_dir, MONTHLY_DIR, split, config, plan.n_films)
        if timed.rows != len(window):
            raise ValueError(
                f"Spark trained on {timed.rows} rows but the window holds {len(window)}"
            )
        total = timed.read_seconds + timed.fit_seconds + timed.collect_seconds
        return Trained(
            timed.model,
            total,
            {
                "read_seconds": timed.read_seconds,
                "model_fit_seconds": timed.fit_seconds,
                "collect_seconds": timed.collect_seconds,
            },
        )

    return train


def two_tower_trainer(raw_dir: Path, plan: HistoryPlan, output_dir: Path, device: str) -> Trainer:
    # PyTorch is an optional extra, so only this trainer may require it.
    from temporal_recsys.data.films import build_film_features, load_film_info
    from temporal_recsys.experiments.two_tower import CHECKPOINT_DIR, file_name
    from temporal_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel

    features = build_film_features(load_film_info(raw_dir), plan.n_films)

    def train(arm_id: str, _split: TemporalSplit, window: RatingArrays, seed: int) -> Trained:
        config = TwoTowerConfig(
            plan.tower_dimensions,
            plan.tower_max_history,
            plan.tower_batch_size,
            plan.tower_epochs,
            plan.tower_learning_rate,
            plan.tower_temperature,
            use_content=True,
            seed=seed,
        )
        checkpoint = output_dir / CHECKPOINT_DIR / f"{file_name(arm_id)}.pt"
        model = TwoTowerModel(config, features, device, checkpoint)
        model.fit(window, plan.n_films)
        return Trained(model.factors, model.training_seconds, {})

    return train


def run_history(
    monthly_dir: Path,
    plan: HistoryPlan,
    model_name: str,
    trainer: Trainer,
    seeds: Sequence[int],
    output_dir: Path,
    code_commit: str,
    machine: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_path = output_dir / ARMS_FILE
    finished = load_finished(arms_path)
    last_end = max(plan.history_split(cutoff).test_end for cutoff in plan.test_cutoffs)
    ratings = load_ratings(monthly_dir, plan.first_day, last_end)

    for cutoff, months, seed in itertools.product(plan.test_cutoffs, plan.window_months, seeds):
        arm_id = f"{cutoff}/{model_name}/months={months or ALL_HISTORY}/seed={seed}"
        if arm_id in finished:
            LOGGER.info("Keeping finished arm %s", arm_id)
            continue
        history, test = split_by_time(ratings, plan.history_split(cutoff))
        split = plan.training_split(cutoff, months)
        window = history.take(history.days >= np.datetime64(split.train_start))

        trained = trainer(arm_id, split, window, seed)
        popularity = PopularityModel()
        popularity.fit(window, plan.n_films)
        scorer = PopularityFallback(trained.scorer, popularity, np.unique(window.user_ids))
        result = evaluate(scorer, history, test, plan.n_films, plan.evaluation)

        record = make_record(
            arm_id, "test", model_name, result, trained.fit_seconds, window, test, UserGroup.KNOWN
        ) | {
            "cutoff": str(cutoff),
            "window": months or ALL_HISTORY,
            "seed": seed,
            "users_in_window": len(np.unique(window.user_ids)),
            **trained.extras,
        }
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info(
            "%s: NDCG@10 %.4f on %d rows in %.1fs",
            arm_id,
            result.ndcg,
            len(window),
            trained.fit_seconds,
        )

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": machine,
        "results": summarize_history(list(finished.values())),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_history(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each model and window, keeping the range across cutoffs and seeds."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model"], str(record["window"])), []).append(record)
    return [
        {
            "model": model,
            "window": window,
            "arms": len(items),
            "ndcg_mean": statistics.mean(item["ndcg"] for item in items),
            "ndcg_min": min(item["ndcg"] for item in items),
            "ndcg_max": max(item["ndcg"] for item in items),
            "recall_mean": statistics.mean(item["recall"] for item in items),
            "recall_min": min(item["recall"] for item in items),
            "recall_max": max(item["recall"] for item in items),
            "train_rows_mean": statistics.mean(item["train_rows"] for item in items),
            "fit_seconds_mean": statistics.mean(item["fit_seconds"] for item in items),
            "fit_seconds_min": min(item["fit_seconds"] for item in items),
            "fit_seconds_max": max(item["fit_seconds"] for item in items),
        }
        for (model, window), items in sorted(grouped.items())
    ]


def history_needed(records: Sequence[dict[str, Any]], tolerance: float = 0.05) -> str | None:
    """Return the shortest window near one model's best window at every cutoff.

    Comparing within a cutoff keeps easier months from hiding differences between windows.
    Each cutoff averages its seeds first. Records must hold one model.
    """
    by_cutoff: dict[str, dict[str, list[float]]] = {}
    for record in records:
        windows = by_cutoff.setdefault(record["cutoff"], {})
        windows.setdefault(str(record["window"]), []).append(record["ndcg"])
    if not by_cutoff:
        return None

    passing: set[str] | None = None
    for windows in by_cutoff.values():
        means = {window: statistics.mean(scores) for window, scores in windows.items()}
        floor = (1 - tolerance) * max(means.values())
        close = {window for window, mean in means.items() if mean >= floor}
        passing = close if passing is None else passing & close
    return min(passing or (), key=_window_length, default=None)


def _window_length(window: str) -> float:
    return float("inf") if window == ALL_HISTORY else int(window)
