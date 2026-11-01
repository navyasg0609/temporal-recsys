"""Find out why the two-tower pilot scored about twice as high as ALS.

A jump that large must be explained before it is believed. Each arm tests one explanation
on the tuning cutoff only. Recent-only arms test whether simple models also gain from
recency, and they share the 12-month model's users and seen films so scores stay comparable.
The other arms remove title and year features, or films absent from training.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from temporal_recsys.data.arrays import RatingArrays, load_ratings
from temporal_recsys.data.films import build_film_features, load_film_info
from temporal_recsys.evaluation.harness import (
    EvaluationConfig,
    EvaluationResult,
    Scorer,
    evaluate,
)
from temporal_recsys.evaluation.split import add_months, split_by_time, window_before
from temporal_recsys.experiments.baseline import ARMS_FILE, SUMMARY_FILE, load_finished
from temporal_recsys.models.als import AlsConfig, AlsModel, AlsVariant
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILE = "two-tower.pt"
ID_ONLY_CHECKPOINT_FILE = "two-tower-id-only.pt"


@dataclass(frozen=True)
class DiagnosticPlan:
    id: str
    cutoff: date
    window_months: int
    test_months: int
    n_films: int
    recent_months: tuple[int, ...]
    als: AlsConfig
    tower: TwoTowerConfig
    evaluation: EvaluationConfig


def load_plan(path: Path) -> DiagnosticPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement = plan["design"], plan["measurement"]
    return DiagnosticPlan(
        id=plan["id"],
        cutoff=date.fromisoformat(design["cutoff"]),
        window_months=design["window_months"],
        test_months=design["test_months"],
        n_films=design["n_films"],
        recent_months=tuple(design["recent_months"]),
        als=AlsConfig(AlsVariant.PREFERENCE, **design["als"]),
        tower=TwoTowerConfig(**design["two_tower"]),
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            max_users=measurement["max_users"],
            seed=measurement["evaluation_seed"],
        ),
    )


def without_films(factors: FactorModel, is_removed: np.ndarray[Any, Any]) -> FactorModel:
    """Copy a model so that the given films score zero for everyone."""
    film_factors = factors.film_factors.copy()
    film_factors[is_removed] = 0.0
    return FactorModel(factors.user_ids, factors.user_factors, film_factors)


def share_of_relevant_ratings_on_unseen_films(
    train: RatingArrays, test: RatingArrays, relevant_stars: int
) -> float:
    relevant = test.film_ids[test.stars >= relevant_stars]
    return float(np.mean(~np.isin(relevant, train.film_ids)))


def run_diagnostic(
    monthly_dir: Path,
    raw_dir: Path,
    plan: DiagnosticPlan,
    output_dir: Path,
    code_commit: str,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_path = output_dir / ARMS_FILE
    finished = load_finished(arms_path)
    split = window_before(plan.cutoff, plan.window_months, plan.test_months)
    train, test = split_by_time(load_ratings(monthly_dir, split.train_start, split.test_end), split)
    features = build_film_features(load_film_info(raw_dir), plan.n_films)

    def run_arm(arm_id: str, build: Callable[[], Scorer]) -> None:
        if arm_id in finished:
            LOGGER.info("Keeping finished arm %s", arm_id)
            return
        result: EvaluationResult = evaluate(build(), train, test, plan.n_films, plan.evaluation)
        record = {
            "arm": arm_id,
            "ndcg": result.ndcg,
            "ndcg_stderr": result.ndcg_stderr,
            "recall": result.recall,
            "recall_stderr": result.recall_stderr,
            "users_evaluated": result.users_evaluated,
        }
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info("%s: NDCG@10 %.4f", arm_id, result.ndcg)

    def recent(months: int) -> RatingArrays:
        return train.take(train.days >= np.datetime64(add_months(plan.cutoff, -months)))

    def fitted(model: Any, ratings: RatingArrays) -> Scorer:
        model.fit(ratings, plan.n_films)
        scorer: Scorer = model
        return scorer

    run_arm("popularity/months=12", lambda: fitted(PopularityModel(), train))
    run_arm("als/months=12", lambda: fitted(AlsModel(plan.als), train))

    def run_recent_arms(months: int) -> None:
        ratings = recent(months)
        run_arm(f"popularity/months={months}", lambda: fitted(PopularityModel(), ratings))
        run_arm(f"als/months={months}", lambda: fitted(AlsModel(plan.als), ratings))

    for months in plan.recent_months:
        run_recent_arms(months)

    tower = TwoTowerModel(plan.tower, features, device, output_dir / CHECKPOINT_FILE)
    tower.fit(train, plan.n_films)
    run_arm("two-tower/full", lambda: tower)
    is_unseen = np.bincount(train.film_ids - 1, minlength=plan.n_films) == 0
    run_arm("two-tower/unseen-films-removed", lambda: without_films(tower.factors, is_unseen))

    id_only_config = TwoTowerConfig(
        plan.tower.dimensions,
        plan.tower.max_history,
        plan.tower.batch_size,
        plan.tower.epochs,
        plan.tower.learning_rate,
        plan.tower.temperature,
        use_content=False,
        seed=plan.tower.seed,
    )
    id_only = TwoTowerModel(id_only_config, features, device, output_dir / ID_ONLY_CHECKPOINT_FILE)
    run_arm("two-tower/id-only", lambda: fitted(id_only, train))

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": {"device": device, "torch": torch.__version__},
        "latest_training_day": str(train.days.max()),
        "first_test_day": str(test.days.min()),
        "films_absent_from_training": int(is_unseen.sum()),
        "share_of_relevant_ratings_on_unseen_films": share_of_relevant_ratings_on_unseen_films(
            train, test, plan.evaluation.relevant_stars
        ),
        "arms": list(finished.values()),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
