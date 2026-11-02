"""Train the two-tower model on a registered plan.

Settings are chosen on the tuning cutoff with title and year features on, and then frozen
for the test cutoffs. Each finished arm is saved at once, and each epoch is saved during an
arm, so a session that ends loses at most one epoch.
"""

import itertools
import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from temporal_recsys.data.arrays import load_ratings
from temporal_recsys.data.films import FilmFeatures, build_film_features, load_film_info
from temporal_recsys.evaluation.harness import EvaluationConfig, UserGroup, evaluate
from temporal_recsys.evaluation.split import (
    TemporalSplit,
    add_months,
    split_by_time,
    window_before,
)
from temporal_recsys.experiments.baseline import (
    ARMS_FILE,
    SUMMARY_FILE,
    load_finished,
    make_record,
    summarize,
)
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel

LOGGER = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
FULL_MODEL = "two-tower"
ID_ONLY_MODEL = "two-tower-id-only"
RECENT_POPULARITY_MODEL = "popularity-recent"
FIRST_GPU = "cuda:0"


@dataclass(frozen=True)
class TwoTowerPlan:
    id: str
    window_months: int
    tuning_cutoff: date
    test_cutoffs: tuple[date, ...]
    test_months: int
    seeds: tuple[int, ...]
    n_films: int
    dimensions: int
    max_history: int
    batch_size: int
    epochs: tuple[int, ...]
    learning_rate: tuple[float, ...]
    temperature: tuple[float, ...]
    compare_id_only: bool
    evaluation: EvaluationConfig
    recent_popularity_months: int | None = None

    def split_at(self, cutoff: date) -> TemporalSplit:
        return window_before(cutoff, self.window_months, self.test_months)


def load_plan(path: Path) -> TwoTowerPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement = plan["design"], plan["measurement"]
    fixed = design["fixed_parameters"]
    return TwoTowerPlan(
        id=plan["id"],
        window_months=design["window_months"],
        tuning_cutoff=date.fromisoformat(design["tuning_cutoff"]),
        test_cutoffs=tuple(date.fromisoformat(day) for day in design["test_cutoffs"]),
        test_months=design["test_months"],
        seeds=tuple(design["seeds"]),
        n_films=design["n_films"],
        dimensions=fixed["dimensions"],
        max_history=fixed["max_history"],
        batch_size=fixed["batch_size"],
        epochs=tuple(design["grid"]["epochs"]),
        learning_rate=tuple(design["grid"]["learning_rate"]),
        temperature=tuple(design["grid"]["temperature"]),
        compare_id_only=design["compare_id_only"],
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            max_users=measurement["max_users"],
            seed=measurement["evaluation_seed"],
        ),
        recent_popularity_months=design.get("recent_popularity_months"),
    )


def tuning_grid(plan: TwoTowerPlan) -> list[TwoTowerConfig]:
    return [
        TwoTowerConfig(
            plan.dimensions,
            plan.max_history,
            plan.batch_size,
            epochs,
            learning_rate,
            temperature,
            use_content=True,
            seed=plan.seeds[0],
        )
        for epochs, learning_rate, temperature in itertools.product(
            plan.epochs, plan.learning_rate, plan.temperature
        )
    ]


def run_two_tower(
    monthly_dir: Path,
    raw_dir: Path,
    plan: TwoTowerPlan,
    output_dir: Path,
    code_commit: str,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_path = output_dir / ARMS_FILE
    finished = load_finished(arms_path)
    features = build_film_features(load_film_info(raw_dir), plan.n_films)
    splits = [plan.split_at(day) for day in (plan.tuning_cutoff, *plan.test_cutoffs)]
    ratings = load_ratings(
        monthly_dir,
        min(split.train_start for split in splits),
        max(split.test_end for split in splits),
    )

    def run_arm(
        arm_id: str, stage: str, name: str, config: TwoTowerConfig, cutoff: date, data: FilmFeatures
    ) -> float:
        if arm_id in finished:
            LOGGER.info("Keeping finished arm %s", arm_id)
            ndcg: float = finished[arm_id]["ndcg"]
            return ndcg
        train, test = split_by_time(ratings, plan.split_at(cutoff))
        checkpoint = output_dir / CHECKPOINT_DIR / f"{_file_name(arm_id)}.pt"
        model = TwoTowerModel(config, data, device, checkpoint)
        model.fit(train, plan.n_films)
        result = evaluate(model, train, test, plan.n_films, plan.evaluation)
        record = make_record(
            arm_id, stage, name, result, model.training_seconds, train, test, UserGroup.KNOWN
        )
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info("%s: NDCG@10 %.4f in %.1fs", arm_id, result.ndcg, model.training_seconds)
        return result.ndcg

    def run_recent_popularity(cutoff: date, months: int) -> None:
        """Fit on the latest months, and score on the 12-month model's users and seen films."""
        arm_id = f"test/{cutoff}/{RECENT_POPULARITY_MODEL}"
        if arm_id in finished:
            return
        train, test = split_by_time(ratings, plan.split_at(cutoff))
        model = PopularityModel()
        model.fit(
            train.take(train.days >= np.datetime64(add_months(cutoff, -months))), plan.n_films
        )
        result = evaluate(model, train, test, plan.n_films, plan.evaluation)
        record = make_record(
            arm_id, "test", RECENT_POPULARITY_MODEL, result, 0.0, train, test, UserGroup.KNOWN
        )
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info("%s: NDCG@10 %.4f", arm_id, result.ndcg)

    grid = tuning_grid(plan)
    if not grid:
        raise ValueError("The plan's grid is empty")
    best = grid[0]
    best_ndcg = -1.0
    for config in grid:
        name = _config_name(config)
        ndcg = run_arm(f"tune/{name}", "tune", name, config, plan.tuning_cutoff, features)
        if ndcg > best_ndcg:
            best, best_ndcg = config, ndcg

    if plan.recent_popularity_months is not None:
        for cutoff in plan.test_cutoffs:
            run_recent_popularity(cutoff, plan.recent_popularity_months)

    models = [(FULL_MODEL, True)] + ([(ID_ONLY_MODEL, False)] if plan.compare_id_only else [])
    for cutoff, seed, (name, use_content) in itertools.product(
        plan.test_cutoffs, plan.seeds, models
    ):
        config = replace(best, seed=seed, use_content=use_content)
        run_arm(f"test/{cutoff}/{name}/seed={seed}", "test", name, config, cutoff, features)

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "machine": {"device": device, "gpu": _gpu_name(device), "torch": torch.__version__},
        "chosen_settings": {FULL_MODEL: _config_name(best)},
        "results": summarize(list(finished.values())),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def pick_device() -> str:
    return FIRST_GPU if torch.cuda.is_available() else "cpu"


def _gpu_name(device: str) -> str | None:
    return torch.cuda.get_device_name(0) if device == FIRST_GPU else None


def _config_name(config: TwoTowerConfig) -> str:
    return (
        f"{FULL_MODEL}/epochs={config.epochs}/lr={config.learning_rate}"
        f"/temperature={config.temperature}"
    )


def _file_name(arm_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", arm_id)
