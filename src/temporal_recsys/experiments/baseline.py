"""Compare the baselines on a registered plan.

Settings are chosen on the tuning cutoff and then frozen for the test cutoffs. Each
finished arm is saved at once, so a run that stops partway can simply be restarted.
"""

import itertools
import json
import logging
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from temporal_recsys.data.arrays import RatingArrays, load_ratings
from temporal_recsys.evaluation.harness import (
    EvaluationConfig,
    EvaluationResult,
    Recommender,
    UserGroup,
    evaluate,
    has_eligible_users,
)
from temporal_recsys.evaluation.split import (
    TemporalSplit,
    split_at_random,
    split_by_time,
    window_before,
)
from temporal_recsys.models.als import AlsConfig, AlsModel, AlsVariant
from temporal_recsys.models.popularity import PopularityModel

LOGGER = logging.getLogger(__name__)

ARMS_FILE = "arms.jsonl"
SUMMARY_FILE = "summary.json"
POPULARITY = "popularity"


@dataclass(frozen=True)
class BaselinePlan:
    id: str
    window_months: int
    tuning_cutoff: date
    test_cutoffs: tuple[date, ...]
    test_months: int
    seeds: tuple[int, ...]
    n_films: int
    rank: int
    iterations: int
    rating_regularization: tuple[float, ...]
    preference_regularization: tuple[float, ...]
    preference_confidence: tuple[float, ...]
    leakage_cutoff: date
    leakage_seeds: tuple[int, ...]
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if self.leakage_cutoff not in self.test_cutoffs:
            raise ValueError("The leakage cutoff must be one of the test cutoffs")

    def split_at(self, cutoff: date) -> TemporalSplit:
        return window_before(cutoff, self.window_months, self.test_months)


def load_plan(path: Path) -> BaselinePlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement = plan["design"], plan["measurement"]
    return BaselinePlan(
        id=plan["id"],
        window_months=design["window_months"],
        tuning_cutoff=date.fromisoformat(design["tuning_cutoff"]),
        test_cutoffs=tuple(date.fromisoformat(day) for day in design["test_cutoffs"]),
        test_months=design["test_months"],
        seeds=tuple(design["seeds"]),
        n_films=design["n_films"],
        rank=design["fixed_parameters"]["rank"],
        iterations=design["fixed_parameters"]["iterations"],
        rating_regularization=tuple(design["rating_grid"]["regularization"]),
        preference_regularization=tuple(design["preference_grid"]["regularization"]),
        preference_confidence=tuple(design["preference_grid"]["confidence"]),
        leakage_cutoff=date.fromisoformat(design["leakage"]["cutoff"]),
        leakage_seeds=tuple(design["leakage"]["seeds"]),
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            max_users=measurement["max_users"],
            seed=measurement["evaluation_seed"],
        ),
    )


def tuning_grid(plan: BaselinePlan) -> Iterator[AlsConfig]:
    shared = {"rank": plan.rank, "iterations": plan.iterations, "seed": plan.seeds[0]}
    for regularization in plan.rating_regularization:
        yield AlsConfig(AlsVariant.RATING, regularization=regularization, **shared)
    for regularization, confidence in itertools.product(
        plan.preference_regularization, plan.preference_confidence
    ):
        yield AlsConfig(
            AlsVariant.PREFERENCE, regularization=regularization, confidence=confidence, **shared
        )


def run_baseline(
    monthly_dir: Path, plan: BaselinePlan, output_dir: Path, code_commit: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_path = output_dir / ARMS_FILE
    finished = load_finished(arms_path)
    ratings = _load_all_ratings(monthly_dir, plan)

    def run_arm(
        arm_id: str,
        stage: str,
        model_name: str,
        model: Recommender,
        train: RatingArrays,
        test: RatingArrays,
        group: UserGroup = UserGroup.KNOWN,
    ) -> dict[str, Any]:
        if arm_id in finished:
            LOGGER.info("Keeping finished arm %s", arm_id)
            return finished[arm_id]
        start = time.perf_counter()
        model.fit(train, plan.n_films)
        fit_seconds = time.perf_counter() - start
        result = evaluate(model, train, test, plan.n_films, plan.evaluation, group)
        record = make_record(arm_id, stage, model_name, result, fit_seconds, train, test, group)
        with arms_path.open("a", encoding="utf-8") as arms_file:
            arms_file.write(json.dumps(record) + "\n")
        finished[arm_id] = record
        LOGGER.info("%s: NDCG@10 %.4f in %.1fs", arm_id, result.ndcg, fit_seconds)
        return record

    best: dict[AlsVariant, AlsConfig] = {}
    best_ndcg: dict[AlsVariant, float] = {}
    tuning_train, tuning_test = split_by_time(ratings, plan.split_at(plan.tuning_cutoff))
    for config in tuning_grid(plan):
        name = _config_name(config)
        record = run_arm(f"tune/{name}", "tune", name, AlsModel(config), tuning_train, tuning_test)
        if record["ndcg"] > best_ndcg.get(config.variant, -1.0):
            best_ndcg[config.variant] = record["ndcg"]
            best[config.variant] = config

    temporal_test_size = 0
    for cutoff in plan.test_cutoffs:
        train, test = split_by_time(ratings, plan.split_at(cutoff))
        temporal_test_size = len(test) if cutoff == plan.leakage_cutoff else temporal_test_size
        run_arm(f"test/{cutoff}/{POPULARITY}", "test", POPULARITY, PopularityModel(), train, test)
        if has_eligible_users(train, test, plan.evaluation, UserGroup.NEW):
            run_arm(
                f"new-users/{cutoff}/{POPULARITY}",
                "new-users",
                POPULARITY,
                PopularityModel(),
                train,
                test,
                UserGroup.NEW,
            )
        for variant, seed in itertools.product(best, plan.seeds):
            config = _with_seed(best[variant], seed)
            arm_id = f"test/{cutoff}/{variant.value}/seed={seed}"
            run_arm(arm_id, "test", variant.value, AlsModel(config), train, test)

    leakage_split = plan.split_at(plan.leakage_cutoff)
    combined = ratings.take(
        (ratings.days >= np.datetime64(leakage_split.train_start))
        & (ratings.days < np.datetime64(leakage_split.test_end))
    )
    for seed in plan.leakage_seeds:
        train, test = split_at_random(combined, temporal_test_size, seed)
        run_arm(
            f"random/{POPULARITY}/seed={seed}", "random", POPULARITY, PopularityModel(), train, test
        )
        for variant in best:
            config = _with_seed(best[variant], seed)
            arm_id = f"random/{variant.value}/seed={seed}"
            run_arm(arm_id, "random", variant.value, AlsModel(config), train, test)

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "chosen_settings": {
            variant.value: _config_name(config) for variant, config in best.items()
        },
        "results": summarize(list(finished.values())),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each model within a stage, keeping the range across arms."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["stage"] != "tune":
            grouped.setdefault((record["stage"], record["model"]), []).append(record)
    return [
        {
            "stage": stage,
            "model": model,
            "arms": len(items),
            "ndcg_mean": statistics.mean(item["ndcg"] for item in items),
            "ndcg_min": min(item["ndcg"] for item in items),
            "ndcg_max": max(item["ndcg"] for item in items),
            "recall_mean": statistics.mean(item["recall"] for item in items),
            "recall_min": min(item["recall"] for item in items),
            "recall_max": max(item["recall"] for item in items),
            "fit_seconds_mean": statistics.mean(item["fit_seconds"] for item in items),
        }
        for (stage, model), items in sorted(grouped.items())
    ]


def _load_all_ratings(monthly_dir: Path, plan: BaselinePlan) -> RatingArrays:
    splits = [plan.split_at(day) for day in (plan.tuning_cutoff, *plan.test_cutoffs)]
    first_day = min(split.train_start for split in splits)
    end_day = max(split.test_end for split in splits)
    return load_ratings(monthly_dir, first_day, end_day)


def load_finished(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines]
    return {record["arm"]: record for record in records}


def make_record(
    arm_id: str,
    stage: str,
    model_name: str,
    result: EvaluationResult,
    fit_seconds: float,
    train: RatingArrays,
    test: RatingArrays,
    group: UserGroup,
) -> dict[str, Any]:
    return {
        "arm": arm_id,
        "stage": stage,
        "model": model_name,
        "user_group": group.value,
        "fit_seconds": fit_seconds,
        "train_rows": len(train),
        "test_rows": len(test),
        **asdict(result),
    }


def _config_name(config: AlsConfig) -> str:
    name = f"{config.variant.value}/reg={config.regularization}"
    if config.variant is AlsVariant.PREFERENCE:
        name += f"/confidence={config.confidence}"
    return name


def _with_seed(config: AlsConfig, seed: int) -> AlsConfig:
    return AlsConfig(
        config.variant,
        config.rank,
        config.regularization,
        config.confidence,
        config.iterations,
        seed,
    )
