"""Train the model to serve, score it, and pack it into a serving bundle.

The model is one arm the main experiment already scored, retrained with the same plan,
so its quality is a measured number and the retrained score can be checked against it.
The bundle holds rating data, so it stays out of git and out of the serving image.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from temporal_recsys.data.arrays import load_ratings
from temporal_recsys.evaluation.harness import Scorer, evaluate
from temporal_recsys.evaluation.ranking import rank_unseen
from temporal_recsys.experiments.history import (
    TWO_TOWER,
    HistoryPlan,
    arm_data,
    arm_id,
    two_tower_trainer,
    with_fallback,
)
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.serving.bundle import build_bundle, save_bundle
from temporal_recsys.serving.recommender import Recommender

LOGGER = logging.getLogger(__name__)

BUNDLE_FILE = "bundle.npz"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ExportConfig:
    history_plan: Path
    cutoff: date
    window_months: int
    seed: int
    check_users: int
    check_seed: int


def load_config(path: Path) -> ExportConfig:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    check = config["serving_check"]
    return ExportConfig(
        history_plan=Path(config["history_plan"]),
        cutoff=date.fromisoformat(config["cutoff"]),
        window_months=config["window_months"],
        seed=config["seed"],
        check_users=check["users"],
        check_seed=check["seed"],
    )


def export_serving(
    monthly_dir: Path,
    raw_dir: Path,
    plan: HistoryPlan,
    config: ExportConfig,
    output_dir: Path,
    code_commit: str,
    device: str,
) -> dict[str, Any]:
    is_scored_arm = (
        config.cutoff in plan.test_cutoffs
        and config.window_months in plan.window_months
        and config.seed in plan.seeds
    )
    if not is_scored_arm:
        raise ValueError("Serve only an arm the main experiment scored")

    ratings = load_ratings(monthly_dir, plan.first_day, plan.history_split(config.cutoff).test_end)
    data = arm_data(plan, ratings, config.cutoff, config.window_months)
    name = arm_id(config.cutoff, TWO_TOWER, config.window_months, config.seed)
    trained = two_tower_trainer(raw_dir, plan, output_dir, device)(
        name, data.split, data.window, config.seed
    )
    if not isinstance(trained.scorer, FactorModel):
        raise TypeError("The serving bundle needs user and film vectors")
    scorer = with_fallback(trained.scorer, data.window, plan.n_films)
    result = evaluate(scorer, data.history, data.test, plan.n_films, plan.evaluation)
    LOGGER.info("%s: NDCG@10 %.4f", name, result.ndcg)

    bundle = build_bundle(
        trained.scorer,
        data.window.film_ids,
        data.history.user_ids,
        data.history.film_ids,
        plan.n_films,
    )
    save_bundle(bundle, output_dir / BUNDLE_FILE)

    rng = np.random.default_rng(config.check_seed)
    sample_size = min(config.check_users, len(bundle.seen_user_ids))
    check_users = np.sort(rng.choice(bundle.seen_user_ids, size=sample_size, replace=False))
    matching = count_matching_lists(scorer, Recommender(bundle), check_users, plan.evaluation.k)

    manifest = {
        "arm": name,
        "code_commit": code_commit,
        "device": device,
        "ndcg": result.ndcg,
        "recall": result.recall,
        "training_seconds": trained.fit_seconds,
        "window_rows": len(data.window),
        "history_rows": len(data.history),
        "users_with_model_vectors": len(bundle.user_ids),
        "users_with_history": len(bundle.seen_user_ids),
        "films": bundle.n_films,
        "dimensions": bundle.film_vectors.shape[1],
        "bundle_bytes": (output_dir / BUNDLE_FILE).stat().st_size,
        "lists_checked": len(check_users),
        "lists_matching_evaluation": matching,
    }
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def count_matching_lists(
    scorer: Scorer, recommender: Recommender, user_ids: np.ndarray, k: int
) -> int:
    """Count users whose served list equals the list the evaluation would rank."""
    bundle = recommender.bundle
    seen = [bundle.seen_for(int(user)) for user in user_ids]
    expected = rank_unseen(scorer.score_users(user_ids.astype(np.int32)), seen, k)
    return sum(
        recommender.recommend(int(user), k).film_ids == [index + 1 for index in ranked]
        for user, ranked in zip(user_ids, expected, strict=True)
    )
