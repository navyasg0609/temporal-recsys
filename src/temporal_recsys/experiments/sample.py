"""Run the main method end to end on synthetic data with a planted answer.

This is a check that anyone can run on a laptop, not a measurement: it uses the same
monthly layout, windows, split, fallback and evaluation as the main experiment, and
confirms that they find a change planted at a known month. It trains popularity and
single-machine ALS, the two models that need neither Spark nor a GPU.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from temporal_recsys.data.arrays import write_monthly
from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.evaluation.split import add_months
from temporal_recsys.experiments.baseline import ARMS_FILE, load_finished
from temporal_recsys.experiments.history import (
    ALL_HISTORY,
    NO_SEED,
    NUMPY_ALS,
    POPULARITY,
    HistoryPlan,
    numpy_als_trainer,
    popularity_trainer,
    run_history,
    window_length,
)
from temporal_recsys.models.als import AlsVariant


@dataclass(frozen=True)
class SampleConfig:
    synthetic: SyntheticConfig
    plan: HistoryPlan
    als_variant: AlsVariant
    after_change_window: int

    def __post_init__(self) -> None:
        change = add_months(
            date(self.synthetic.first_year, 1, 1), self.synthetic.change_month_index
        )
        for cutoff in self.plan.test_cutoffs:
            if add_months(cutoff, -self.after_change_window) < change:
                raise ValueError(
                    f"A {self.after_change_window}-month window before {cutoff} reaches back "
                    "before the taste change, so it cannot be the known answer"
                )


def load_config(path: Path) -> SampleConfig:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, measurement, als = config["design"], config["measurement"], config["design"]["als"]
    synthetic = SyntheticConfig(**config["synthetic"])
    plan = HistoryPlan(
        id="sample",
        first_day=date.fromisoformat(design["first_day"]),
        window_months=tuple(design["window_months"]),
        test_cutoffs=tuple(date.fromisoformat(day) for day in design["test_cutoffs"]),
        test_months=design["test_months"],
        seeds=tuple(design["seeds"]),
        n_films=synthetic.films,
        als_rank=als["rank"],
        als_regularization=als["regularization"],
        # Rating ALS has no confidence weight.
        als_confidence=0.0,
        als_iterations=als["iterations"],
        # The sample trains no two-tower model.
        tower_dimensions=0,
        tower_max_history=0,
        tower_batch_size=0,
        tower_epochs=0,
        tower_learning_rate=0.0,
        tower_temperature=0.0,
        evaluation=EvaluationConfig(
            k=measurement["k"],
            relevant_stars=measurement["relevant_stars"],
            seed=measurement["evaluation_seed"],
        ),
    )
    return SampleConfig(
        synthetic,
        plan,
        AlsVariant(als["variant"]),
        config["known_answer"]["after_change_window"],
    )


def run_sample(config: SampleConfig, output_dir: Path, code_commit: str) -> dict[str, Any]:
    """Write the synthetic data, train both models on every window, and check the answer."""
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it or choose another")
    monthly_dir = output_dir / MONTHLY_DIR
    write_monthly(generate_ratings(config.synthetic).ratings, monthly_dir)

    plan = config.plan
    runs = (
        (POPULARITY, popularity_trainer(plan), (NO_SEED,)),
        (NUMPY_ALS, numpy_als_trainer(plan, config.als_variant), plan.seeds),
    )
    results, records = [], []
    for model_name, trainer, seeds in runs:
        model_dir = output_dir / model_name
        summary = run_history(
            monthly_dir, plan, model_name, trainer, seeds, model_dir, code_commit, {}
        )
        results += sorted(summary["results"], key=lambda row: window_length(str(row["window"])))
        records += list(load_finished(model_dir / ARMS_FILE).values())

    return {
        "code_commit": code_commit,
        "results": results,
        "planted_answer_found": planted_answer_found(records, config.after_change_window),
        "after_change_window": config.after_change_window,
    }


def planted_answer_found(records: Sequence[dict[str, Any]], after_change_window: int) -> bool:
    """True if ALS on the window after the change beats ALS on all history at every cutoff."""
    by_cutoff: dict[str, dict[str, list[float]]] = {}
    for record in records:
        if record["model"] == NUMPY_ALS:
            windows = by_cutoff.setdefault(record["cutoff"], {})
            windows.setdefault(str(record["window"]), []).append(record["ndcg"])
    if not by_cutoff:
        raise ValueError("No ALS results to check")
    return all(
        statistics.mean(windows[str(after_change_window)]) > statistics.mean(windows[ALL_HISTORY])
        for windows in by_cutoff.values()
    )
