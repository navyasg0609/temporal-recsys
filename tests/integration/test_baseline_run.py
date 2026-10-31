from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from monthly_data import write_monthly

from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.baseline import (
    ARMS_FILE,
    BaselinePlan,
    load_plan,
    run_baseline,
    tuning_grid,
)
from temporal_recsys.models.als import AlsVariant

N_FILMS = 60
CONFIG = SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)
PLAN = BaselinePlan(
    id="test",
    window_months=6,
    tuning_cutoff=date(2005, 7, 1),
    test_cutoffs=(date(2005, 8, 1), date(2005, 9, 1)),
    test_months=1,
    seeds=(1, 2),
    n_films=N_FILMS,
    rank=3,
    iterations=2,
    rating_regularization=(0.05, 0.1),
    preference_regularization=(1.0,),
    preference_confidence=(10.0,),
    leakage_cutoff=date(2005, 9, 1),
    leakage_seeds=(1,),
    evaluation=EvaluationConfig(max_users=50),
)


@pytest.fixture(scope="module")
def monthly_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("monthly")
    write_monthly(generate_ratings(CONFIG).ratings, root)
    return root


def test_baseline_reports_every_stage_and_model(monthly_dir: Path, tmp_path: Path) -> None:
    summary = run_baseline(monthly_dir, PLAN, tmp_path, code_commit="abc")

    assert summary["code_commit"] == "abc"
    assert {(row["stage"], row["model"]) for row in summary["results"]} == {
        ("test", "popularity"),
        ("test", "rating"),
        ("test", "preference"),
        ("random", "popularity"),
        ("random", "rating"),
        ("random", "preference"),
    }


def test_each_als_arm_runs_once_per_cutoff_and_seed(monthly_dir: Path, tmp_path: Path) -> None:
    summary = run_baseline(monthly_dir, PLAN, tmp_path, code_commit="abc")

    arms = {(row["stage"], row["model"]): row["arms"] for row in summary["results"]}
    assert arms[("test", "popularity")] == 2
    assert arms[("test", "rating")] == 4
    assert arms[("random", "rating")] == 1


def test_restart_keeps_finished_arms(monthly_dir: Path, tmp_path: Path) -> None:
    run_baseline(monthly_dir, PLAN, tmp_path, code_commit="abc")
    finished = (tmp_path / ARMS_FILE).read_text(encoding="utf-8")

    run_baseline(monthly_dir, PLAN, tmp_path, code_commit="abc")

    assert (tmp_path / ARMS_FILE).read_text(encoding="utf-8") == finished


def test_leakage_cutoff_outside_the_test_cutoffs_raises() -> None:
    with pytest.raises(ValueError, match="leakage cutoff"):
        replace(PLAN, leakage_cutoff=date(2005, 3, 1))


def test_registered_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/baseline-table.yaml"))

    assert plan.window_months == 12
    assert plan.test_cutoffs == (date(2005, 8, 1), date(2005, 9, 1), date(2005, 10, 1))
    assert plan.evaluation.max_users == 50_000
    assert plan.leakage_cutoff in plan.test_cutoffs


def test_widened_plan_runs_preference_als_only() -> None:
    plan = load_plan(Path("experiments/registered/baseline-table-v2.yaml"))

    assert plan.rating_regularization == ()
    assert plan.leakage_seeds == ()
    assert all(config.variant is AlsVariant.PREFERENCE for config in tuning_grid(plan))
    assert len(list(tuning_grid(plan))) == 9
