from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
from monthly_data import write_monthly

from temporal_recsys.data.arrays import from_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.two_tower_diagnostic import (
    load_plan,
    run_diagnostic,
    share_of_relevant_ratings_on_unseen_films,
    without_films,
)
from temporal_recsys.models.als import AlsConfig, AlsVariant
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.models.two_tower import TwoTowerConfig

DAY = date(2005, 1, 1)
N_FILMS = 60


def test_removed_films_score_zero_and_the_original_is_unchanged() -> None:
    model = FactorModel(np.array([1], dtype=np.int32), np.array([[2.0]]), np.array([[1.0], [3.0]]))

    removed = without_films(model, np.array([True, False]))

    users = np.array([1], dtype=np.int32)
    assert removed.score_users(users).tolist() == [[0.0, 6.0]]
    assert model.score_users(users).tolist() == [[2.0, 6.0]]


def test_share_counts_only_relevant_ratings() -> None:
    train = from_ratings([Rating(1, 1, 5, DAY)])
    test = from_ratings(
        [
            Rating(1, 2, 5, DAY),
            Rating(2, 2, 4, DAY),
            Rating(3, 2, 1, DAY),
        ]
    )

    assert share_of_relevant_ratings_on_unseen_films(train, test, relevant_stars=4) == 0.5


def test_run_covers_every_explanation(tmp_path: Path) -> None:
    monthly = tmp_path / "monthly"
    config = SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)
    write_monthly(generate_ratings(config).ratings, monthly)
    titles = "".join(f"{film},2000,Film {film % 5}\n" for film in range(1, N_FILMS + 1))
    (tmp_path / "movie_titles.csv").write_text(titles, encoding="latin-1")
    plan = replace(
        load_plan(Path("experiments/registered/two-tower-diagnostic.yaml")),
        window_months=6,
        n_films=N_FILMS,
        als=AlsConfig(AlsVariant.PREFERENCE, rank=3, iterations=2, regularization=1.0),
        tower=TwoTowerConfig(dimensions=4, max_history=5, batch_size=64, epochs=1, seed=1),
        evaluation=EvaluationConfig(max_users=50),
    )

    summary = run_diagnostic(monthly, tmp_path, plan, tmp_path / "out", "abc", device="cpu")

    assert {arm["arm"] for arm in summary["arms"]} == {
        "popularity/months=12",
        "popularity/months=1",
        "popularity/months=3",
        "als/months=12",
        "als/months=1",
        "als/months=3",
        "two-tower/full",
        "two-tower/unseen-films-removed",
        "two-tower/id-only",
    }
    assert summary["latest_training_day"] < summary["first_test_day"]
