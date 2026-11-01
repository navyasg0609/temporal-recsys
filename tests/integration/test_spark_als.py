from datetime import date
from pathlib import Path

import numpy as np
import pytest
from monthly_data import write_monthly
from pyspark.sql import SparkSession

from temporal_recsys.data.arrays import load_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.evaluation.harness import EvaluationConfig, evaluate
from temporal_recsys.evaluation.split import TemporalSplit, split_by_time
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.models.spark_als import (
    SparkAlsConfig,
    collect_factors,
    fit_spark_als,
    training_window,
)

pytestmark = pytest.mark.spark

N_FILMS = 40
TRAIN_DAY = date(2005, 3, 10)
SPLIT = TemporalSplit(date(2005, 1, 1), date(2005, 6, 1), date(2005, 7, 1))


def taste_groups() -> list[Rating]:
    """Odd users pick films 1 to 20 and even users pick 21 to 40.

    Each user skips one film of their group in training and picks it in the test month,
    so popularity cannot tell the groups apart but a personalized model can.
    """
    ratings: list[Rating] = []
    for user in range(1, 121):
        group = range(1, 21) if user % 2 else range(21, 41)
        held_out = group[user % len(group)]
        ratings += [Rating(film, user, 5, TRAIN_DAY) for film in group if film != held_out]
        ratings.append(Rating(held_out, user, 5, date(2005, 6, 15)))
    return ratings


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("spark_als")
    write_monthly(taste_groups(), root / MONTHLY_DIR)
    return root


def test_training_window_stops_before_the_cutoff(spark: SparkSession, data_dir: Path) -> None:
    window = training_window(spark, data_dir, MONTHLY_DIR, SPLIT)

    days = [row["rated_on"] for row in window.select("rated_on").collect()]
    assert days
    assert max(days) < SPLIT.cutoff


def test_training_window_matches_the_rows_used_for_evaluation(
    spark: SparkSession, data_dir: Path
) -> None:
    ratings = load_ratings(data_dir / MONTHLY_DIR, SPLIT.train_start, SPLIT.test_end)
    train, _ = split_by_time(ratings, SPLIT)

    assert training_window(spark, data_dir, MONTHLY_DIR, SPLIT).count() == len(train)


def test_training_window_rejects_a_mid_month_cutoff(spark: SparkSession, data_dir: Path) -> None:
    split = TemporalSplit(date(2005, 1, 1), date(2005, 6, 15), date(2005, 7, 1))

    with pytest.raises(ValueError, match="first of a month"):
        training_window(spark, data_dir, MONTHLY_DIR, split)


def test_spark_als_beats_popularity_on_a_known_answer(spark: SparkSession, data_dir: Path) -> None:
    ratings = load_ratings(data_dir / MONTHLY_DIR, SPLIT.train_start, SPLIT.test_end)
    train, test = split_by_time(ratings, SPLIT)
    config = EvaluationConfig()
    window = training_window(spark, data_dir, MONTHLY_DIR, SPLIT)

    spark_model = fit_spark_als(window, SparkAlsConfig(rank=4, regularization=0.01, seed=1))
    popularity = PopularityModel()
    popularity.fit(train, N_FILMS)

    als_score = evaluate(collect_factors(spark_model, N_FILMS), train, test, N_FILMS, config)
    popularity_score = evaluate(popularity, train, test, N_FILMS, config)
    print(f"spark {als_score.ndcg:.4f}, popularity {popularity_score.ndcg:.4f}")
    assert als_score.ndcg > 0.5
    assert als_score.ndcg > popularity_score.ndcg


def test_same_seed_gives_the_same_factors(spark: SparkSession, data_dir: Path) -> None:
    window = training_window(spark, data_dir, MONTHLY_DIR, SPLIT)
    config = SparkAlsConfig(rank=3, iterations=2, seed=7)

    first = collect_factors(fit_spark_als(window, config), N_FILMS)
    second = collect_factors(fit_spark_als(window, config), N_FILMS)

    assert np.allclose(first.film_factors, second.film_factors)
