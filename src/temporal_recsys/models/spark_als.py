"""Preference-weighted ALS trained by Spark and scored like every other model.

Spark's penalty grows with how many ratings each user or film has, while the NumPy
model's penalty is fixed. The same `regularization` value therefore means different
things in the two models, so each gets its own settings.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from temporal_recsys.data.benchmark import select_window
from temporal_recsys.evaluation.split import TemporalSplit
from temporal_recsys.models.factors import FactorModel

CHOSE_TO_RATE = "chose_to_rate"


@dataclass(frozen=True)
class SparkAlsConfig:
    rank: int = 20
    regularization: float = 0.05
    confidence: float = 10.0
    iterations: int = 10
    seed: int = 0


def training_window(
    spark: SparkSession, data_dir: Path, layout: str, split: TemporalSplit
) -> DataFrame:
    """Select the training months, which stop before the cutoff."""
    for day in (split.train_start, split.cutoff):
        _require_first_of_month(day)
    return select_window(spark, data_dir, layout, split.train_start, split.cutoff)


def fit_spark_als(window: DataFrame, config: SparkAlsConfig) -> ALSModel:
    # Every rating counts as 1, because this model learns which films people chose to
    # rate, not how many stars they gave.
    chosen = window.select("user_id", "film_id").withColumn(CHOSE_TO_RATE, F.lit(1.0))
    als = ALS(
        rank=config.rank,
        maxIter=config.iterations,
        regParam=config.regularization,
        implicitPrefs=True,
        alpha=config.confidence,
        userCol="user_id",
        itemCol="film_id",
        ratingCol=CHOSE_TO_RATE,
        seed=config.seed,
    )
    return als.fit(chosen)


def collect_factors(model: ALSModel, n_films: int) -> FactorModel:
    users = model.userFactors.collect()
    films = model.itemFactors.collect()
    return factors_from_rows(
        [row["id"] for row in users],
        [row["features"] for row in users],
        [row["id"] for row in films],
        [row["features"] for row in films],
        n_films,
    )


def factors_from_rows(
    user_ids: Sequence[int],
    user_vectors: Sequence[Sequence[float]],
    film_ids: Sequence[int],
    film_vectors: Sequence[Sequence[float]],
    n_films: int,
) -> FactorModel:
    """Arrange Spark's unordered rows the way `FactorModel` expects.

    Spark returns only films that appear in the window. The rest keep zeros, so they
    score zero for everyone.
    """
    ids = np.array(user_ids, dtype=np.int32)
    order = np.argsort(ids)
    film_rows: NDArray[np.int64] = np.array(film_ids, dtype=np.int64) - 1
    if len(film_rows) and not 0 <= film_rows.min() <= film_rows.max() < n_films:
        raise ValueError(f"Film IDs must be between 1 and {n_films}")

    vectors = np.array(film_vectors, dtype=np.float64)
    film_factors = np.zeros((n_films, vectors.shape[1]))
    film_factors[film_rows] = vectors
    return FactorModel(ids[order], np.array(user_vectors, dtype=np.float64)[order], film_factors)


def _require_first_of_month(day: date) -> None:
    # Month folders cannot hold part of a month, so a mid-month day would be silently
    # rounded to a whole month.
    if day.day != 1:
        raise ValueError(f"Spark training windows must start on the first of a month: {day}")
