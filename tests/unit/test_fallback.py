from datetime import date

import numpy as np

from temporal_recsys.data.arrays import from_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.models.fallback import PopularityFallback
from temporal_recsys.models.popularity import PopularityModel

DAY = date(2005, 1, 1)


def fallback() -> PopularityFallback:
    popularity = PopularityModel()
    popularity.fit(from_ratings([Rating(2, 1, 5, DAY), Rating(2, 3, 5, DAY)]), n_films=2)
    known = np.array([3], dtype=np.int32)
    model = FactorModel(known, np.array([[2.0]]), np.array([[1.0], [0.5]]))
    return PopularityFallback(model, popularity, known)


def test_known_user_keeps_the_model_scores() -> None:
    scores = fallback().score_users(np.array([3], dtype=np.int32))

    assert scores.tolist() == [[2.0, 1.0]]


def test_unknown_user_gets_popularity_counts() -> None:
    scores = fallback().score_users(np.array([9], dtype=np.int32))

    assert scores.tolist() == [[0.0, 2.0]]


def test_mixed_batch_keeps_each_user_in_place() -> None:
    scores = fallback().score_users(np.array([9, 3], dtype=np.int32))

    assert scores.tolist() == [[0.0, 2.0], [2.0, 1.0]]
