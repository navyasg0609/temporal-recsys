"""Recommend the most rated films to everyone."""

import numpy as np
from numpy.typing import NDArray

from temporal_recsys.data.arrays import RatingArrays


class PopularityModel:
    """The baseline every personalized model must beat."""

    def __init__(self) -> None:
        self._rating_counts: NDArray[np.float64] | None = None

    def fit(self, train: RatingArrays, n_films: int) -> None:
        counts = np.bincount(train.film_ids - 1, minlength=n_films)
        self._rating_counts = counts.astype(np.float64)

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        if self._rating_counts is None:
            raise RuntimeError("Call fit before scoring")
        return np.tile(self._rating_counts, (len(user_ids), 1))
