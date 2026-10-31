"""Alternating Least Squares on a single machine.

ALS describes each user and film with a short list of numbers. It fixes the film
numbers and solves for the users, then swaps, and repeats. Each step is an ordinary
least squares problem with an exact answer.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from temporal_recsys.data.arrays import RatingArrays


class AlsVariant(Enum):
    RATING = "rating"
    """Predict the stars a user would give (Zhou et al., 2008)."""

    PREFERENCE = "preference"
    """Learn which films users chose to rate (Hu, Koren and Volinsky, 2008)."""


@dataclass(frozen=True)
class AlsConfig:
    variant: AlsVariant
    rank: int = 20
    regularization: float = 0.05
    confidence: float = 40.0
    iterations: int = 10
    seed: int = 0


class AlsModel:
    def __init__(self, config: AlsConfig) -> None:
        self.config = config
        self._user_ids: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._user_factors: NDArray[np.float64] = np.empty((0, config.rank))
        self._film_factors: NDArray[np.float64] = np.empty((0, config.rank))

    def fit(self, train: RatingArrays, n_films: int) -> None:
        config = self.config
        self._user_ids = np.unique(train.user_ids)
        rows = np.searchsorted(self._user_ids, train.user_ids)
        by_user = csr_matrix(
            (self._targets(train), (rows, train.film_ids - 1)),
            shape=(len(self._user_ids), n_films),
        )
        by_film = by_user.T.tocsr()

        rng = np.random.default_rng(config.seed)
        film_factors = rng.normal(scale=0.1, size=(n_films, config.rank))
        user_factors = np.zeros((len(self._user_ids), config.rank))
        for _ in range(config.iterations):
            user_factors = self._solve_rows(by_user, film_factors)
            film_factors = self._solve_rows(by_film, user_factors)
        self._user_factors = user_factors
        self._film_factors = film_factors

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        """Score every film. Users absent from training score zero for all films."""
        rows = np.searchsorted(self._user_ids, user_ids)
        rows = np.minimum(rows, len(self._user_ids) - 1)
        is_known = self._user_ids[rows] == user_ids
        factors = self._user_factors[rows] * is_known[:, None]
        scores: NDArray[np.float64] = factors @ self._film_factors.T
        return scores

    def _targets(self, train: RatingArrays) -> NDArray[np.float64]:
        if self.config.variant is AlsVariant.PREFERENCE:
            return np.ones(len(train))
        stars = train.stars.astype(np.float64)
        return stars - stars.mean()

    def _solve_rows(
        self, matrix: csr_matrix, other_factors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Solve for one side's factors while the other side stays fixed."""
        config = self.config
        rank_identity = np.eye(config.rank)
        solved = np.zeros((matrix.shape[0], config.rank))
        gram = other_factors.T @ other_factors

        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            if start == end:
                continue
            rated = other_factors[matrix.indices[start:end]]
            if config.variant is AlsVariant.PREFERENCE:
                system = (
                    gram
                    + config.confidence * rated.T @ rated
                    + config.regularization * rank_identity
                )
                target = (1 + config.confidence) * rated.sum(axis=0)
            else:
                # Scaling by the number of ratings is the weighted penalty of Zhou et al.
                penalty = config.regularization * (end - start)
                system = rated.T @ rated + penalty * rank_identity
                target = rated.T @ matrix.data[start:end]
            solved[row] = np.linalg.solve(system, target)
        return solved
