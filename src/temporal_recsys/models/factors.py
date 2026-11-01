"""Score films from the short lists of numbers a matrix factorization model learned."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FactorModel:
    """User and film factors, whichever library trained them.

    `user_ids` is sorted and lines up with the rows of `user_factors`. Row n of
    `film_factors` belongs to film ID n + 1.
    """

    user_ids: NDArray[np.int32]
    user_factors: NDArray[np.float64]
    film_factors: NDArray[np.float64]

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        """Score every film. Users absent from training score zero for all films."""
        rows = np.searchsorted(self.user_ids, user_ids)
        rows = np.minimum(rows, len(self.user_ids) - 1)
        is_known = self.user_ids[rows] == user_ids
        factors = self.user_factors[rows] * is_known[:, None]
        scores: NDArray[np.float64] = factors @ self.film_factors.T
        return scores
