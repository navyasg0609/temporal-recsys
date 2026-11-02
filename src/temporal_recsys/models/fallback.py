"""Serve popular films to users a model never saw."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from temporal_recsys.evaluation.harness import Scorer


@dataclass(frozen=True)
class PopularityFallback:
    """Score users known to `model` with it, and everyone else with `popularity`."""

    model: Scorer
    popularity: Scorer
    known_user_ids: NDArray[np.int32]

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        scores = self.model.score_users(user_ids)
        is_unknown = ~np.isin(user_ids, self.known_user_ids)
        if is_unknown.any():
            scores[is_unknown] = self.popularity.score_users(user_ids[is_unknown])
        return scores
