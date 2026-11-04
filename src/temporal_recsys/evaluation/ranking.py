"""Turn scores into ranked lists, leaving out films each user already rated.

Evaluation and serving both rank through this function, so the lists a service returns
are the lists that were measured.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def rank_unseen(
    scores: NDArray[np.float64], seen: Sequence[NDArray[np.integer]], k: int
) -> list[list[int]]:
    """Return the top `k` film indices per row, best first, never one listed in `seen`.

    `scores` has one row per user and one column per film, indexed by film ID minus 1.
    It is changed in place: seen films are set to minus infinity.
    """
    if len(seen) != len(scores):
        raise ValueError(f"Got {len(seen)} seen lists for {len(scores)} users")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be between 1 and {scores.shape[1]}, got {k}")
    for row, films in enumerate(seen):
        scores[row, films] = -np.inf
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    ranked: list[list[int]] = np.take_along_axis(candidates, order, axis=1).tolist()
    return ranked
