import numpy as np
import pytest

from temporal_recsys.evaluation.ranking import rank_unseen


def test_ranks_best_first_and_skips_seen_films() -> None:
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]])
    seen = [np.array([1]), np.array([], dtype=np.int64)]

    # Row 0 loses film index 1 (0.9), so 0.7, 0.5 lead. Row 1 keeps its order.
    assert rank_unseen(scores, seen, k=2) == [[3, 2], [0, 1]]


def test_k_outside_the_catalog_raises() -> None:
    scores = np.zeros((1, 3))
    with pytest.raises(ValueError, match="k must be"):
        rank_unseen(scores, [np.array([], dtype=np.int64)], k=4)


def test_seen_lists_must_match_users() -> None:
    with pytest.raises(ValueError, match="seen lists"):
        rank_unseen(np.zeros((2, 3)), [np.array([0])], k=1)
