import math

import pytest

from temporal_recsys.evaluation.metrics import ndcg_at_k, recall_at_k


def test_recall_counts_hits_inside_the_top_k() -> None:
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=3) == pytest.approx(1 / 3)


def test_recall_ignores_hits_after_the_top_k() -> None:
    assert recall_at_k([1, 2, 3], {3}, k=2) == 0.0


def test_recall_is_one_for_a_perfect_list_with_more_relevant_films_than_k() -> None:
    assert recall_at_k([1, 2], {1, 2, 3, 4}, k=2) == 1.0


def test_ndcg_is_one_for_a_perfect_order() -> None:
    assert ndcg_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computed_answer() -> None:
    # Hits at positions 1 and 3 give 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5.
    # The best order for two relevant films gives 1 + 1/log2(3) = 1.63093.
    expected = 1.5 / (1 + 1 / math.log2(3))

    assert ndcg_at_k([7, 8, 9], {7, 9}, k=3) == pytest.approx(expected)
    assert ndcg_at_k([7, 8, 9], {7, 9}, k=3) == pytest.approx(0.91972, abs=1e-5)


def test_ndcg_rewards_placing_a_hit_higher() -> None:
    assert ndcg_at_k([1, 2], {1}, k=2) > ndcg_at_k([2, 1], {1}, k=2)


def test_ndcg_is_zero_with_no_hits() -> None:
    assert ndcg_at_k([1, 2], {3}, k=2) == 0.0
