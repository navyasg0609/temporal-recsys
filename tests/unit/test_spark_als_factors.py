import numpy as np
import pytest

from temporal_recsys.models.spark_als import factors_from_rows


def test_rows_are_sorted_by_user_and_placed_by_film() -> None:
    model = factors_from_rows(
        user_ids=[9, 2],
        user_vectors=[[1.0, 1.0], [2.0, 0.0]],
        film_ids=[3, 1],
        film_vectors=[[0.0, 5.0], [1.0, 0.0]],
        n_films=3,
    )

    assert model.user_ids.tolist() == [2, 9]
    assert model.user_factors.tolist() == [[2.0, 0.0], [1.0, 1.0]]
    assert model.film_factors.tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 5.0]]


def test_film_missing_from_the_window_scores_zero() -> None:
    model = factors_from_rows([1], [[2.0]], [2], [[3.0]], n_films=2)

    assert model.score_users(np.array([1], dtype=np.int32)).tolist() == [[0.0, 6.0]]


@pytest.mark.parametrize("film_id", [0, 3])
def test_film_outside_the_catalog_raises(film_id: int) -> None:
    with pytest.raises(ValueError, match="Film IDs"):
        factors_from_rows([1], [[1.0]], [film_id], [[1.0]], n_films=2)
