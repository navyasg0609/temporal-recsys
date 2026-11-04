from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from temporal_recsys.models.factors import FactorModel
from temporal_recsys.serving.bundle import ServingBundle, build_bundle, load_bundle, save_bundle

N_FILMS = 4


def small_bundle() -> ServingBundle:
    factors = FactorModel(
        user_ids=np.array([10, 20], dtype=np.int32),
        user_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        film_factors=np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.0, 0.0]]),
    )
    # The window holds two ratings of film 2 and one of film 3.
    window_films = np.array([2, 2, 3], dtype=np.int32)
    # User 30 rated only before the window, so the model has no vector for them.
    history_users = np.array([20, 10, 30, 10], dtype=np.int32)
    history_films = np.array([2, 3, 1, 1], dtype=np.int32)
    return build_bundle(factors, window_films, history_users, history_films, N_FILMS)


def test_build_groups_rated_films_by_user() -> None:
    bundle = small_bundle()

    assert bundle.seen_user_ids.tolist() == [10, 20, 30]
    assert bundle.seen_for(10).tolist() == [0, 2]
    assert bundle.seen_for(20).tolist() == [1]
    assert bundle.seen_for(30).tolist() == [0]
    assert bundle.seen_for(99).tolist() == []
    assert bundle.popularity.tolist() == [0.0, 2.0, 1.0, 0.0]


def test_user_row_finds_only_users_the_model_saw() -> None:
    bundle = small_bundle()

    assert bundle.user_row(20) == 1
    assert bundle.user_row(30) is None


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    bundle = small_bundle()
    path = tmp_path / "bundle.npz"

    save_bundle(bundle, path)
    loaded = load_bundle(path)

    for name in ServingBundle.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(bundle, name))


def test_wrong_format_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bundle.npz"
    save_bundle(small_bundle(), path)
    with np.load(path) as arrays:
        contents = dict(arrays)
    contents["format_version"] = np.array(99)
    np.savez(path, **contents)

    with pytest.raises(ValueError, match="format 99"):
        load_bundle(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"user_vectors": np.zeros((2, 3))}, "user vectors of shape"),
        ({"popularity": np.zeros(3)}, "popularity scores"),
        ({"user_ids": np.array([20, 10], dtype=np.int32)}, "user_ids must be sorted"),
        ({"seen_starts": np.array([0, 2, 3], dtype=np.int64)}, "one more entry"),
        ({"seen_starts": np.array([0, 2, 3, 3], dtype=np.int64)}, "from 0 to the number"),
        ({"seen_films": np.array([0, 2, 1, 4], dtype=np.int32)}, "lie in 0 to 3"),
    ],
)
def test_inconsistent_bundles_are_rejected(change: dict[str, np.ndarray], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(small_bundle(), **change)
