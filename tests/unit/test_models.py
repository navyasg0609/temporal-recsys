from datetime import date

import numpy as np
import pytest

from temporal_recsys.data.arrays import RatingArrays, from_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.models.als import AlsConfig, AlsModel, AlsVariant
from temporal_recsys.models.popularity import PopularityModel

DAY = date(2005, 1, 1)
N_FILMS = 40


def two_taste_groups() -> RatingArrays:
    """Odd users love films 1 to 20 and even users love 21 to 40.

    Users 1 and 2 skip the last film of their group, which a good model should suggest.
    """
    ratings: list[Rating] = []
    for user in range(1, 61):
        group = range(1, 21) if user % 2 else range(21, 41)
        loved = group[:-1] if user <= 2 else group
        ratings += [Rating(film, user, 5, DAY) for film in loved]
        disliked = range(21, 26) if user % 2 else range(1, 6)
        ratings += [Rating(film, user, 1, DAY) for film in disliked]
    return from_ratings(ratings)


def test_popularity_ranks_by_training_counts() -> None:
    train = from_ratings([Rating(3, 1, 5, DAY), Rating(3, 2, 1, DAY), Rating(1, 1, 5, DAY)])
    model = PopularityModel()

    model.fit(train, n_films=4)

    assert model.score_users(np.array([7], dtype=np.int32)).tolist() == [[1.0, 0.0, 2.0, 0.0]]


def test_popularity_learns_only_from_the_data_it_is_given() -> None:
    model = PopularityModel()
    model.fit(from_ratings([Rating(1, 1, 5, DAY)]), n_films=2)

    scores = model.score_users(np.array([1], dtype=np.int32))

    assert scores[0, 1] == 0.0


def test_popularity_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        PopularityModel().score_users(np.array([1], dtype=np.int32))


@pytest.mark.parametrize(
    ("variant", "regularization"),
    [(AlsVariant.RATING, 0.01), (AlsVariant.PREFERENCE, 1.0)],
)
def test_als_suggests_the_unseen_film_of_the_right_group(
    variant: AlsVariant, regularization: float
) -> None:
    config = AlsConfig(variant=variant, rank=4, regularization=regularization, confidence=10)
    model = AlsModel(config)
    model.fit(two_taste_groups(), N_FILMS)

    odd_user, even_user = model.score_users(np.array([1, 2], dtype=np.int32))

    assert odd_user[20 - 1] > odd_user[40 - 1]
    assert even_user[40 - 1] > even_user[20 - 1]


def test_als_gives_the_same_result_with_the_same_seed() -> None:
    config = AlsConfig(variant=AlsVariant.RATING, rank=3, iterations=2)
    first, second = AlsModel(config), AlsModel(config)
    users = np.array([1], dtype=np.int32)

    first.fit(two_taste_groups(), N_FILMS)
    second.fit(two_taste_groups(), N_FILMS)

    assert first.score_users(users).tolist() == second.score_users(users).tolist()


def test_als_gives_unknown_users_zero_scores() -> None:
    model = AlsModel(AlsConfig(variant=AlsVariant.RATING, rank=3, iterations=1))
    model.fit(two_taste_groups(), N_FILMS)

    scores = model.score_users(np.array([999], dtype=np.int32))

    assert np.all(scores == 0)
