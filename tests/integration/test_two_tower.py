from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from temporal_recsys.data.arrays import RatingArrays, from_ratings
from temporal_recsys.data.films import FilmFeatures, FilmInfo, build_film_features
from temporal_recsys.data.parser import Rating
from temporal_recsys.evaluation.harness import EvaluationConfig, evaluate
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel

N_FILMS = 41
UNSEEN_FILM = 41
FIRST_DAY = date(2005, 1, 1)
TEST_DAY = date(2005, 6, 15)
CONFIG = TwoTowerConfig(
    dimensions=8,
    max_history=10,
    batch_size=32,
    epochs=6,
    learning_rate=0.01,
    temperature=0.1,
    seed=1,
)


def taste_groups() -> tuple[RatingArrays, RatingArrays]:
    """Odd users pick films 1 to 20 and even users pick 21 to 40, one film a day.

    Each user skips one film of their group in training and picks it in the test
    period, so popularity cannot tell the groups apart but a personalized model can.
    """
    train: list[Rating] = []
    test: list[Rating] = []
    for user in range(1, 121):
        group = range(1, 21) if user % 2 else range(21, 41)
        held_out = group[user % len(group)]
        watched = [film for film in group if film != held_out]
        train += [
            Rating(film, user, 5, FIRST_DAY + timedelta(days=day))
            for day, film in enumerate(watched)
        ]
        test.append(Rating(held_out, user, 5, TEST_DAY))
    return from_ratings(train), from_ratings(test)


def features() -> FilmFeatures:
    films = [FilmInfo(film, 2000, f"Film {film % 3}") for film in range(1, N_FILMS + 1)]
    return build_film_features(films, N_FILMS)


def test_model_beats_popularity_on_a_known_answer() -> None:
    train, test = taste_groups()
    config = EvaluationConfig()
    model = TwoTowerModel(CONFIG, features())
    model.fit(train, N_FILMS)
    popularity = PopularityModel()
    popularity.fit(train, N_FILMS)

    model_score = evaluate(model, train, test, N_FILMS, config)
    popularity_score = evaluate(popularity, train, test, N_FILMS, config)

    print(f"two tower {model_score.ndcg:.4f}, popularity {popularity_score.ndcg:.4f}")
    assert model_score.ndcg > 2 * popularity_score.ndcg


@pytest.mark.parametrize("use_content", [True, False])
def test_restart_matches_an_unbroken_run(tmp_path: Path, use_content: bool) -> None:
    train, _ = taste_groups()
    config = replace(CONFIG, epochs=2, use_content=use_content)

    unbroken = TwoTowerModel(config, features())
    unbroken.fit(train, N_FILMS)

    checkpoint = tmp_path / "arm.pt"
    TwoTowerModel(replace(config, epochs=1), features(), checkpoint_path=checkpoint).fit(
        train, N_FILMS
    )
    restarted = TwoTowerModel(config, features(), checkpoint_path=checkpoint)
    restarted.fit(train, N_FILMS)

    assert np.array_equal(unbroken.factors.film_factors, restarted.factors.film_factors)
    assert np.array_equal(unbroken.factors.user_factors, restarted.factors.user_factors)


def test_finished_checkpoint_skips_training(tmp_path: Path) -> None:
    train, _ = taste_groups()
    config = replace(CONFIG, epochs=1)
    checkpoint = tmp_path / "arm.pt"
    first = TwoTowerModel(config, features(), checkpoint_path=checkpoint)
    first.fit(train, N_FILMS)

    second = TwoTowerModel(config, features(), checkpoint_path=checkpoint)
    second.fit(train, N_FILMS)

    assert second.training_seconds == first.training_seconds
    assert np.array_equal(first.factors.film_factors, second.factors.film_factors)


def test_unseen_film_is_scored_only_from_its_content() -> None:
    train, _ = taste_groups()

    with_content = TwoTowerModel(replace(CONFIG, epochs=1), features())
    with_content.fit(train, N_FILMS)
    id_only = TwoTowerModel(replace(CONFIG, epochs=1, use_content=False), features())
    id_only.fit(train, N_FILMS)

    assert np.any(with_content.factors.film_factors[UNSEEN_FILM - 1] != 0)
    assert np.all(id_only.factors.film_factors[UNSEEN_FILM - 1] == 0)


def test_too_few_rows_for_a_batch_raise() -> None:
    train, _ = taste_groups()
    config = replace(CONFIG, batch_size=1_000_000)

    with pytest.raises(ValueError, match="one batch"):
        TwoTowerModel(config, features()).fit(train, N_FILMS)


def test_scoring_needs_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TwoTowerModel(CONFIG, features()).score_users(np.array([1], dtype=np.int32))
