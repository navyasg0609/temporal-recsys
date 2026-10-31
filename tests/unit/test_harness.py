from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from temporal_recsys.data.arrays import RatingArrays, from_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.evaluation.harness import EvaluationConfig, UserGroup, evaluate

N_FILMS = 100
DAY = date(2005, 1, 1)


class FixedScores:
    """Scores films from a table, so tests control the ranking exactly."""

    def __init__(self, scores_by_user: dict[int, NDArray[np.float64]]) -> None:
        self.scores_by_user = scores_by_user

    def fit(self, train: RatingArrays, n_films: int) -> None:
        pass

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        return np.array([self.scores_by_user[user].copy() for user in user_ids.tolist()])


def favoring(*film_ids: int) -> NDArray[np.float64]:
    scores = np.zeros(N_FILMS)
    for rank, film_id in enumerate(film_ids):
        scores[film_id - 1] = len(film_ids) - rank
    return scores


def test_a_perfect_model_scores_one() -> None:
    train = from_ratings([Rating(1, 1, 5, DAY)])
    test = from_ratings([Rating(2, 1, 5, DAY), Rating(3, 1, 4, DAY)])

    result = evaluate(FixedScores({1: favoring(2, 3)}), train, test, N_FILMS, EvaluationConfig(k=2))

    assert result.ndcg == pytest.approx(1.0)
    assert result.recall == 1.0


def test_low_star_films_do_not_count_as_relevant() -> None:
    train = from_ratings([Rating(1, 1, 5, DAY)])
    test = from_ratings([Rating(2, 1, 3, DAY), Rating(3, 1, 4, DAY)])

    result = evaluate(FixedScores({1: favoring(2, 4)}), train, test, N_FILMS, EvaluationConfig(k=2))

    assert result.recall == 0.0


def test_films_seen_in_training_are_removed_from_the_ranking() -> None:
    train = from_ratings([Rating(5, 1, 5, DAY)])
    test = from_ratings([Rating(6, 1, 5, DAY)])

    result = evaluate(FixedScores({1: favoring(5, 6)}), train, test, N_FILMS, EvaluationConfig(k=1))

    assert result.recall == 1.0


def test_random_scores_match_chance() -> None:
    n_users = 3_000
    rng = np.random.default_rng(0)
    train = from_ratings([Rating(1, user, 5, DAY) for user in range(1, n_users + 1)])
    test = from_ratings(
        [Rating(int(rng.integers(2, N_FILMS + 1)), user, 5, DAY) for user in range(1, n_users + 1)]
    )
    model = FixedScores({user: rng.random(N_FILMS) for user in range(1, n_users + 1)})

    result = evaluate(model, train, test, N_FILMS, EvaluationConfig(k=10))

    # Each user has 1 relevant film among 99 unseen, so 10 picks hit it 10/99 of the time.
    assert result.recall == pytest.approx(10 / 99, abs=4 * result.recall_stderr)


def test_user_groups_are_separated() -> None:
    train = from_ratings([Rating(1, 1, 5, DAY)])
    test = from_ratings([Rating(2, 1, 5, DAY), Rating(2, 2, 5, DAY)])
    model = FixedScores({1: favoring(2), 2: favoring(3)})

    known = evaluate(model, train, test, N_FILMS, EvaluationConfig(k=1), UserGroup.KNOWN)
    new = evaluate(model, train, test, N_FILMS, EvaluationConfig(k=1), UserGroup.NEW)

    assert (known.recall, new.recall) == (1.0, 0.0)


def test_user_sample_is_limited_and_repeatable() -> None:
    train = from_ratings([Rating(1, user, 5, DAY) for user in range(1, 51)])
    test = from_ratings([Rating(2, user, 5, DAY) for user in range(1, 51)])
    model = FixedScores({user: favoring(2) for user in range(1, 51)})
    config = EvaluationConfig(k=1, max_users=10)

    first = evaluate(model, train, test, N_FILMS, config)
    second = evaluate(model, train, test, N_FILMS, config)

    assert first == second
    assert (first.users_evaluated, first.users_eligible) == (10, 50)


def test_no_eligible_users_raises() -> None:
    train = from_ratings([Rating(1, 1, 5, DAY)])
    test = from_ratings([Rating(2, 1, 1, DAY)])

    with pytest.raises(ValueError, match="No eligible"):
        evaluate(FixedScores({}), train, test, N_FILMS, EvaluationConfig())
