import numpy as np
import pytest

from temporal_recsys.data.arrays import RatingArrays
from temporal_recsys.evaluation.ranking import rank_unseen
from temporal_recsys.models.factors import FactorModel
from temporal_recsys.models.fallback import PopularityFallback
from temporal_recsys.models.popularity import PopularityModel
from temporal_recsys.serving.bundle import build_bundle
from temporal_recsys.serving.recommender import Recommender, Source


def small_recommender() -> Recommender:
    factors = FactorModel(
        user_ids=np.array([10, 20], dtype=np.int32),
        user_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        film_factors=np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.0, 0.0]]),
    )
    window_films = np.array([2, 2, 3], dtype=np.int32)
    history_users = np.array([20, 10, 30, 10], dtype=np.int32)
    history_films = np.array([2, 3, 1, 1], dtype=np.int32)
    return Recommender(build_bundle(factors, window_films, history_users, history_films, 4))


@pytest.mark.parametrize(
    ("user_id", "source", "film_ids"),
    [
        # Scores 0.9, 0.1, 0.5, 0.0. Films 1 and 3 were rated, so 2 then 4 remain.
        (10, Source.MODEL, [2, 4]),
        # Scores 0.1, 0.9, 0.5, 0.0. Film 2 was rated, so 3 then 1 lead.
        (20, Source.MODEL, [3, 1]),
        # Rated only before the window: popularity 0, 2, 1, 0 without film 1.
        (30, Source.POPULAR, [2, 3]),
        # Never seen at all: plain popularity.
        (99, Source.POPULAR, [2, 3]),
    ],
)
def test_recommendations_match_hand_computed_lists(
    user_id: int, source: Source, film_ids: list[int]
) -> None:
    recommendation = small_recommender().recommend(user_id, k=2)

    assert recommendation.source is source
    assert recommendation.film_ids == film_ids


def test_service_lists_match_the_evaluation_scorer() -> None:
    """The service must return what the main experiment scored, user by user."""
    rng = np.random.default_rng(0)
    n_films, n_users = 30, 40
    history = RatingArrays(
        film_ids=rng.integers(1, n_films + 1, size=400).astype(np.int32),
        user_ids=rng.integers(1, n_users + 1, size=400).astype(np.int32),
        stars=np.full(400, 4, dtype=np.int8),
        days=np.full(400, np.datetime64("2005-01-01")),
    )
    window = history.take(history.user_ids <= n_users // 2)
    window_users = np.unique(window.user_ids)
    factors = FactorModel(
        window_users, rng.normal(size=(len(window_users), 3)), rng.normal(size=(n_films, 3))
    )
    popularity = PopularityModel()
    popularity.fit(window, n_films)
    scorer = PopularityFallback(factors, popularity, window_users)
    bundle = build_bundle(factors, window.film_ids, history.user_ids, history.film_ids, n_films)
    recommender = Recommender(bundle)

    users = np.arange(1, n_users + 6, dtype=np.int32)
    seen = [bundle.seen_for(int(user)) for user in users]
    expected = rank_unseen(scorer.score_users(users), seen, k=5)

    served = [recommender.recommend(int(user), k=5).film_ids for user in users]
    assert served == [[index + 1 for index in ranked] for ranked in expected]
