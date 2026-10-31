"""The evaluation must find a known answer that was planted in synthetic data."""

from datetime import date

import numpy as np

from temporal_recsys.data.arrays import from_ratings
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig, EvaluationResult, evaluate
from temporal_recsys.models.als import AlsConfig, AlsModel, AlsVariant

N_FILMS = 200
CONFIG = SyntheticConfig(users=500, films=N_FILMS, ratings=40_000, noise=0.3)
TEST_START = date(2005, 10, 1)


def score_model_trained_on(first_day: date, end_day: date) -> EvaluationResult:
    ratings = from_ratings(generate_ratings(CONFIG).ratings)
    in_window = (ratings.days >= np.datetime64(first_day)) & (ratings.days < np.datetime64(end_day))
    train = ratings.take(in_window)
    test = ratings.take(ratings.days >= np.datetime64(TEST_START))

    model = AlsModel(AlsConfig(AlsVariant.RATING, rank=5, regularization=0.05, iterations=15))
    model.fit(train, N_FILMS)
    return evaluate(model, train, test, N_FILMS, EvaluationConfig())


def test_history_from_before_the_taste_change_is_clearly_worse() -> None:
    before_change = score_model_trained_on(date(2004, 1, 1), date(2005, 1, 1))
    after_change = score_model_trained_on(date(2005, 1, 1), TEST_START)

    gap = after_change.ndcg - before_change.ndcg
    combined_stderr = np.hypot(after_change.ndcg_stderr, before_change.ndcg_stderr)
    print(f"before {before_change.ndcg:.4f}, after {after_change.ndcg:.4f}")
    assert gap > 5 * combined_stderr
