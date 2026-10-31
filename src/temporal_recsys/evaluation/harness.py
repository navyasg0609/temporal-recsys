"""Score any recommender the same way, ranking it against the whole catalog."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from temporal_recsys.data.arrays import RatingArrays
from temporal_recsys.evaluation.metrics import ndcg_at_k, recall_at_k


class Recommender(Protocol):
    def fit(self, train: RatingArrays, n_films: int) -> None: ...

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        """Return one row per user and one column per film, indexed by film ID minus 1."""
        ...


class UserGroup(Enum):
    KNOWN = "known"
    NEW = "new"


@dataclass(frozen=True)
class EvaluationConfig:
    k: int = 10
    relevant_stars: int = 4
    max_users: int = 50_000
    seed: int = 0
    batch_size: int = 1_000


@dataclass(frozen=True)
class EvaluationResult:
    ndcg: float
    recall: float
    ndcg_stderr: float
    recall_stderr: float
    users_evaluated: int
    users_eligible: int


def has_eligible_users(
    train: RatingArrays, test: RatingArrays, config: EvaluationConfig, group: UserGroup
) -> bool:
    relevant_rows = test.take(test.stars >= config.relevant_stars)
    return len(_eligible_users(relevant_rows, train, group)) > 0


def evaluate(
    model: Recommender,
    train: RatingArrays,
    test: RatingArrays,
    n_films: int,
    config: EvaluationConfig,
    group: UserGroup = UserGroup.KNOWN,
) -> EvaluationResult:
    """Score a fitted model on users with at least one relevant test film.

    Films a user rated in training are removed from their list. They cannot be
    recommended again, and leaving them in would hide real recommendations.
    """
    relevant_rows = test.take(test.stars >= config.relevant_stars)
    users = _eligible_users(relevant_rows, train, group)
    if len(users) == 0:
        raise ValueError(f"No eligible {group.value} users in the test set")

    rng = np.random.default_rng(config.seed)
    sample_size = min(config.max_users, len(users))
    sampled = np.sort(rng.choice(users, size=sample_size, replace=False))
    relevant_by_user = _films_by_user(relevant_rows, sampled)
    seen = _seen_matrix(train, sampled, n_films)

    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    for start in range(0, len(sampled), config.batch_size):
        batch = sampled[start : start + config.batch_size]
        scores = model.score_users(batch)
        for row in range(len(batch)):
            scores[row, seen[start + row].indices] = -np.inf
        for user_id, ranked in zip(batch.tolist(), _top_k(scores, config.k), strict=True):
            relevant = relevant_by_user[user_id]
            ndcg_scores.append(ndcg_at_k(ranked, relevant, config.k))
            recall_scores.append(recall_at_k(ranked, relevant, config.k))

    return EvaluationResult(
        ndcg=float(np.mean(ndcg_scores)),
        recall=float(np.mean(recall_scores)),
        ndcg_stderr=_standard_error(ndcg_scores),
        recall_stderr=_standard_error(recall_scores),
        users_evaluated=len(sampled),
        users_eligible=len(users),
    )


def _eligible_users(
    relevant_rows: RatingArrays, train: RatingArrays, group: UserGroup
) -> NDArray[np.int32]:
    users = np.unique(relevant_rows.user_ids)
    is_known = np.isin(users, train.user_ids)
    return users[is_known if group is UserGroup.KNOWN else ~is_known]


def _films_by_user(rows: RatingArrays, users: NDArray[np.int32]) -> dict[int, set[int]]:
    selected = rows.take(np.isin(rows.user_ids, users))
    films_by_user: dict[int, set[int]] = {}
    for user_id, film_id in zip(
        selected.user_ids.tolist(), (selected.film_ids - 1).tolist(), strict=True
    ):
        films_by_user.setdefault(user_id, set()).add(film_id)
    return films_by_user


def _seen_matrix(train: RatingArrays, users: NDArray[np.int32], n_films: int) -> csr_matrix:
    selected = train.take(np.isin(train.user_ids, users))
    rows = np.searchsorted(users, selected.user_ids)
    values = np.ones(len(selected), dtype=np.bool_)
    return csr_matrix(
        (values, (rows, selected.film_ids - 1)), shape=(len(users), n_films), dtype=np.bool_
    )


def _top_k(scores: NDArray[np.float64], k: int) -> list[list[int]]:
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    ranked: list[list[int]] = np.take_along_axis(candidates, order, axis=1).tolist()
    return ranked


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))
