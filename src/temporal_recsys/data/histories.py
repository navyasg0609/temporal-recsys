"""Arrange each user's ratings in time order and build model inputs from them.

Dates have no time of day, and people often rate many films in one sitting. A training
input therefore holds only ratings from strictly earlier days, which matches the real
task of predicting a later period.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from temporal_recsys.data.arrays import RatingArrays

PADDING = 0


@dataclass(frozen=True)
class UserHistories:
    """All ratings sorted by user, then day.

    `row_starts` and `day_starts` give, for each row, where its user and its user's day
    begin. Inputs hold film ID values, because zero is reserved for padding.
    """

    user_ids: NDArray[np.int32]
    user_starts: NDArray[np.int64]
    film_ids: NDArray[np.int32]
    row_starts: NDArray[np.int64]
    day_starts: NDArray[np.int64]

    def training_rows(self) -> NDArray[np.int64]:
        """Return rows that have at least one rating on an earlier day."""
        return np.flatnonzero(self.day_starts > self.row_starts)


def build_histories(ratings: RatingArrays) -> UserHistories:
    if len(ratings) == 0:
        raise ValueError("Cannot build histories from no ratings")
    days = ratings.days.astype("datetime64[D]").astype(np.int64)
    order = np.lexsort((ratings.film_ids, days, ratings.user_ids))
    users, days = ratings.user_ids[order], days[order]

    is_new_user = np.r_[True, users[1:] != users[:-1]]
    is_new_day = is_new_user | np.r_[True, days[1:] != days[:-1]]
    rows = np.arange(len(users), dtype=np.int64)
    return UserHistories(
        users[is_new_user],
        rows[is_new_user],
        ratings.film_ids[order],
        np.maximum.accumulate(np.where(is_new_user, rows, 0)),
        np.maximum.accumulate(np.where(is_new_day, rows, 0)),
    )


def training_inputs(
    histories: UserHistories, rows: NDArray[np.int64], max_history: int
) -> NDArray[np.int64]:
    """Return the films each row's user rated on earlier days, most recent last."""
    return _recent_films(
        histories, histories.day_starts[rows], histories.row_starts[rows], max_history
    )


def latest_inputs(histories: UserHistories, max_history: int) -> NDArray[np.int64]:
    """Return each user's most recent films, one row per user."""
    user_ends = np.r_[histories.user_starts[1:], len(histories.film_ids)]
    return _recent_films(histories, user_ends, histories.user_starts, max_history)


def _recent_films(
    histories: UserHistories,
    ends: NDArray[np.int64],
    lowest: NDArray[np.int64],
    max_history: int,
) -> NDArray[np.int64]:
    positions = ends[:, None] + np.arange(-max_history, 0)
    is_inside = positions >= lowest[:, None]
    inputs = np.full(positions.shape, PADDING, dtype=np.int64)
    inputs[is_inside] = histories.film_ids[positions[is_inside]]
    return inputs
