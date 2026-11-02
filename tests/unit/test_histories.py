from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from temporal_recsys.data.arrays import RatingArrays, from_ratings
from temporal_recsys.data.histories import build_histories, latest_inputs, training_inputs
from temporal_recsys.data.parser import Rating

DAY_1 = date(2005, 1, 1)
DAY_2 = date(2005, 1, 2)
DAY_3 = date(2005, 1, 3)


def sample() -> RatingArrays:
    """User 5 rates 10 on day 1, then 20 and 30 on day 2, then 40 on day 3.

    User 2 rates 50 on day 1 and is listed last so that sorting has work to do.
    """
    return from_ratings(
        [
            Rating(40, 5, 3, DAY_3),
            Rating(30, 5, 3, DAY_2),
            Rating(10, 5, 3, DAY_1),
            Rating(20, 5, 3, DAY_2),
            Rating(50, 2, 3, DAY_1),
        ]
    )


def test_ratings_are_sorted_by_user_then_day() -> None:
    histories = build_histories(sample())

    assert histories.user_ids.tolist() == [2, 5]
    assert histories.film_ids.tolist() == [50, 10, 20, 30, 40]


def test_only_rows_with_an_earlier_day_can_train() -> None:
    histories = build_histories(sample())

    assert histories.training_rows().tolist() == [2, 3, 4]


def test_input_excludes_the_same_day() -> None:
    histories = build_histories(sample())

    inputs = training_inputs(histories, np.array([3, 4]), max_history=3)

    assert inputs.tolist() == [[0, 0, 10], [10, 20, 30]]


def test_input_keeps_only_the_most_recent_films() -> None:
    histories = build_histories(sample())

    assert training_inputs(histories, np.array([4]), max_history=2).tolist() == [[20, 30]]


def test_input_never_crosses_into_another_user() -> None:
    histories = build_histories(sample())

    assert training_inputs(histories, np.array([2]), max_history=3).tolist() == [[0, 0, 10]]


def test_latest_inputs_use_everything_up_to_the_end() -> None:
    histories = build_histories(sample())

    assert latest_inputs(histories, max_history=2).tolist() == [[0, 50], [30, 40]]


def test_no_ratings_raise() -> None:
    with pytest.raises(ValueError, match="no ratings"):
        build_histories(sample().take(np.zeros(5, dtype=np.bool_)))


@given(
    st.lists(
        st.tuples(st.integers(1, 3), st.integers(1, 6), st.integers(0, 4)),
        min_size=1,
        unique_by=lambda row: (row[0], row[1]),
    )
)
def test_every_input_film_was_rated_on_an_earlier_day(rows: list[tuple[int, int, int]]) -> None:
    day_of = {(user, film): day for user, film, day in rows}
    ratings = from_ratings(
        [Rating(film, user, 3, DAY_1 + timedelta(days=day)) for user, film, day in rows]
    )
    histories = build_histories(ratings)
    training = histories.training_rows()

    inputs = training_inputs(histories, training, max_history=5)

    users = np.repeat(histories.user_ids, np.diff(np.r_[histories.user_starts, len(rows)]))
    for row, films in zip(training.tolist(), inputs.tolist(), strict=True):
        user = int(users[row])
        target_day = day_of[(user, int(histories.film_ids[row]))]
        assert any(films)
        assert all(day_of[(user, film)] < target_day for film in films if film)
