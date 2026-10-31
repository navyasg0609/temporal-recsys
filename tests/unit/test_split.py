from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from temporal_recsys.data.arrays import RatingArrays, from_ratings
from temporal_recsys.data.parser import Rating
from temporal_recsys.evaluation.split import (
    TemporalSplit,
    add_months,
    split_at_random,
    split_by_time,
)

SPLIT = TemporalSplit(date(2005, 1, 1), date(2005, 6, 1), date(2005, 7, 1))
dates = st.dates(min_value=date(2004, 1, 1), max_value=date(2006, 12, 31))


def ratings_on(days: list[date]) -> RatingArrays:
    return from_ratings([Rating(1, index + 1, 3, day) for index, day in enumerate(days)])


@given(st.lists(dates, min_size=1))
def test_no_training_row_reaches_the_cutoff(days: list[date]) -> None:
    train, _ = split_by_time(ratings_on(days), SPLIT)

    assert all(day < SPLIT.cutoff for day in train.days.tolist())


@given(st.lists(dates, min_size=1))
def test_no_test_row_comes_before_the_cutoff(days: list[date]) -> None:
    _, test = split_by_time(ratings_on(days), SPLIT)

    assert all(SPLIT.cutoff <= day < SPLIT.test_end for day in test.days.tolist())


@given(st.lists(dates, min_size=1))
def test_every_row_inside_the_range_lands_in_exactly_one_set(days: list[date]) -> None:
    train, test = split_by_time(ratings_on(days), SPLIT)

    expected = sum(SPLIT.train_start <= day < SPLIT.test_end for day in days)
    assert len(train) + len(test) == expected


def test_boundary_days_fall_on_the_correct_side() -> None:
    last_train_day = SPLIT.cutoff - timedelta(days=1)

    train, test = split_by_time(ratings_on([last_train_day, SPLIT.cutoff]), SPLIT)

    assert train.days.tolist() == [last_train_day]
    assert test.days.tolist() == [SPLIT.cutoff]


def test_dates_out_of_order_raise() -> None:
    with pytest.raises(ValueError, match="train_start"):
        TemporalSplit(date(2005, 6, 1), date(2005, 1, 1), date(2005, 7, 1))


def test_random_split_keeps_every_row_once() -> None:
    ratings = ratings_on([date(2005, 1, 1)] * 100)

    train, test = split_at_random(ratings, test_size=20, seed=0)

    assert len(test) == 20
    assert sorted(np.concatenate([train.user_ids, test.user_ids]).tolist()) == list(range(1, 101))


def test_random_split_repeats_with_the_same_seed() -> None:
    ratings = ratings_on([date(2005, 1, 1)] * 100)

    first, _ = split_at_random(ratings, test_size=20, seed=3)
    second, _ = split_at_random(ratings, test_size=20, seed=3)

    assert first.user_ids.tolist() == second.user_ids.tolist()


@pytest.mark.parametrize("test_size", [0, 3])
def test_random_split_rejects_an_empty_set(test_size: int) -> None:
    with pytest.raises(ValueError, match="test_size"):
        split_at_random(ratings_on([date(2005, 1, 1)] * 3), test_size=test_size, seed=0)


@pytest.mark.parametrize(
    ("day", "months", "expected"),
    [
        (date(2005, 10, 1), -12, date(2004, 10, 1)),
        (date(2005, 2, 1), -3, date(2004, 11, 1)),
        (date(2005, 12, 1), 1, date(2006, 1, 1)),
    ],
)
def test_add_months(day: date, months: int, expected: date) -> None:
    assert add_months(day, months) == expected


def test_add_months_needs_the_first_of_a_month() -> None:
    with pytest.raises(ValueError, match="first day"):
        add_months(date(2005, 1, 2), 1)
