"""Divide ratings into training and test sets."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from temporal_recsys.data.arrays import RatingArrays


@dataclass(frozen=True)
class TemporalSplit:
    """Train on `train_start` up to `cutoff`, then test on `cutoff` up to `test_end`."""

    train_start: date
    cutoff: date
    test_end: date

    def __post_init__(self) -> None:
        if not self.train_start < self.cutoff < self.test_end:
            raise ValueError("Expected train_start < cutoff < test_end")


def add_months(day: date, months: int) -> date:
    """Move a first-of-month date forward, or backward with negative `months`."""
    if day.day != 1:
        raise ValueError("Expected the first day of a month")
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def window_before(cutoff: date, window_months: int, test_months: int) -> TemporalSplit:
    return TemporalSplit(
        add_months(cutoff, -window_months), cutoff, add_months(cutoff, test_months)
    )


def split_by_time(ratings: RatingArrays, split: TemporalSplit) -> tuple[RatingArrays, RatingArrays]:
    days = ratings.days
    train = (days >= np.datetime64(split.train_start)) & (days < np.datetime64(split.cutoff))
    test = (days >= np.datetime64(split.cutoff)) & (days < np.datetime64(split.test_end))
    return ratings.take(train), ratings.take(test)


def split_at_random(
    ratings: RatingArrays, test_size: int, seed: int
) -> tuple[RatingArrays, RatingArrays]:
    """Hold out random ratings, ignoring dates, to measure how much leakage inflates scores.

    This is the flawed method that temporal evaluation replaces, kept only for comparison.
    """
    if not 0 < test_size < len(ratings):
        raise ValueError("test_size must leave both sets non-empty")
    is_test = np.zeros(len(ratings), dtype=np.bool_)
    is_test[np.random.default_rng(seed).choice(len(ratings), size=test_size, replace=False)] = True
    return ratings.take(~is_test), ratings.take(is_test)
