"""Hold ratings as columns of numbers for single-machine models and evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from numpy.typing import NDArray

from temporal_recsys.data.parser import Rating
from temporal_recsys.data.partition import MONTH_COLUMN


@dataclass(frozen=True)
class RatingArrays:
    film_ids: NDArray[np.int32]
    user_ids: NDArray[np.int32]
    stars: NDArray[np.int8]
    days: NDArray[np.datetime64]

    def __len__(self) -> int:
        return len(self.film_ids)

    def take(self, selection: NDArray[np.bool_] | NDArray[np.int64]) -> "RatingArrays":
        return RatingArrays(
            self.film_ids[selection],
            self.user_ids[selection],
            self.stars[selection],
            self.days[selection],
        )


def from_ratings(ratings: Sequence[Rating]) -> RatingArrays:
    return RatingArrays(
        np.array([rating.film_id for rating in ratings], dtype=np.int32),
        np.array([rating.user_id for rating in ratings], dtype=np.int32),
        np.array([rating.stars for rating in ratings], dtype=np.int8),
        np.array([rating.rated_on for rating in ratings], dtype="datetime64[D]"),
    )


def load_ratings(monthly_dir: Path, first_day: date, end_day: date) -> RatingArrays:
    """Load ratings from `first_day` up to, but not including, `end_day`.

    The month filter skips folders outside the range, and the day filter trims any
    window that does not start on the first of a month.
    """
    partitioning = ds.partitioning(pa.schema([(MONTH_COLUMN, pa.string())]), flavor="hive")
    dataset = ds.dataset(str(monthly_dir), format="parquet", partitioning=partitioning)
    months = (ds.field(MONTH_COLUMN) >= first_day.strftime("%Y-%m")) & (
        ds.field(MONTH_COLUMN) <= end_day.strftime("%Y-%m")
    )
    days = (ds.field("rated_on") >= first_day) & (ds.field("rated_on") < end_day)
    table = dataset.to_table(
        columns=["film_id", "user_id", "stars", "rated_on"], filter=months & days
    )
    return RatingArrays(
        table["film_id"].to_numpy(),
        table["user_id"].to_numpy(),
        table["stars"].to_numpy(),
        table["rated_on"].to_numpy().astype("datetime64[D]"),
    )
