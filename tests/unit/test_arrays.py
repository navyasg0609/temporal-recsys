from datetime import date
from pathlib import Path

import numpy as np
import pytest

from temporal_recsys.data.arrays import from_ratings, load_ratings, write_monthly
from temporal_recsys.data.parser import Rating

RATINGS = [
    Rating(1, 10, 3, date(2005, 8, 31)),
    Rating(2, 10, 4, date(2005, 9, 1)),
    Rating(3, 10, 5, date(2005, 9, 30)),
    Rating(4, 10, 1, date(2005, 10, 1)),
]


def test_from_ratings_keeps_every_column() -> None:
    arrays = from_ratings([Rating(1, 10, 3, date(2005, 9, 6))])

    assert arrays.film_ids.tolist() == [1]
    assert arrays.user_ids.tolist() == [10]
    assert arrays.stars.tolist() == [3]
    assert arrays.days.tolist() == [date(2005, 9, 6)]


def test_write_monthly_makes_one_folder_per_month(tmp_path: Path) -> None:
    write_monthly(RATINGS, tmp_path)

    folders = sorted(path.name for path in tmp_path.iterdir())
    assert folders == ["rating_month=2005-08", "rating_month=2005-09", "rating_month=2005-10"]


def test_write_monthly_refuses_to_overwrite(tmp_path: Path) -> None:
    write_monthly(RATINGS, tmp_path)

    with pytest.raises(FileExistsError):
        write_monthly(RATINGS, tmp_path)


def test_load_ratings_returns_only_the_requested_days(tmp_path: Path) -> None:
    write_monthly(RATINGS, tmp_path)

    arrays = load_ratings(tmp_path, date(2005, 9, 1), date(2005, 10, 1))

    assert sorted(arrays.film_ids.tolist()) == [2, 3]
    assert arrays.stars[np.argsort(arrays.film_ids)].tolist() == [4, 5]
    assert arrays.days.dtype == np.dtype("datetime64[D]")
