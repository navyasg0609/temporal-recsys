from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_recsys.data.arrays import from_ratings, load_ratings
from temporal_recsys.data.parser import Rating


def write_month(root: Path, month: str, rows: list[Rating]) -> None:
    folder = root / f"rating_month={month}"
    folder.mkdir(parents=True)
    film_ids, user_ids, stars, days = zip(*rows, strict=True)
    table = pa.table(
        {
            "film_id": pa.array(film_ids, pa.int32()),
            "user_id": pa.array(user_ids, pa.int32()),
            "stars": pa.array(stars, pa.int8()),
            "rated_on": pa.array(days, pa.date32()),
        }
    )
    pq.write_table(table, folder / "part-0.parquet")


def test_from_ratings_keeps_every_column() -> None:
    arrays = from_ratings([Rating(1, 10, 3, date(2005, 9, 6))])

    assert arrays.film_ids.tolist() == [1]
    assert arrays.user_ids.tolist() == [10]
    assert arrays.stars.tolist() == [3]
    assert arrays.days.tolist() == [date(2005, 9, 6)]


def test_load_ratings_returns_only_the_requested_days(tmp_path: Path) -> None:
    write_month(tmp_path, "2005-08", [Rating(1, 10, 3, date(2005, 8, 31))])
    write_month(
        tmp_path,
        "2005-09",
        [Rating(2, 10, 4, date(2005, 9, 1)), Rating(3, 10, 5, date(2005, 9, 30))],
    )
    write_month(tmp_path, "2005-10", [Rating(4, 10, 1, date(2005, 10, 1))])

    arrays = load_ratings(tmp_path, date(2005, 9, 1), date(2005, 10, 1))

    assert sorted(arrays.film_ids.tolist()) == [2, 3]
    assert arrays.days.dtype == np.dtype("datetime64[D]")
