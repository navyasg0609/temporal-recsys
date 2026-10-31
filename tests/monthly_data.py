"""Write ratings in the monthly folder layout without needing Spark."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from temporal_recsys.data.parser import Rating


def write_monthly(ratings: list[Rating], root: Path) -> None:
    by_month: dict[str, list[Rating]] = {}
    for rating in ratings:
        by_month.setdefault(rating.rated_on.strftime("%Y-%m"), []).append(rating)

    for month, rows in by_month.items():
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
