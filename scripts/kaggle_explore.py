"""Explain the fall in monthly ratings at the end of 2005.

Tests whether the fall comes from ratings the competition organizers held back. It
counts the qualifying and probe sets by month next to the ratings. Prints only monthly
totals, never individual ratings. Arguments: monthly data folder, then raw data folder.
"""

import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DEFAULT_MONTHLY_DIR = Path("/kaggle/working/ratings/monthly")
DEFAULT_RAW_DIR = Path("/kaggle/input")
DRIVER_MEMORY = "20g"


def find_file(raw_dir: Path, name: str) -> Path | None:
    return next(raw_dir.rglob(name), None)


def count_qualifying_by_month(path: Path) -> Counter[str]:
    months: Counter[str] = Counter()
    with path.open(encoding="ascii") as lines:
        for line in lines:
            if not line.rstrip().endswith(":"):
                months[line.strip().split(",")[1][:7]] += 1
    return months


def read_probe_pairs(path: Path) -> Iterator[tuple[int, int]]:
    film_id = 0
    with path.open(encoding="ascii") as lines:
        for raw_line in lines:
            line = raw_line.strip()
            if line.endswith(":"):
                film_id = int(line[:-1])
            elif line:
                yield film_id, int(line)


def main() -> None:
    monthly_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MONTHLY_DIR
    raw_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RAW_DIR
    if not monthly_dir.exists():
        sys.exit(f"No monthly data at {monthly_dir}. Run trs ingest first.")

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("explore")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .getOrCreate()
    )
    ratings = spark.read.parquet(str(monthly_dir))

    table = {
        row["rating_month"]: {
            "ratings": row["ratings"],
            "active_users": row["active_users"],
            "active_films": row["active_films"],
        }
        for row in ratings.groupBy("rating_month")
        .agg(
            F.count("*").alias("ratings"),
            F.countDistinct("user_id").alias("active_users"),
            F.countDistinct("film_id").alias("active_films"),
        )
        .collect()
    }

    first_months = ratings.groupBy("user_id").agg(F.min("rating_month").alias("rating_month"))
    new_users = first_months.groupBy("rating_month").count().collect()
    new_user_counts = {row["rating_month"]: row["count"] for row in new_users}

    qualifying_path = find_file(raw_dir, "qualifying.txt")
    qualifying = count_qualifying_by_month(qualifying_path) if qualifying_path else Counter()
    print(f"qualifying_file_found: {qualifying_path is not None}")

    probe_counts: dict[str, int] = {}
    probe_path = find_file(raw_dir, "probe.txt")
    print(f"probe_file_found: {probe_path is not None}")
    if probe_path:
        probe = spark.createDataFrame(
            list(read_probe_pairs(probe_path)), "film_id int, user_id int"
        )
        matched = ratings.join(F.broadcast(probe), ["film_id", "user_id"])
        probe_rows = matched.groupBy("rating_month").count().collect()
        probe_counts = {row["rating_month"]: row["count"] for row in probe_rows}
    spark.stop()

    months = sorted(table.keys() | qualifying.keys())
    rows = [
        {
            "month": month,
            **table.get(month, {}),
            "new_users": new_user_counts.get(month, 0),
            "probe": probe_counts.get(month, 0),
            "qualifying": qualifying.get(month, 0),
        }
        for month in months
    ]
    print(f"qualifying_total: {sum(qualifying.values())}")
    print(f"probe_total: {sum(probe_counts.values())}")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
