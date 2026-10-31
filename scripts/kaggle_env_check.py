"""Check that Spark can read the Netflix Prize data on a Kaggle notebook.

Attach the Netflix Prize dataset, run `pip install -q pyspark` in one cell, then paste
this file into the next. It prints only machine details, counts, dates and timings,
never ratings. The timings are a feasibility check, not published results.
"""

import os
import platform
import shutil
import sys
import time
from pathlib import Path

from pyspark import __version__ as pyspark_version
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_ROOT = Path("/kaggle/input")
DRIVER_MEMORY = "20g"
RATING_LINE = r"^\d+,[1-5],\d{4}-\d{2}-\d{2}$"
FILM_HEADER_LINE = r"^\d+:$"

# Figures from third-party descriptions of the dataset, used only as a comparison.
EXPECTED_RATINGS = 100_480_507
EXPECTED_FILMS = 17_770


def print_machine() -> None:
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    memory_kb = next(int(line.split()[1]) for line in meminfo if line.startswith("MemTotal"))
    disk = shutil.disk_usage("/kaggle/working")
    print(f"cpu_count: {os.cpu_count()}")
    print(f"memory_gb: {memory_kb / 1024**2:.1f}")
    print(f"disk_free_gb: {disk.free / 1024**3:.1f}")
    print(f"python: {platform.python_version()}")
    print(f"pyspark: {pyspark_version}")
    print(f"java_on_path: {shutil.which('java')}")


def find_rating_files() -> list[str]:
    files = sorted(str(path) for path in INPUT_ROOT.rglob("combined_data_*.txt"))
    if not files:
        sys.exit(f"No combined_data_*.txt files under {INPUT_ROOT}. Attach the dataset first.")
    return files


def main() -> None:
    print_machine()
    files = find_rating_files()
    print(f"rating_files: {len(files)}")
    print(f"rating_files_gb: {sum(Path(f).stat().st_size for f in files) / 1024**3:.2f}")

    start = time.perf_counter()
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("kaggle-env-check")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .getOrCreate()
    )
    print(f"spark_start_seconds: {time.perf_counter() - start:.1f}")

    lines = spark.read.text(files)
    is_rating = F.col("value").rlike(RATING_LINE)
    is_header = F.col("value").rlike(FILM_HEADER_LINE)
    rating_date = F.when(is_rating, F.substring_index(F.col("value"), ",", -1))

    start = time.perf_counter()
    summary = lines.agg(
        F.count("*").alias("lines"),
        F.sum(is_rating.cast("long")).alias("ratings"),
        F.sum(is_header.cast("long")).alias("film_headers"),
        F.min(rating_date).alias("first_date"),
        F.max(rating_date).alias("last_date"),
    ).first()
    print(f"full_scan_seconds: {time.perf_counter() - start:.1f}")
    spark.stop()

    other = summary["lines"] - summary["ratings"] - summary["film_headers"]
    print(f"lines: {summary['lines']}")
    print(f"ratings: {summary['ratings']} (expected about {EXPECTED_RATINGS})")
    print(f"film_headers: {summary['film_headers']} (expected about {EXPECTED_FILMS})")
    print(f"unrecognized_lines: {other}")
    print(f"date_range: {summary['first_date']} to {summary['last_date']}")


if __name__ == "__main__":
    main()
