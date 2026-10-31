"""Check that processed ratings are complete and valid."""

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from temporal_recsys.data.parser import MAX_RATING, MIN_RATING
from temporal_recsys.data.partition import MONTH_COLUMN


@dataclass(frozen=True)
class QualityReport:
    rows: int
    films: int
    users: int
    duplicate_ratings: int
    first_date: str
    last_date: str
    rows_per_stars: dict[int, int]
    rows_per_month: dict[str, int]


@dataclass(frozen=True)
class ExpectedData:
    rows: int
    films: int
    first_date: str
    last_date: str


# From the source check recorded in docs/LAB_NOTEBOOK.md.
NETFLIX_PRIZE = ExpectedData(
    rows=100_480_507, films=17_770, first_date="1999-11-11", last_date="2005-12-31"
)


def build_report(ratings: DataFrame) -> QualityReport:
    """Summarize partitioned ratings, which must include the month column."""
    summary = ratings.agg(
        F.count("*").alias("rows"),
        F.countDistinct("film_id").alias("films"),
        F.countDistinct("user_id").alias("users"),
        F.min("rated_on").alias("first_date"),
        F.max("rated_on").alias("last_date"),
    ).first()
    if summary is None or summary["rows"] == 0:
        raise ValueError("Cannot report on an empty dataset")

    unique_ratings = ratings.select("film_id", "user_id").distinct().count()
    return QualityReport(
        rows=summary["rows"],
        films=summary["films"],
        users=summary["users"],
        duplicate_ratings=summary["rows"] - unique_ratings,
        first_date=summary["first_date"].isoformat(),
        last_date=summary["last_date"].isoformat(),
        rows_per_stars=_count_by(ratings, "stars"),
        rows_per_month=_count_by(ratings, MONTH_COLUMN),
    )


def find_problems(report: QualityReport, expected: ExpectedData | None = None) -> list[str]:
    """Return every problem found, or an empty list if the data passes."""
    problems: list[str] = []
    if report.duplicate_ratings:
        problems.append(f"{report.duplicate_ratings} duplicate film and user pairs")

    invalid_stars = sorted(set(report.rows_per_stars) - set(range(MIN_RATING, MAX_RATING + 1)))
    if invalid_stars:
        problems.append(f"invalid star values: {invalid_stars}")

    if sum(report.rows_per_month.values()) != report.rows:
        problems.append("monthly row counts do not add up to the total")

    if expected is not None:
        for field in ("rows", "films", "first_date", "last_date"):
            found, wanted = getattr(report, field), getattr(expected, field)
            if found != wanted:
                problems.append(f"{field}: found {found}, expected {wanted}")
    return problems


def _count_by[K](ratings: DataFrame, column: str) -> dict[K, int]:
    rows = ratings.groupBy(column).count().orderBy(column).collect()
    return {row[column]: row["count"] for row in rows}
