from datetime import date

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.partition import with_rating_month
from temporal_recsys.data.quality import QualityReport, build_report

pytestmark = pytest.mark.spark


def test_report_summarizes_ratings(spark: SparkSession) -> None:
    ratings = with_rating_month(
        spark.createDataFrame(
            [
                (1, 10, 3, date(2005, 9, 6)),
                (1, 11, 5, date(2005, 9, 30)),
                (2, 10, 1, date(2004, 1, 31)),
                (2, 10, 1, date(2004, 1, 31)),
            ],
            ["film_id", "user_id", "stars", "rated_on"],
        )
    )

    assert build_report(ratings) == QualityReport(
        rows=4,
        films=2,
        users=2,
        duplicate_ratings=1,
        first_date="2004-01-31",
        last_date="2005-09-30",
        rows_per_stars={1: 2, 3: 1, 5: 1},
        rows_per_month={"2004-01": 2, "2005-09": 2},
    )


def test_empty_data_raises(spark: SparkSession) -> None:
    empty = spark.createDataFrame(
        [], "film_id int, user_id int, stars int, rated_on date, rating_month string"
    )

    with pytest.raises(ValueError, match="empty"):
        build_report(empty)
