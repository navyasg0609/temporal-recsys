"""Organize ratings into one folder per month."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

MONTH_COLUMN = "rating_month"


def with_rating_month(ratings: DataFrame) -> DataFrame:
    return ratings.withColumn(MONTH_COLUMN, F.date_format("rated_on", "yyyy-MM"))


def partition_by_month(spark: SparkSession, source: Path, target: Path) -> None:
    """Rewrite Parquet ratings under `target`, partitioned by rating month.

    Existing output raises rather than being overwritten, so finished data is never
    replaced by accident.
    """
    ratings = with_rating_month(spark.read.parquet(str(source)))
    ratings.repartition(MONTH_COLUMN).write.partitionBy(MONTH_COLUMN).parquet(str(target))
