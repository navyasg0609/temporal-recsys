from pathlib import Path

import pytest
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession

from temporal_recsys.data.convert import convert_raw_files
from temporal_recsys.data.partition import MONTH_COLUMN, partition_by_month

pytestmark = pytest.mark.spark

RAW_TEXT = "1:\n10,3,2005-09-06\n11,5,2005-09-30\n2:\n10,1,2004-01-31\n11,2,2005-10-01\n"


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    raw_path = tmp_path / "combined_data_1.txt"
    raw_path.write_text(RAW_TEXT, encoding="ascii")
    output_dir = tmp_path / "parquet"
    convert_raw_files([raw_path], output_dir)
    return output_dir


def test_each_month_gets_its_own_folder(
    spark: SparkSession, parquet_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "monthly"

    partition_by_month(spark, parquet_dir, target)

    assert sorted(path.name for path in target.glob(f"{MONTH_COLUMN}=*")) == [
        "rating_month=2004-01",
        "rating_month=2005-09",
        "rating_month=2005-10",
    ]


def test_no_rows_are_lost_and_each_is_in_the_right_month(
    spark: SparkSession, parquet_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "monthly"

    partition_by_month(spark, parquet_dir, target)

    rows = spark.read.parquet(str(target)).collect()
    assert len(rows) == 4
    assert all(row["rated_on"].strftime("%Y-%m") == row[MONTH_COLUMN] for row in rows)


def test_existing_output_is_not_overwritten(
    spark: SparkSession, parquet_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "monthly"
    partition_by_month(spark, parquet_dir, target)

    with pytest.raises(AnalysisException):
        partition_by_month(spark, parquet_dir, target)
