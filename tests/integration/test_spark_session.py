from pathlib import Path

import pytest
from pyspark.sql import SparkSession

pytestmark = pytest.mark.spark


def test_parquet_round_trip(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "rows")

    spark.createDataFrame([(1, "a"), (2, "b")], ["id", "label"]).write.parquet(path)

    assert sorted(tuple(row) for row in spark.read.parquet(path).collect()) == [
        (1, "a"),
        (2, "b"),
    ]
