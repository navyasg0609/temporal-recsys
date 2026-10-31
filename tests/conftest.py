from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.spark import create_spark_session


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("tests")
    yield session
    session.stop()
