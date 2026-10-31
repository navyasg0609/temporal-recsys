"""Create Spark sessions."""

from pyspark.sql import SparkSession

UTC = "UTC"


def create_spark_session(app_name: str, driver_memory: str = "2g") -> SparkSession:
    """Start a local Spark session that uses every available core."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", UTC)
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
