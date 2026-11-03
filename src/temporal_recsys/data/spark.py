"""Create Spark sessions."""

from pyspark.sql import SparkSession

UTC = "UTC"


def create_spark_session(
    app_name: str, driver_memory: str = "2g", cores: int | None = None
) -> SparkSession:
    """Start a local Spark session on `cores` threads, or on every core if not given.

    The core count is fixed when the session starts, so changing it needs a new process.
    """
    if cores is not None and cores < 1:
        raise ValueError(f"Spark needs at least one core, got {cores}")
    return (
        SparkSession.builder.master(f"local[{cores or '*'}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", UTC)
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
