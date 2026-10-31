"""Turn raw rating files into validated monthly Parquet data."""

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from pyspark.sql import SparkSession

from temporal_recsys.data.convert import convert_raw_files
from temporal_recsys.data.partition import partition_by_month
from temporal_recsys.data.quality import ExpectedData, build_report, find_problems

LOGGER = logging.getLogger(__name__)

RAW_FILE_PATTERN = "combined_data_*.txt"
FLAT_DIR = "flat"
MONTHLY_DIR = "monthly"
REPORT_FILE = "quality_report.json"
SPARK_SUCCESS_MARKER = "_SUCCESS"


def find_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob(RAW_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No {RAW_FILE_PATTERN} files under {raw_dir}")
    return files


def ingest(
    spark: SparkSession, raw_dir: Path, output_dir: Path, expected: ExpectedData | None = None
) -> list[str]:
    """Build the flat and monthly datasets, and return any quality problems found.

    Finished stages are kept, so a run that stops partway can simply be restarted.
    """
    flat_dir = output_dir / FLAT_DIR
    monthly_dir = output_dir / MONTHLY_DIR

    convert_raw_files(find_raw_files(raw_dir), flat_dir)

    if not (monthly_dir / SPARK_SUCCESS_MARKER).exists():
        if monthly_dir.exists():
            LOGGER.info("Removing unfinished output %s", monthly_dir)
            shutil.rmtree(monthly_dir)
        partition_by_month(spark, flat_dir, monthly_dir)

    report = build_report(spark.read.parquet(str(monthly_dir)))
    (output_dir / REPORT_FILE).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return find_problems(report, expected)
