import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import MONTHLY_DIR, REPORT_FILE, ingest
from temporal_recsys.data.quality import ExpectedData
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file

pytestmark = pytest.mark.spark

CONFIG = SyntheticConfig(users=200, films=50, ratings=2_000)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    path = tmp_path / "raw"
    path.mkdir()
    write_raw_file(generate_ratings(CONFIG).ratings, path / "combined_data_1.txt")
    return path


def test_ingest_keeps_every_row_and_passes_quality_checks(
    spark: SparkSession, raw_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output"

    assert ingest(spark, raw_dir, output) == []

    report = json.loads((output / REPORT_FILE).read_text(encoding="utf-8"))
    assert report["rows"] == CONFIG.ratings
    assert report["duplicate_ratings"] == 0
    assert len(report["rows_per_month"]) == CONFIG.months


def test_ingest_reports_a_mismatch_with_expected_totals(
    spark: SparkSession, raw_dir: Path, tmp_path: Path
) -> None:
    expected = ExpectedData(rows=1, films=50, first_date="x", last_date="y")

    problems = ingest(spark, raw_dir, tmp_path / "output", expected)

    assert f"rows: found {CONFIG.ratings}, expected 1" in problems


def test_restart_replaces_an_unfinished_monthly_folder(
    spark: SparkSession, raw_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    unfinished = output / MONTHLY_DIR
    unfinished.mkdir(parents=True)
    (unfinished / "partial-file").write_text("incomplete", encoding="utf-8")

    assert ingest(spark, raw_dir, output) == []

    assert not (unfinished / "partial-file").exists()


def test_restart_keeps_finished_monthly_data(
    spark: SparkSession, raw_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    ingest(spark, raw_dir, output)
    marker = output / MONTHLY_DIR / "_SUCCESS"
    finished_at = marker.stat().st_mtime_ns

    assert ingest(spark, raw_dir, output) == []

    assert marker.stat().st_mtime_ns == finished_at
