from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.benchmark import (
    DATE_SORTED_DIR,
    BenchmarkPlan,
    run_benchmark,
    select_window,
    window_bounds,
)
from temporal_recsys.data.pipeline import ingest
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file

pytestmark = pytest.mark.spark

CONFIG = SyntheticConfig(users=200, films=50, ratings=2_000)
PLAN = BenchmarkPlan(
    id="test", last_month="2005-12", window_months=(3,), repetitions=1, warmup_passes=1, seed=1
)


@pytest.fixture(scope="module")
def data_dir(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("benchmark")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    write_raw_file(generate_ratings(CONFIG).ratings, raw_dir / "combined_data_1.txt")
    ingest(spark, raw_dir, root / "output")
    return root / "output"


def test_every_layout_matches_the_same_rows(spark: SparkSession, data_dir: Path) -> None:
    result = run_benchmark(spark, data_dir, PLAN, code_commit="abc")

    rows = {window["rows_matched"] for window in result["windows"]}
    assert len(rows) == 1
    assert rows.pop() > 0


def test_results_record_warmup_and_every_arm(spark: SparkSession, data_dir: Path) -> None:
    result = run_benchmark(spark, data_dir, PLAN, code_commit="abc")

    assert result["code_commit"] == "abc"
    assert len(result["measurements"]) == 3 * 2 * 2
    assert len(result["summary"]) == 3 * 2


def test_monthly_selection_excludes_rows_outside_the_window(
    spark: SparkSession, data_dir: Path
) -> None:
    run_benchmark(spark, data_dir, PLAN, code_commit="abc")
    first_day, end_day = window_bounds("2005-12", 3)

    dates = [
        row["rated_on"]
        for row in select_window(spark, data_dir, "monthly", first_day, end_day).collect()
    ]

    assert min(dates) >= first_day
    assert max(dates) < end_day


def test_date_sorted_layout_is_built_once(spark: SparkSession, data_dir: Path) -> None:
    run_benchmark(spark, data_dir, PLAN, code_commit="abc")
    marker = data_dir / DATE_SORTED_DIR / "_SUCCESS"
    finished_at = marker.stat().st_mtime_ns

    run_benchmark(spark, data_dir, PLAN, code_commit="abc")

    assert marker.stat().st_mtime_ns == finished_at
