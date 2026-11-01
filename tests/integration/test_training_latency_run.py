from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import ingest
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file
from temporal_recsys.experiments.training_latency import (
    MEASUREMENTS_FILE,
    LatencyPlan,
    run_training_latency,
)
from temporal_recsys.models.spark_als import SparkAlsConfig

pytestmark = pytest.mark.spark

N_FILMS = 50
PLAN = LatencyPlan(
    id="test",
    cutoff=date(2005, 10, 1),
    window_months=(3, 6),
    layouts=("flat", "monthly"),
    test_months=1,
    repetitions=1,
    warmup_passes=1,
    order_seed=1,
    n_films=N_FILMS,
    model=SparkAlsConfig(rank=2, iterations=1, seed=1),
)


@pytest.fixture(scope="module")
def data_dir(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("latency")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    ratings = generate_ratings(SyntheticConfig(users=200, films=N_FILMS, ratings=2_000)).ratings
    write_raw_file(ratings, raw_dir / "combined_data_1.txt")
    ingest(spark, raw_dir, root / "output")
    return root / "output"


def test_every_layout_and_window_is_measured(
    spark: SparkSession, data_dir: Path, tmp_path: Path
) -> None:
    summary = run_training_latency(spark, data_dir, PLAN, tmp_path, code_commit="abc")

    assert summary["code_commit"] == "abc"
    assert len(summary["results"]) == 4
    lines = (tmp_path / MEASUREMENTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8


def test_restart_keeps_finished_arms(spark: SparkSession, data_dir: Path, tmp_path: Path) -> None:
    run_training_latency(spark, data_dir, PLAN, tmp_path, code_commit="abc")
    finished = (tmp_path / MEASUREMENTS_FILE).read_text(encoding="utf-8")

    run_training_latency(spark, data_dir, PLAN, tmp_path, code_commit="abc")

    assert (tmp_path / MEASUREMENTS_FILE).read_text(encoding="utf-8") == finished
