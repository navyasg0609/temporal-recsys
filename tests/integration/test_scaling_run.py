import json
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import ingest
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file
from temporal_recsys.experiments.scaling import MEASUREMENTS_FILE, ScalingPlan, run_scaling
from temporal_recsys.models.als import AlsConfig, AlsVariant
from temporal_recsys.models.spark_als import SparkAlsConfig

pytestmark = pytest.mark.spark

N_FILMS = 50
PLAN = ScalingPlan(
    id="test",
    cutoff=date(2005, 10, 1),
    window_months=6,
    core_counts=(1, 2),
    repetitions=1,
    warmup_passes=1,
    order_seed=1,
    n_films=N_FILMS,
    driver_memory="1g",
    spark_model=SparkAlsConfig(rank=2, iterations=1, seed=1),
    numpy_model=AlsConfig(AlsVariant.PREFERENCE, rank=2, iterations=1),
)


@pytest.fixture(scope="module")
def data_dir(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("scaling")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    ratings = generate_ratings(SyntheticConfig(users=200, films=N_FILMS, ratings=2_000)).ratings
    write_raw_file(ratings, raw_dir / "combined_data_1.txt")
    ingest(spark, raw_dir, root / "output")
    return root / "output"


def test_each_arm_runs_on_its_cores_and_a_restart_keeps_them(
    data_dir: Path, tmp_path: Path
) -> None:
    summary = run_scaling(data_dir, PLAN, tmp_path, code_commit="abc")
    finished = (tmp_path / MEASUREMENTS_FILE).read_text(encoding="utf-8")
    run_scaling(data_dir, PLAN, tmp_path, code_commit="abc")

    assert summary["code_commit"] == "abc"
    assert summary["rows"] > 0
    assert len(summary["results"]) == 3
    records = [json.loads(line) for line in finished.splitlines()]
    assert len(records) == 6
    assert {record["master"] for record in records} == {"local[1]", "local[2]", "numpy, 1 thread"}
    assert (tmp_path / MEASUREMENTS_FILE).read_text(encoding="utf-8") == finished
