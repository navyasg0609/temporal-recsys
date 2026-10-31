from datetime import date
from pathlib import Path

import pytest
from monthly_data import write_monthly
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.baseline import ARMS_FILE
from temporal_recsys.experiments.spark_als import (
    SparkAlsPlan,
    load_plan,
    run_spark_als,
    train_and_time,
)
from temporal_recsys.models.spark_als import SparkAlsConfig

pytestmark = pytest.mark.spark

N_FILMS = 60
CONFIG = SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)
PLAN = SparkAlsPlan(
    id="test",
    window_months=6,
    tuning_cutoff=date(2005, 7, 1),
    test_cutoffs=(date(2005, 8, 1),),
    test_months=1,
    seeds=(1, 2),
    n_films=N_FILMS,
    rank=3,
    iterations=2,
    regularization=(0.01, 0.1),
    confidence=(10.0,),
    evaluation=EvaluationConfig(max_users=50),
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("spark_run")
    write_monthly(generate_ratings(CONFIG).ratings, root / MONTHLY_DIR)
    return root


def test_run_records_every_arm_and_the_machine(
    spark: SparkSession, data_dir: Path, tmp_path: Path
) -> None:
    summary = run_spark_als(spark, data_dir, PLAN, tmp_path, code_commit="abc")

    assert summary["code_commit"] == "abc"
    assert summary["machine"]["master"].startswith("local")
    assert [(row["stage"], row["model"], row["arms"]) for row in summary["results"]] == [
        ("test", "spark", 2)
    ]
    assert len((tmp_path / ARMS_FILE).read_text(encoding="utf-8").splitlines()) == 4


def test_restart_keeps_finished_arms(spark: SparkSession, data_dir: Path, tmp_path: Path) -> None:
    run_spark_als(spark, data_dir, PLAN, tmp_path, code_commit="abc")
    finished = (tmp_path / ARMS_FILE).read_text(encoding="utf-8")

    run_spark_als(spark, data_dir, PLAN, tmp_path, code_commit="abc")

    assert (tmp_path / ARMS_FILE).read_text(encoding="utf-8") == finished


def test_each_part_of_the_job_is_timed(spark: SparkSession, data_dir: Path) -> None:
    config = SparkAlsConfig(rank=3, iterations=1, seed=1)

    timed = train_and_time(
        spark, data_dir, MONTHLY_DIR, PLAN.split_at(PLAN.tuning_cutoff), config, N_FILMS
    )

    assert timed.rows > 0
    assert min(timed.read_seconds, timed.fit_seconds, timed.collect_seconds) > 0


def test_pilot_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/spark-pilot.yaml"))

    assert plan.window_months == 12
    assert plan.test_cutoffs == ()
    assert len(plan.regularization) * len(plan.confidence) == 1
