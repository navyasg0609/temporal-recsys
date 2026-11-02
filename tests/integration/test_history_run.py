from datetime import date
from pathlib import Path

import pytest
from monthly_data import write_monthly
from pyspark.sql import SparkSession

from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.baseline import ARMS_FILE, load_finished
from temporal_recsys.experiments.history import (
    NO_SEED,
    POPULARITY,
    SPARK_ALS,
    TWO_TOWER,
    HistoryPlan,
    popularity_trainer,
    run_history,
    spark_als_trainer,
    two_tower_trainer,
)

N_FILMS = 60
CONFIG = SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)
PLAN = HistoryPlan(
    id="test",
    first_day=date(2004, 1, 1),
    window_months=(1, 6, None),
    test_cutoffs=(date(2005, 8, 1), date(2005, 9, 1)),
    test_months=1,
    seeds=(1,),
    n_films=N_FILMS,
    als_rank=3,
    als_regularization=0.1,
    als_confidence=3.0,
    als_iterations=2,
    tower_dimensions=4,
    tower_max_history=5,
    tower_batch_size=32,
    tower_epochs=1,
    tower_learning_rate=0.01,
    tower_temperature=0.1,
    evaluation=EvaluationConfig(),
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("history")
    write_monthly(generate_ratings(CONFIG).ratings, root / MONTHLY_DIR)
    titles = "".join(f"{film},2000,Film {film % 5}\n" for film in range(1, N_FILMS + 1))
    (root / "movie_titles.csv").write_text(titles, encoding="latin-1")
    return root


def run_popularity(data_dir: Path, output_dir: Path) -> dict[str, object]:
    return run_history(
        data_dir / MONTHLY_DIR,
        PLAN,
        POPULARITY,
        popularity_trainer(PLAN),
        (NO_SEED,),
        output_dir,
        "abc",
        {},
    )


def test_all_windows_at_a_cutoff_share_the_same_users(data_dir: Path, tmp_path: Path) -> None:
    run_popularity(data_dir, tmp_path)

    records = load_finished(tmp_path / ARMS_FILE).values()
    assert len(records) == 6
    for cutoff in ("2005-08-01", "2005-09-01"):
        eligible = {r["users_eligible"] for r in records if r["cutoff"] == cutoff}
        assert len(eligible) == 1


def test_shorter_windows_train_on_fewer_rows(data_dir: Path, tmp_path: Path) -> None:
    run_popularity(data_dir, tmp_path)

    rows = {
        r["window"]: r["train_rows"]
        for r in load_finished(tmp_path / ARMS_FILE).values()
        if r["cutoff"] == "2005-08-01"
    }
    assert rows[1] < rows[6] < rows["all"]


def test_restart_keeps_finished_arms(data_dir: Path, tmp_path: Path) -> None:
    run_popularity(data_dir, tmp_path)
    finished = (tmp_path / ARMS_FILE).read_text(encoding="utf-8")

    run_popularity(data_dir, tmp_path)

    assert (tmp_path / ARMS_FILE).read_text(encoding="utf-8") == finished


def test_two_tower_runs_on_every_window(data_dir: Path, tmp_path: Path) -> None:
    trainer = two_tower_trainer(data_dir, PLAN, tmp_path, device="cpu")

    summary = run_history(
        data_dir / MONTHLY_DIR, PLAN, TWO_TOWER, trainer, PLAN.seeds, tmp_path, "abc", {}
    )

    assert sorted(row["window"] for row in summary["results"]) == ["1", "6", "all"]
    assert {row["arms"] for row in summary["results"]} == {2}


@pytest.mark.spark
def test_spark_als_records_each_part_of_the_job(
    spark: SparkSession, data_dir: Path, tmp_path: Path
) -> None:
    trainer = spark_als_trainer(spark, data_dir, PLAN)

    run_history(data_dir / MONTHLY_DIR, PLAN, SPARK_ALS, trainer, PLAN.seeds, tmp_path, "abc", {})

    record = next(iter(load_finished(tmp_path / ARMS_FILE).values()))
    total = record["read_seconds"] + record["model_fit_seconds"] + record["collect_seconds"]
    assert record["fit_seconds"] == pytest.approx(total)
