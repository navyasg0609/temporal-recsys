from datetime import date
from pathlib import Path

import pytest
from monthly_data import write_monthly

from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.baseline import ARMS_FILE
from temporal_recsys.experiments.two_tower import (
    CHECKPOINT_DIR,
    TwoTowerPlan,
    load_plan,
    run_two_tower,
)

N_FILMS = 60
CONFIG = SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)
PLAN = TwoTowerPlan(
    id="test",
    window_months=6,
    tuning_cutoff=date(2005, 7, 1),
    test_cutoffs=(date(2005, 8, 1),),
    test_months=1,
    seeds=(1, 2),
    n_films=N_FILMS,
    dimensions=4,
    max_history=5,
    batch_size=64,
    epochs=(1,),
    learning_rate=(0.01, 0.001),
    temperature=(0.1,),
    compare_id_only=True,
    evaluation=EvaluationConfig(max_users=50),
)


@pytest.fixture(scope="module")
def data_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("two_tower")
    write_monthly(generate_ratings(CONFIG).ratings, root / "monthly")
    titles = "".join(f"{film},2000,Film {film % 5}\n" for film in range(1, N_FILMS + 1))
    (root / "movie_titles.csv").write_text(titles, encoding="latin-1")
    return root / "monthly", root


def test_run_records_every_arm(data_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    summary = run_two_tower(*data_dirs, PLAN, tmp_path, code_commit="abc", device="cpu")

    assert summary["code_commit"] == "abc"
    assert summary["machine"]["device"] == "cpu"
    assert [(row["model"], row["arms"]) for row in summary["results"]] == [
        ("two-tower", 2),
        ("two-tower-id-only", 2),
    ]
    assert len((tmp_path / ARMS_FILE).read_text(encoding="utf-8").splitlines()) == 6
    assert len(list((tmp_path / CHECKPOINT_DIR).iterdir())) == 6


def test_restart_keeps_finished_arms(data_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    run_two_tower(*data_dirs, PLAN, tmp_path, code_commit="abc", device="cpu")
    finished = (tmp_path / ARMS_FILE).read_text(encoding="utf-8")

    run_two_tower(*data_dirs, PLAN, tmp_path, code_commit="abc", device="cpu")

    assert (tmp_path / ARMS_FILE).read_text(encoding="utf-8") == finished


def test_pilot_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/two-tower-pilot.yaml"))

    assert plan.window_months == 12
    assert plan.test_cutoffs == ()
    assert plan.compare_id_only is False
