from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from monthly_data import write_monthly

from temporal_recsys.data.pipeline import MONTHLY_DIR
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings
from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.experiments.history import HistoryPlan
from temporal_recsys.experiments.serving_export import (
    BUNDLE_FILE,
    ExportConfig,
    export_serving,
    load_config,
)
from temporal_recsys.serving.bundle import load_bundle

N_FILMS = 60
PLAN = HistoryPlan(
    id="test",
    first_day=date(2004, 1, 1),
    window_months=(6,),
    test_cutoffs=(date(2005, 9, 1),),
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
CONFIG = ExportConfig(
    history_plan=Path("unused.yaml"),
    cutoff=date(2005, 9, 1),
    window_months=6,
    seed=1,
    check_users=200,
    check_seed=0,
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("serving")
    ratings = generate_ratings(SyntheticConfig(users=150, films=N_FILMS, ratings=6_000)).ratings
    write_monthly(ratings, root / MONTHLY_DIR)
    titles = "".join(f"{film},2000,Film {film % 5}\n" for film in range(1, N_FILMS + 1))
    (root / "movie_titles.csv").write_text(titles, encoding="latin-1")
    return root


def test_export_writes_a_bundle_whose_lists_match_the_evaluation(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = export_serving(
        data_dir / MONTHLY_DIR, data_dir, PLAN, CONFIG, tmp_path, "abc", "cpu"
    )

    bundle = load_bundle(tmp_path / BUNDLE_FILE)
    assert manifest["code_commit"] == "abc"
    assert manifest["users_with_model_vectors"] == len(bundle.user_ids)
    assert 0 < manifest["users_with_model_vectors"] <= manifest["users_with_history"]
    assert manifest["lists_checked"] > 0
    assert manifest["lists_matching_evaluation"] == manifest["lists_checked"]


def test_only_a_scored_arm_can_be_served(data_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scored"):
        export_serving(
            data_dir / MONTHLY_DIR,
            data_dir,
            PLAN,
            replace(CONFIG, window_months=3),
            tmp_path,
            "abc",
            "cpu",
        )


def test_serving_config_names_a_scored_two_tower_arm() -> None:
    config = load_config(Path("configs/serving.yaml"))

    assert config.cutoff == date(2005, 10, 1)
    assert config.window_months == 6
    assert config.seed == 1
