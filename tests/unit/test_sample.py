from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from temporal_recsys.cli import main
from temporal_recsys.experiments.sample import load_config, planted_answer_found

CONFIG = Path("configs/sample.yaml")


def als(cutoff: str, window: int | str, ndcg: float) -> dict[str, Any]:
    return {"model": "numpy-als", "cutoff": cutoff, "window": window, "ndcg": ndcg}


def test_planted_answer_needs_the_short_window_ahead_at_every_cutoff() -> None:
    ahead_everywhere = [
        als("2005-08-01", 6, 0.12),
        als("2005-08-01", "all", 0.09),
        als("2005-09-01", 6, 0.11),
        als("2005-09-01", "all", 0.10),
    ]
    behind_once = [*ahead_everywhere[:2], als("2005-09-01", 6, 0.10), ahead_everywhere[3]]

    assert planted_answer_found(ahead_everywhere, after_change_window=6)
    assert not planted_answer_found(behind_once, after_change_window=6)


def test_planted_answer_ignores_other_models() -> None:
    records = [
        als("2005-08-01", 6, 0.12),
        als("2005-08-01", "all", 0.09),
        {"model": "popularity", "cutoff": "2005-08-01", "window": "all", "ndcg": 0.5},
    ]

    assert planted_answer_found(records, after_change_window=6)


def test_planted_answer_without_als_results_raises() -> None:
    with pytest.raises(ValueError, match="No ALS results"):
        planted_answer_found([], after_change_window=6)


def test_a_window_reaching_before_the_change_is_refused() -> None:
    config = load_config(CONFIG)

    # The change is in January 2005, and 8 months before August 2005 is December 2004.
    with pytest.raises(ValueError, match="before the taste change"):
        replace(config, after_change_window=8)


def test_reproduce_without_sample_points_to_the_full_data_instructions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["reproduce"]) == 2

    assert "docs/REPRODUCING.md" in capsys.readouterr().err
