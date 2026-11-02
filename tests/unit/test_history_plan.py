from datetime import date
from pathlib import Path
from typing import Any

from temporal_recsys.evaluation.harness import EvaluationConfig
from temporal_recsys.evaluation.split import TemporalSplit
from temporal_recsys.experiments.history import (
    HistoryPlan,
    history_needed,
    load_plan,
    summarize_history,
)

PLAN = HistoryPlan(
    "x",
    date(2004, 1, 1),
    (1, None),
    (date(2005, 8, 1),),
    1,
    (1,),
    10,
    2,
    0.1,
    3.0,
    1,
    4,
    5,
    8,
    1,
    0.01,
    0.1,
    EvaluationConfig(),
)


def record(window: int | str, ndcg: float, seconds: float) -> dict[str, Any]:
    return {
        "model": "m",
        "window": window,
        "ndcg": ndcg,
        "recall": ndcg * 2,
        "train_rows": 100,
        "fit_seconds": seconds,
    }


def test_month_window_ends_at_the_cutoff() -> None:
    split = PLAN.training_split(date(2005, 8, 1), 3)

    assert split == TemporalSplit(date(2005, 5, 1), date(2005, 8, 1), date(2005, 9, 1))


def test_no_month_count_means_all_history() -> None:
    assert PLAN.training_split(date(2005, 8, 1), None).train_start == date(2004, 1, 1)


def test_window_cannot_start_before_the_data() -> None:
    assert PLAN.training_split(date(2005, 8, 1), 60).train_start == date(2004, 1, 1)


def test_summary_averages_and_keeps_the_range() -> None:
    rows = summarize_history([record(3, 0.1, 10.0), record(3, 0.3, 20.0), record("all", 0.5, 7.0)])

    by_window = {row["window"]: row for row in rows}
    assert by_window["3"]["arms"] == 2
    assert by_window["3"]["ndcg_mean"] == 0.2
    assert by_window["3"]["ndcg_min"] == 0.1
    assert by_window["3"]["ndcg_max"] == 0.3
    assert by_window["3"]["fit_seconds_mean"] == 15.0
    assert by_window["all"]["recall_mean"] == 1.0


def scored(cutoff: str, window: int | str, ndcg: float) -> dict[str, Any]:
    return {"cutoff": cutoff, "window": window, "ndcg": ndcg}


def test_history_needed_is_the_shortest_window_near_the_best() -> None:
    records = [scored("a", 1, 0.80), scored("a", 3, 0.96), scored("a", 12, 1.0)]

    assert history_needed(records) == "3"


def test_history_needed_must_hold_at_every_cutoff() -> None:
    records = [
        scored("a", 1, 1.0),
        scored("a", 3, 0.99),
        scored("b", 1, 0.5),
        scored("b", 3, 1.0),
    ]

    assert history_needed(records) == "3"


def test_history_needed_averages_seeds_first() -> None:
    records = [scored("a", 1, 0.5), scored("a", 1, 1.5), scored("a", 3, 1.0)]

    assert history_needed(records) == "1"


def test_all_history_counts_as_the_longest_window() -> None:
    records = [scored("a", "all", 1.0), scored("a", 24, 0.5), scored("b", "all", 1.0)]

    assert history_needed(records) == "all"


def test_history_needed_with_no_records_is_none() -> None:
    assert history_needed([]) is None


def test_registered_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/history-windows.yaml"))

    assert plan.window_months == (1, 3, 6, 12, 24, None)
    assert plan.als_confidence == 3.0
    assert plan.tower_temperature == 0.1
