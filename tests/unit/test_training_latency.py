from datetime import date
from pathlib import Path
from typing import Any

import pytest

from temporal_recsys.experiments.training_latency import (
    LatencyPlan,
    load_plan,
    summarize_latency,
)
from temporal_recsys.models.spark_als import SparkAlsConfig


def record(
    layout: str, total: float, rows: int = 100, is_warmup: bool = False, months: int = 3
) -> dict[str, Any]:
    return {
        "layout": layout,
        "window_months": months,
        "rows": rows,
        "is_warmup": is_warmup,
        "read_seconds": 10.0,
        "fit_seconds": total - 11.0,
        "collect_seconds": 1.0,
    }


def test_saving_compares_median_whole_job_times() -> None:
    records = [
        record("flat", 100.0),
        record("flat", 110.0),
        record("flat", 200.0),
        record("monthly", 90.0),
        record("monthly", 99.0),
        record("monthly", 300.0),
    ]

    by_layout = {row["layout"]: row for row in summarize_latency(records)}

    assert by_layout["flat"]["total_seconds_median"] == 110.0
    assert by_layout["monthly"]["total_seconds_median"] == 99.0
    assert by_layout["flat"]["saving_against_flat"] == 0.0
    assert by_layout["monthly"]["saving_against_flat"] == pytest.approx(0.1)
    assert by_layout["monthly"]["total_seconds_max"] == 300.0


def test_warmup_passes_are_left_out() -> None:
    records = [
        record("flat", 500.0, is_warmup=True),
        record("flat", 100.0),
        record("monthly", 50.0),
    ]

    by_layout = {row["layout"]: row for row in summarize_latency(records)}

    assert by_layout["flat"]["passes"] == 1
    assert by_layout["flat"]["total_seconds_median"] == 100.0


def test_layouts_reading_different_rows_raise() -> None:
    records = [record("flat", 100.0, rows=100), record("monthly", 90.0, rows=99)]

    with pytest.raises(ValueError, match="different rows"):
        summarize_latency(records)


def test_plan_needs_the_flat_layout() -> None:
    with pytest.raises(ValueError, match="flat"):
        LatencyPlan("x", date(2005, 10, 1), (3,), ("monthly",), 1, 1, 0, 0, 10, SparkAlsConfig())


def test_registered_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/training-latency.yaml"))

    assert plan.window_months == (3, 12)
    assert plan.layouts == ("flat", "monthly")
    assert plan.model.confidence == 10.0
