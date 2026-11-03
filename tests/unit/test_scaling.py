from datetime import date
from pathlib import Path
from typing import Any

import pytest

from temporal_recsys.experiments.scaling import (
    NUMPY,
    SPARK,
    ScalingPlan,
    load_plan,
    summarize_scaling,
)
from temporal_recsys.models.als import AlsConfig, AlsVariant
from temporal_recsys.models.spark_als import SparkAlsConfig


def record(
    engine: str, cores: int, total: float, rows: int = 100, is_warmup: bool = False
) -> dict[str, Any]:
    return {
        "engine": engine,
        "cores": cores,
        "rows": rows,
        "is_warmup": is_warmup,
        "read_seconds": 2.0,
        "fit_seconds": total - 3.0,
        "collect_seconds": 1.0,
    }


def runs(engine: str, cores: int, *totals: float) -> list[dict[str, Any]]:
    return [record(engine, cores, total) for total in totals]


def by_arm(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["engine"], row["cores"]): row for row in summary["results"]}


def test_speedup_divides_one_core_median_by_each_median() -> None:
    records = [
        *runs(SPARK, 1, 98.0, 100.0, 102.0),
        *runs(SPARK, 2, 59.0, 60.0, 61.0),
        *runs(SPARK, 4, 39.0, 40.0, 41.0),
        *runs(NUMPY, 1, 49.0, 50.0, 51.0),
    ]

    results = by_arm(summarize_scaling(records))

    assert results[(SPARK, 1)]["speedup"] == 1.0
    assert results[(SPARK, 2)]["speedup"] == pytest.approx(100 / 60)
    assert results[(SPARK, 4)]["speedup"] == 2.5
    assert results[(SPARK, 4)]["total_seconds_min"] == 39.0
    assert results[(SPARK, 4)]["total_seconds_max"] == 41.0
    assert "speedup" not in results[(NUMPY, 1)]


def test_cost_is_fewest_cores_whose_slowest_run_beats_numpys_fastest() -> None:
    # Two cores have a lower median than NumPy, but the ranges overlap.
    records = [
        *runs(SPARK, 1, 98.0, 100.0, 102.0),
        *runs(SPARK, 2, 45.0, 48.0, 52.0),
        *runs(SPARK, 4, 39.0, 40.0, 41.0),
        *runs(NUMPY, 1, 49.0, 50.0, 51.0),
    ]

    summary = summarize_scaling(records)

    assert by_arm(summary)[(SPARK, 2)]["beats_numpy"] is False
    assert summary["cost"] == 4


def test_cost_is_none_when_no_core_count_beats_numpy() -> None:
    records = [*runs(SPARK, 1, 100.0), *runs(SPARK, 4, 60.0), *runs(NUMPY, 1, 50.0)]

    assert summarize_scaling(records)["cost"] is None


def test_warmup_passes_are_left_out() -> None:
    records = [
        record(SPARK, 1, 500.0, is_warmup=True),
        *runs(SPARK, 1, 100.0),
        *runs(NUMPY, 1, 50.0),
    ]

    result = by_arm(summarize_scaling(records))[(SPARK, 1)]

    assert result["passes"] == 1
    assert result["total_seconds_median"] == 100.0


def test_arms_reading_different_rows_raise() -> None:
    records = [record(SPARK, 1, 100.0, rows=100), record(NUMPY, 1, 50.0, rows=99)]

    with pytest.raises(ValueError, match="different rows"):
        summarize_scaling(records)


def test_missing_reference_arm_raises() -> None:
    with pytest.raises(ValueError, match="one core and NumPy"):
        summarize_scaling(runs(SPARK, 1, 100.0))


def test_plan_needs_spark_on_one_core() -> None:
    with pytest.raises(ValueError, match="one core"):
        ScalingPlan(
            "x",
            date(2005, 8, 1),
            12,
            (2, 4),
            1,
            0,
            0,
            10,
            "1g",
            SparkAlsConfig(),
            AlsConfig(AlsVariant.PREFERENCE),
        )


def test_registered_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/scaling.yaml"))

    assert plan.core_counts == (1, 2, 4)
    assert plan.arms[-1] == (NUMPY, 1)
    assert plan.spark_model.regularization == 0.01
    assert plan.numpy_model.regularization == 10.0
    assert plan.numpy_model.variant is AlsVariant.PREFERENCE
    assert plan.split.train_start == date(2004, 8, 1)
