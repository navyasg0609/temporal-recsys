from datetime import date
from pathlib import Path

import pytest

from temporal_recsys.data.benchmark import (
    BenchmarkPlan,
    Measurement,
    eligible_bytes,
    load_plan,
    summarize,
    window_bounds,
)


@pytest.mark.parametrize(
    ("last_month", "months", "first_day", "end_day"),
    [
        ("2005-12", 3, date(2005, 10, 1), date(2006, 1, 1)),
        ("2005-12", 12, date(2005, 1, 1), date(2006, 1, 1)),
        ("2005-02", 3, date(2004, 12, 1), date(2005, 3, 1)),
        ("2005-06", 1, date(2005, 6, 1), date(2005, 7, 1)),
    ],
)
def test_window_bounds(last_month: str, months: int, first_day: date, end_day: date) -> None:
    assert window_bounds(last_month, months) == (first_day, end_day)


def test_summary_excludes_warmup_and_uses_the_median() -> None:
    measurements = [
        Measurement(0, "flat", 3, "read", 100.0, is_warmup=True),
        Measurement(1, "flat", 3, "read", 1.0, is_warmup=False),
        Measurement(2, "flat", 3, "read", 5.0, is_warmup=False),
        Measurement(3, "flat", 3, "read", 2.0, is_warmup=False),
    ]

    [summary] = summarize(measurements)

    assert (summary.minimum_seconds, summary.median_seconds, summary.maximum_seconds) == (
        1.0,
        2.0,
        5.0,
    )


def test_monthly_layout_counts_only_files_inside_the_window(tmp_path: Path) -> None:
    for month, size in [("2005-09", 1), ("2005-10", 2), ("2005-12", 4), ("2006-01", 8)]:
        folder = tmp_path / "monthly" / f"rating_month={month}"
        folder.mkdir(parents=True)
        (folder / "part.parquet").write_bytes(b"x" * size)

    assert eligible_bytes(tmp_path, "monthly", date(2005, 10, 1), date(2006, 1, 1)) == 6


def test_flat_layout_counts_every_file(tmp_path: Path) -> None:
    folder = tmp_path / "flat"
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"xx")
    (folder / "b.parquet").write_bytes(b"xxx")

    assert eligible_bytes(tmp_path, "flat", date(2005, 10, 1), date(2006, 1, 1)) == 5


def test_registered_plan_loads() -> None:
    plan = load_plan(Path("experiments/registered/read-benchmark.yaml"))

    assert plan == BenchmarkPlan(
        id="read-benchmark",
        last_month="2005-12",
        window_months=(3, 12),
        repetitions=5,
        warmup_passes=1,
        seed=2026,
    )
