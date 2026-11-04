from typing import Any

import pytest

from temporal_recsys.experiments.load_test import step_measurement, summarize_load

TARGET_MS = 100.0
MAX_ERRORS = 0.01


def k6_metrics(requests: int, p95: float, error_rate: float) -> dict[str, Any]:
    return {
        "http_req_duration{phase:measured}": {
            "values": {"med": p95 / 2, "p(95)": p95, "p(99)": p95 * 2, "max": p95 * 3}
        },
        "http_reqs{phase:measured}": {"values": {"count": requests, "rate": 1.0}},
        "http_req_failed{phase:measured}": {"values": {"rate": error_rate}},
    }


def record(copies: int, users: int, p95: float, error_rate: float = 0.0) -> dict[str, Any]:
    return {"copies": copies, "users": users} | step_measurement(
        k6_metrics(users * 60, p95, error_rate), measured_seconds=60
    )


def test_step_measurement_uses_measured_time_for_the_rate() -> None:
    step = step_measurement(k6_metrics(requests=3000, p95=40.0, error_rate=0.002), 60)

    assert step["requests_per_second"] == 50.0
    assert step["p95_ms"] == 40.0
    assert step["p50_ms"] == 20.0
    assert step["error_rate"] == 0.002


def test_a_summary_without_measured_requests_raises() -> None:
    with pytest.raises(ValueError, match="no measured requests"):
        step_measurement({}, 60)


def test_failure_point_is_the_first_step_missing_the_target() -> None:
    records = [
        *(record(1, 100, p95) for p95 in (10.0, 12.0, 11.0)),
        # Median p95 is 90 ms: inside the target despite one slow pass.
        *(record(1, 400, p95) for p95 in (80.0, 90.0, 300.0)),
        # Median p95 is 150 ms.
        *(record(1, 600, p95) for p95 in (140.0, 150.0, 160.0)),
        # Back inside the target, but after the failure point.
        *(record(1, 800, p95) for p95 in (50.0, 50.0, 50.0)),
    ]

    summary = summarize_load(records, TARGET_MS, MAX_ERRORS)

    (result,) = summary["by_copies"]
    assert result["failure_point_users"] == 600
    assert result["highest_step_meeting_target"] == 400
    step_400 = next(step for step in summary["steps"] if step["users"] == 400)
    assert step_400["p95_ms_median"] == 90.0
    assert step_400["p95_ms_max"] == 300.0


def test_errors_alone_can_mark_the_failure_point() -> None:
    records = [record(2, 100, 10.0), record(2, 200, 10.0, error_rate=0.05)]

    (result,) = summarize_load(records, TARGET_MS, MAX_ERRORS)["by_copies"]

    assert result["failure_point_users"] == 200


def test_no_failure_point_when_every_step_meets_the_target() -> None:
    records = [record(1, 100, 10.0), record(1, 200, 20.0), record(2, 100, 5.0)]

    summary = summarize_load(records, TARGET_MS, MAX_ERRORS)

    one, two = summary["by_copies"]
    assert one["failure_point_users"] is None
    assert one["highest_step_meeting_target"] == 200
    assert one["peak_requests_per_second"] == 200.0
    assert two["copies"] == 2
