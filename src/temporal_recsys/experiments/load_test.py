"""Load-test the service in the local kind cluster, step by step, until it misses its target.

Each step runs k6 in its own container on the cluster's network, pinned to CPUs the
cluster's nodes do not use. Each finished step is saved at once, so a stopped run resumes.
Results hold totals and percentiles only, never user IDs.
"""

import json
import logging
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from temporal_recsys.experiments.baseline import SUMMARY_FILE, load_finished

LOGGER = logging.getLogger(__name__)

MEASUREMENTS_FILE = "measurements.jsonl"
SERVICE_NODES = ("recsys-worker", "recsys-worker2")
CONTROL_NODE = "recsys-control-plane"
TARGET_PORT = 30080
SETTLE_SECONDS = 5
MEASURED = "{phase:measured}"


@dataclass(frozen=True)
class LoadPlan:
    id: str
    copies: tuple[int, ...]
    user_steps: tuple[int, ...]
    repetitions: int
    order_seed: int
    measured_seconds: int
    p95_target_ms: float
    max_error_rate: float
    service_cpus: str
    control_cpus: str
    load_cpus: str
    k6_image: str

    def __post_init__(self) -> None:
        if list(self.user_steps) != sorted(set(self.user_steps)):
            raise ValueError("User steps must rise with no repeats")


def load_plan(path: Path) -> LoadPlan:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    design, target, cpus = plan["design"], plan["target"], plan["design"]["cpus"]
    return LoadPlan(
        id=plan["id"],
        copies=tuple(design["copies"]),
        user_steps=tuple(design["user_steps"]),
        repetitions=design["repetitions"],
        order_seed=design["order_seed"],
        measured_seconds=design["measured_seconds"],
        p95_target_ms=target["p95_ms"],
        max_error_rate=target["max_error_rate"],
        service_cpus=cpus["service_nodes"],
        control_cpus=cpus["control_node"],
        load_cpus=cpus["load_generator"],
        k6_image=design["k6_image"],
    )


def run_load_test(
    plan: LoadPlan, repo_dir: Path, output_dir: Path, code_commit: str
) -> dict[str, Any]:
    """Run every step for each number of copies, in a freshly shuffled order each pass."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MEASUREMENTS_FILE
    finished = load_finished(path)
    _pin_nodes(plan)
    target = f"http://{_node_address(SERVICE_NODES[0])}:{TARGET_PORT}"
    rng = np.random.default_rng(plan.order_seed)

    for pass_number in range(plan.repetitions):
        for copies in rng.permutation(plan.copies).tolist():
            steps = [
                users
                for users in plan.user_steps
                if _arm(pass_number, copies, users) not in finished
            ]
            if not steps:
                continue
            _scale(copies)
            for users in steps:
                arm = _arm(pass_number, copies, users)
                metrics = _run_step(plan, repo_dir, output_dir, target, users)
                record = {
                    "arm": arm,
                    "pass": pass_number,
                    "copies": copies,
                    "users": users,
                } | step_measurement(metrics, plan.measured_seconds)
                with path.open("a", encoding="utf-8") as measurements:
                    measurements.write(json.dumps(record) + "\n")
                finished[arm] = record
                LOGGER.info(
                    "%s: %.0f requests/s, p95 %.1f ms, errors %.2f%%",
                    arm,
                    record["requests_per_second"],
                    record["p95_ms"],
                    100 * record["error_rate"],
                )
                time.sleep(SETTLE_SECONDS)

    summary = {
        "plan": plan.id,
        "code_commit": code_commit,
        "setup": {
            "k6_image": plan.k6_image,
            "service_cpus": plan.service_cpus,
            "control_cpus": plan.control_cpus,
            "load_cpus": plan.load_cpus,
        },
    } | summarize_load(list(finished.values()), plan.p95_target_ms, plan.max_error_rate)
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def step_measurement(metrics: dict[str, Any], measured_seconds: int) -> dict[str, float]:
    """Pick the measured-phase numbers from a k6 summary. Warm-up requests are left out."""
    try:
        duration = metrics[f"http_req_duration{MEASURED}"]["values"]
        requests = metrics[f"http_reqs{MEASURED}"]["values"]["count"]
        error_rate = metrics[f"http_req_failed{MEASURED}"]["values"]["rate"]
    except KeyError as missing:
        raise ValueError(f"The k6 summary has no measured requests: {missing}") from None
    return {
        "requests": requests,
        # k6's own rate spans the warm-up too, so divide by the measured time only.
        "requests_per_second": requests / measured_seconds,
        "p50_ms": duration["med"],
        "p95_ms": duration["p(95)"],
        "p99_ms": duration["p(99)"],
        "max_ms": duration["max"],
        "error_rate": error_rate,
    }


def summarize_load(
    records: Sequence[dict[str, Any]], p95_target_ms: float, max_error_rate: float
) -> dict[str, Any]:
    """Report medians and ranges per step, and the failure point for each number of copies.

    A step meets the target if its median p95 and median error rate across passes are both
    within the limits. The failure point is the fewest users at which a step does not.
    """
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["copies"], record["users"]), []).append(record)

    steps = []
    for (copies, users), items in sorted(grouped.items()):
        step: dict[str, Any] = {"copies": copies, "users": users, "passes": len(items)}
        for key in ("requests_per_second", "p50_ms", "p95_ms", "p99_ms", "error_rate"):
            values = [item[key] for item in items]
            step |= {
                f"{key}_median": statistics.median(values),
                f"{key}_min": min(values),
                f"{key}_max": max(values),
            }
        step["meets_target"] = (
            step["p95_ms_median"] <= p95_target_ms and step["error_rate_median"] <= max_error_rate
        )
        steps.append(step)

    by_copies = []
    for copies in sorted({step["copies"] for step in steps}):
        own = [step for step in steps if step["copies"] == copies]
        failing = [step["users"] for step in own if not step["meets_target"]]
        failure_point = min(failing, default=None)
        passing = [
            step["users"]
            for step in own
            if step["meets_target"] and (failure_point is None or step["users"] < failure_point)
        ]
        by_copies.append(
            {
                "copies": copies,
                "failure_point_users": failure_point,
                "highest_step_meeting_target": max(passing, default=None),
                "peak_requests_per_second": max(step["requests_per_second_median"] for step in own),
            }
        )
    return {"steps": steps, "by_copies": by_copies}


def _arm(pass_number: int, copies: int, users: int) -> str:
    return f"{pass_number}/copies={copies}/users={users}"


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603


def _pin_nodes(plan: LoadPlan) -> None:
    _run(["docker", "update", "--cpuset-cpus", plan.service_cpus, *SERVICE_NODES])
    _run(["docker", "update", "--cpuset-cpus", plan.control_cpus, CONTROL_NODE])


def _node_address(node: str) -> str:
    template = "{{.NetworkSettings.Networks.kind.IPAddress}}"
    return _run(["docker", "inspect", node, "--format", template]).strip()


def _scale(copies: int) -> None:
    _run(["kubectl", "scale", "deployment/recsys", f"--replicas={copies}"])
    _run(["kubectl", "rollout", "status", "deployment/recsys", "--timeout=180s"])
    # Let copies that are shutting down drain before measuring.
    time.sleep(3 * SETTLE_SECONDS)


def _run_step(
    plan: LoadPlan, repo_dir: Path, output_dir: Path, target: str, users: int
) -> dict[str, Any]:
    step_file = output_dir / "step.json"
    step_file.unlink(missing_ok=True)
    _run(
        [
            "docker", "run", "--rm", "--network", "kind",
            "--cpuset-cpus", plan.load_cpus,
            "-v", f"{(repo_dir / 'deploy' / 'load').resolve()}:/scripts:ro",
            "-v", f"{(repo_dir / 'artifacts').resolve()}:/load:ro",
            "-v", f"{output_dir.resolve()}:/out",
            "-e", f"VUS={users}",
            "-e", f"SECONDS={plan.measured_seconds}",
            "-e", f"TARGET={target}",
            "-e", "OUT=/out/step.json",
            plan.k6_image, "run", "--quiet", "/scripts/recommend.js",
        ]
    )  # fmt: skip
    metrics: dict[str, Any] = json.loads(step_file.read_text(encoding="utf-8"))
    step_file.unlink()
    return metrics
