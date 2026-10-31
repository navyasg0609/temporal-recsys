"""Command line entry point, installed as `trs`."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from temporal_recsys.data.benchmark import load_plan, run_benchmark
from temporal_recsys.data.pipeline import ingest
from temporal_recsys.data.quality import NETFLIX_PRIZE
from temporal_recsys.data.spark import create_spark_session
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file
from temporal_recsys.experiments.baseline import load_plan as load_baseline_plan
from temporal_recsys.experiments.baseline import run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("synthetic", help="write ratings with a planted change")
    synthetic.add_argument("--output-dir", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=0)

    ingest_command = commands.add_parser("ingest", help="build validated monthly data")
    ingest_command.add_argument("--raw-dir", type=Path, required=True)
    ingest_command.add_argument("--output-dir", type=Path, required=True)
    ingest_command.add_argument("--driver-memory", default="2g")
    ingest_command.add_argument(
        "--check-netflix", action="store_true", help="require the known Netflix Prize totals"
    )

    benchmark = commands.add_parser("benchmark", help="time a registered read benchmark")
    benchmark.add_argument("--data-dir", type=Path, required=True)
    benchmark.add_argument("--plan", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--code-commit", required=True)
    benchmark.add_argument("--driver-memory", default="2g")

    baseline = commands.add_parser("baseline", help="compare baselines on a registered plan")
    baseline.add_argument("--monthly-dir", type=Path, required=True)
    baseline.add_argument("--plan", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--code-commit", required=True)
    return parser


def run_baseline_command(args: argparse.Namespace) -> int:
    summary = run_baseline(
        args.monthly_dir, load_baseline_plan(args.plan), args.output_dir, args.code_commit
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_benchmark_command(args: argparse.Namespace) -> int:
    spark = create_spark_session("benchmark", driver_memory=args.driver_memory)
    try:
        result = run_benchmark(spark, args.data_dir, load_plan(args.plan), args.code_commit)
    finally:
        spark.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("windows", "summary")}, indent=2))
    return 0


def run_synthetic(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_ratings(SyntheticConfig(seed=args.seed))
    write_raw_file(data.ratings, args.output_dir / "combined_data_1.txt")
    print(f"Wrote {len(data.ratings)} ratings to {args.output_dir}")
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    expected = NETFLIX_PRIZE if args.check_netflix else None
    spark = create_spark_session("ingest", driver_memory=args.driver_memory)
    try:
        problems = ingest(spark, args.raw_dir, args.output_dir, expected)
    finally:
        spark.stop()

    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"All quality checks passed for {args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    commands = {
        "synthetic": run_synthetic,
        "ingest": run_ingest,
        "benchmark": run_benchmark_command,
        "baseline": run_baseline_command,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
