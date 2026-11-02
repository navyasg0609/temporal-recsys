"""Command line entry point, installed as `trs`."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from temporal_recsys.data.benchmark import load_plan, run_benchmark
from temporal_recsys.data.pipeline import MONTHLY_DIR, ingest
from temporal_recsys.data.quality import NETFLIX_PRIZE
from temporal_recsys.data.spark import create_spark_session
from temporal_recsys.data.synthetic import SyntheticConfig, generate_ratings, write_raw_file
from temporal_recsys.experiments.baseline import load_plan as load_baseline_plan
from temporal_recsys.experiments.baseline import run_baseline
from temporal_recsys.experiments.history import (
    NO_SEED,
    POPULARITY,
    SPARK_ALS,
    TWO_TOWER,
    popularity_trainer,
    run_history,
    spark_als_trainer,
    two_tower_trainer,
)
from temporal_recsys.experiments.history import load_plan as load_history_plan
from temporal_recsys.experiments.spark_als import load_plan as load_spark_als_plan
from temporal_recsys.experiments.spark_als import run_spark_als
from temporal_recsys.experiments.training_latency import load_plan as load_latency_plan
from temporal_recsys.experiments.training_latency import run_training_latency


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

    spark_als = commands.add_parser("spark-als", help="train Spark ALS on a registered plan")
    spark_als.add_argument("--data-dir", type=Path, required=True)
    spark_als.add_argument("--plan", type=Path, required=True)
    spark_als.add_argument("--output-dir", type=Path, required=True)
    spark_als.add_argument("--code-commit", required=True)
    spark_als.add_argument("--driver-memory", default="2g")

    latency = commands.add_parser("training-latency", help="time training on each storage layout")
    latency.add_argument("--data-dir", type=Path, required=True)
    latency.add_argument("--plan", type=Path, required=True)
    latency.add_argument("--output-dir", type=Path, required=True)
    latency.add_argument("--code-commit", required=True)
    latency.add_argument("--driver-memory", default="2g")

    two_tower = commands.add_parser("two-tower", help="train the neural model on a registered plan")
    two_tower.add_argument("--monthly-dir", type=Path, required=True)
    two_tower.add_argument("--raw-dir", type=Path, required=True)
    two_tower.add_argument("--plan", type=Path, required=True)
    two_tower.add_argument("--output-dir", type=Path, required=True)
    two_tower.add_argument("--code-commit", required=True)

    diagnostic = commands.add_parser("two-tower-diagnostic", help="explain the two-tower score")
    diagnostic.add_argument("--monthly-dir", type=Path, required=True)
    diagnostic.add_argument("--raw-dir", type=Path, required=True)
    diagnostic.add_argument("--plan", type=Path, required=True)
    diagnostic.add_argument("--output-dir", type=Path, required=True)
    diagnostic.add_argument("--code-commit", required=True)

    history = commands.add_parser("history", help="train one model on every history window")
    history.add_argument("--model", choices=(POPULARITY, SPARK_ALS, TWO_TOWER), required=True)
    history.add_argument("--data-dir", type=Path, required=True)
    history.add_argument("--raw-dir", type=Path, help="film titles, for the two-tower model")
    history.add_argument("--plan", type=Path, required=True)
    history.add_argument("--output-dir", type=Path, required=True)
    history.add_argument("--code-commit", required=True)
    history.add_argument("--driver-memory", default="2g")
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


def run_spark_als_command(args: argparse.Namespace) -> int:
    spark = create_spark_session("spark-als", driver_memory=args.driver_memory)
    try:
        summary = run_spark_als(
            spark, args.data_dir, load_spark_als_plan(args.plan), args.output_dir, args.code_commit
        )
    finally:
        spark.stop()
    print(json.dumps(summary, indent=2))
    return 0


def run_training_latency_command(args: argparse.Namespace) -> int:
    spark = create_spark_session("training-latency", driver_memory=args.driver_memory)
    try:
        summary = run_training_latency(
            spark, args.data_dir, load_latency_plan(args.plan), args.output_dir, args.code_commit
        )
    finally:
        spark.stop()
    print(json.dumps(summary, indent=2))
    return 0


def run_two_tower_command(args: argparse.Namespace) -> int:
    # PyTorch is an optional extra, so only this command may require it.
    from temporal_recsys.experiments.two_tower import load_plan as load_two_tower_plan
    from temporal_recsys.experiments.two_tower import pick_device, run_two_tower

    device = pick_device()
    summary = run_two_tower(
        args.monthly_dir,
        args.raw_dir,
        load_two_tower_plan(args.plan),
        args.output_dir,
        args.code_commit,
        device,
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_two_tower_diagnostic_command(args: argparse.Namespace) -> int:
    # PyTorch is an optional extra, so only this command may require it.
    from temporal_recsys.experiments.two_tower import pick_device
    from temporal_recsys.experiments.two_tower_diagnostic import load_plan as load_diagnostic_plan
    from temporal_recsys.experiments.two_tower_diagnostic import run_diagnostic

    summary = run_diagnostic(
        args.monthly_dir,
        args.raw_dir,
        load_diagnostic_plan(args.plan),
        args.output_dir,
        args.code_commit,
        pick_device(),
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_history_command(args: argparse.Namespace) -> int:
    plan = load_history_plan(args.plan)
    monthly_dir = args.data_dir / MONTHLY_DIR
    if args.model == POPULARITY:
        trainer = popularity_trainer(plan)
        summary = run_history(
            monthly_dir,
            plan,
            POPULARITY,
            trainer,
            (NO_SEED,),
            args.output_dir,
            args.code_commit,
            {},
        )
    elif args.model == SPARK_ALS:
        spark = create_spark_session("history", driver_memory=args.driver_memory)
        try:
            machine = {"spark": spark.version, "master": spark.sparkContext.master}
            summary = run_history(
                monthly_dir,
                plan,
                SPARK_ALS,
                spark_als_trainer(spark, args.data_dir, plan),
                plan.seeds,
                args.output_dir,
                args.code_commit,
                machine,
            )
        finally:
            spark.stop()
    else:
        # PyTorch is an optional extra, so only this branch may require it.
        from temporal_recsys.experiments.two_tower import pick_device

        device = pick_device()
        summary = run_history(
            monthly_dir,
            plan,
            TWO_TOWER,
            two_tower_trainer(args.raw_dir, plan, args.output_dir, device),
            plan.seeds,
            args.output_dir,
            args.code_commit,
            {"device": device},
        )
    print(json.dumps(summary, indent=2))
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
        "spark-als": run_spark_als_command,
        "training-latency": run_training_latency_command,
        "two-tower": run_two_tower_command,
        "two-tower-diagnostic": run_two_tower_diagnostic_command,
        "history": run_history_command,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
