"""Thin commands for running and reporting reproducible developer benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._common import expected_input_errors, require_distinct_output


def setup_benchmark_commands(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Run one pinned dataset/system experiment (developer toolchains)",
        description=(
            "Train and evaluate on fixed, shared train/dev/test data. "
            "External systems need separately installed developer toolchains; "
            "see docs/REPRODUCIBLE_BENCHMARKS.md."
        ),
    )
    parser.add_argument(
        "--dataset", required=True, choices=("cmudict", "french", "italian")
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=("cart", "multigram", "sequitur", "phonetisaurus"),
    )
    parser.add_argument(
        "--remove-stress",
        action="store_true",
        help="Remove CMU stress digits before deduplication (CMUdict only)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/data"),
        help="Verified download cache (default: .cache/benchmarks/data)",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Experiment directory for models and logs; use a separate directory per run",
    )
    parser.add_argument(
        "--sequitur-executable",
        type=Path,
        help="Pinned Sequitur executable or isolated-environment launcher",
    )
    parser.add_argument(
        "--phonetisaurus-prefix",
        type=Path,
        help="Developer prefix containing Phonetisaurus and MITLM binaries",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Aggregate result JSON; raw predictions remain in the work directory",
    )
    parser.set_defaults(func=handle_benchmark)

    report = subparsers.add_parser(
        "benchmark-report",
        help="Render measured benchmark JSON as a Markdown comparison",
    )
    report.add_argument(
        "results", nargs="+", type=Path, help="Result JSON files from compare benchmark"
    )
    report.add_argument("--output", required=True, type=Path)
    report.set_defaults(func=handle_benchmark_report)


@expected_input_errors
def handle_benchmark(args: argparse.Namespace) -> int:
    from phonebox.eval.benchmark import run_benchmark
    from phonebox.eval.benchmark_data import load_dataset
    from phonebox.eval.cmudict_compare import write_results

    if args.remove_stress and args.dataset != "cmudict":
        raise ValueError("--remove-stress applies only to CMUdict")
    if args.system == "sequitur" and args.sequitur_executable is None:
        raise ValueError("Sequitur requires --sequitur-executable")
    if args.system == "phonetisaurus" and args.phonetisaurus_prefix is None:
        raise ValueError("Phonetisaurus requires --phonetisaurus-prefix")
    dataset = load_dataset(
        args.dataset, args.cache_dir, remove_stress=args.remove_stress
    )
    result = run_benchmark(
        dataset,
        args.system,
        args.work_dir,
        sequitur_executable=args.sequitur_executable,
        phonetisaurus_prefix=args.phonetisaurus_prefix,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    write_results(args.output, result)
    return 0


@expected_input_errors
def handle_benchmark_report(args: argparse.Namespace) -> int:
    from phonebox.eval.benchmark_report import render_benchmark_report

    for path in args.results:
        if (status := require_distinct_output(path, args.output)) is not None:
            return status
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    rendered = render_benchmark_report(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0
