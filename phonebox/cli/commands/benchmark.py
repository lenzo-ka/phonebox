"""Thin commands for running and reporting reproducible developer benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._common import expected_input_errors, require_distinct_output


def setup_benchmark_commands(subparsers) -> None:
    from phonebox.eval.benchmark_systems import SYSTEMS

    from .benchmark_receipt import setup_benchmark_receipt_command

    setup_benchmark_receipt_command(subparsers)
    patch = subparsers.add_parser(
        "benchmark-neural-patch",
        help="Export the credited patch for pinned DeepPhonemizer0.0.19",
    )
    patch.add_argument("--output", required=True, type=Path)
    patch.set_defaults(func=handle_neural_patch)
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
        choices=SYSTEMS,
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
    for name, default, description in (
        ("min", 20, "Minimum upstream EM iterations"),
        ("max", 100, "Initial upstream EM iteration cap"),
        ("extension", 200, "Fresh restart cap when the initial cap is reached"),
    ):
        parser.add_argument(
            f"--sequitur-{name}-iterations",
            type=int,
            default=default,
            help=f"{description} (default: {default})",
        )
    parser.add_argument(
        "--phonetisaurus-prefix",
        type=Path,
        help="Developer prefix containing Phonetisaurus and MITLM binaries",
    )
    parser.add_argument(
        "--neural-python",
        type=Path,
        help="Python in the pinned isolated DeepPhonemizer toolchain",
    )
    parser.add_argument(
        "--neural-device",
        choices=("cpu", "mps"),
        default="cpu",
        help="Neural device; MPS is seeded but not bit deterministic",
    )
    parser.add_argument(
        "--neural-profile-only",
        action="store_true",
        help="One full training epoch and dev evaluation; no test references or benchmark score",
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
    if args.system == "deepphonemizer" and args.neural_python is None:
        raise ValueError("DeepPhonemizer requires --neural-python")
    if args.neural_profile_only and args.system != "deepphonemizer":
        raise ValueError("--neural-profile-only requires --system deepphonemizer")
    if args.system == "deepphonemizer":
        if not args.neural_python.is_file():
            raise ValueError(
                "--neural-python must name an existing isolated interpreter"
            )
        for source in (
            args.neural_python,
            Path(str(args.neural_python) + ".provenance.json"),
            args.work_dir / "neural-model.pt",
        ):
            rejected = require_distinct_output(source, args.output)
            if rejected is not None:
                return rejected
    dataset = load_dataset(
        args.dataset, args.cache_dir, remove_stress=args.remove_stress
    )
    if args.neural_profile_only:
        from phonebox.eval.benchmark_neural import run_neural_training

        result = run_neural_training(
            dataset,
            args.work_dir,
            python_executable=args.neural_python,
            device=args.neural_device,
            profile_only=True,
        )
        write_results(args.output, result)
        return 0
    result = run_benchmark(
        dataset,
        args.system,
        args.work_dir,
        sequitur_executable=args.sequitur_executable,
        sequitur_min_iterations=args.sequitur_min_iterations,
        sequitur_max_iterations=args.sequitur_max_iterations,
        sequitur_extension_iterations=args.sequitur_extension_iterations,
        phonetisaurus_prefix=args.phonetisaurus_prefix,
        neural_python=args.neural_python,
        neural_device=args.neural_device,
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


@expected_input_errors
def handle_neural_patch(args: argparse.Namespace) -> int:
    from phonebox.eval.benchmark_neural import write_neural_patch

    digest = write_neural_patch(args.output)
    print(json.dumps({"patch_sha256": digest}))
    return 0
