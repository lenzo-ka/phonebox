"""CLI parsing for the library-owned optional-tool receipt producer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import expected_input_errors


def setup_benchmark_receipt_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark-receipt",
        help="Bind a tool binary to its declared version and checked Git source",
        description="Write a path-free provenance receipt beside the resolved executable.",
    )
    parser.add_argument("executable", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--version",
        required=True,
        help="Declared source version, not inferred binary output",
    )
    parser.add_argument(
        "--expected-revision", required=True, help="Full expected source Git HEAD"
    )
    parser.add_argument(
        "--build-metadata",
        help="JSON object or JSON file of public compiler/dependency facts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Receipt destination (default: resolved executable + .provenance.json)",
    )
    parser.set_defaults(func=handle_benchmark_receipt)


@expected_input_errors
def handle_benchmark_receipt(args: argparse.Namespace) -> int:
    from phonebox.eval.benchmark import _invalid_constant
    from phonebox.eval.benchmark_provenance import write_tool_receipt

    build = None
    if args.build_metadata is not None:
        text = args.build_metadata
        if not text.lstrip().startswith(("{", "[")):
            text = Path(text).read_text(encoding="utf-8")
        build = json.loads(text, parse_constant=_invalid_constant)
    result = write_tool_receipt(
        args.executable,
        args.source_dir,
        declared_version=args.version,
        expected_revision=args.expected_revision,
        build=build,
        output=args.output,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0
