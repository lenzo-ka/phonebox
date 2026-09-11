"""Thin CLI for the reproducible CMUdict model comparison."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

from phonebox.eval.cmudict_compare import (
    fetch_cmudict,
    render_markdown,
    run_cmudict_comparison,
    write_results,
)


def setup_cmudict_compare_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "cmudict", help="Reproduce the pinned CMUdict CART/n:m benchmark"
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        help="Verified pinned cmudict.dict copy (default: download it)",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--refresh", type=Path, help="Run the benchmark and write JSON results"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Rendered report path (default: docs/CMUDICT_COMPARISON.md)",
    )
    action.add_argument(
        "--check",
        nargs=2,
        type=Path,
        metavar=("JSON", "MARKDOWN"),
        help="Check that MARKDOWN is the rendering of committed JSON",
    )
    parser.add_argument(
        "--em-iterations",
        type=int,
        default=10,
        help="Multigram EM iterations for a refresh (default: 10)",
    )
    parser.set_defaults(func=handle_compare_cmudict)


def handle_compare_cmudict(args: argparse.Namespace) -> int:
    if args.check:
        try:
            result_path, markdown_path = args.check
            expected = render_markdown(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
            actual = markdown_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"CMUdict comparison check failed: {exc}", file=sys.stderr)
            return 2
        return 0 if actual == expected else 1
    try:
        if args.lexicon is None:
            with tempfile.TemporaryDirectory() as tmp:
                lexicon = fetch_cmudict(Path(tmp) / "cmudict.dict")
                result = run_cmudict_comparison(
                    lexicon, em_iterations=args.em_iterations, progress=print
                )
        else:
            result = run_cmudict_comparison(
                args.lexicon, em_iterations=args.em_iterations, progress=print
            )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"CMUdict comparison failed: {exc}", file=sys.stderr)
        return 2
    write_results(args.refresh, result)
    markdown = args.markdown or Path("docs/CMUDICT_COMPARISON.md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(result), encoding="utf-8")
    return 0
