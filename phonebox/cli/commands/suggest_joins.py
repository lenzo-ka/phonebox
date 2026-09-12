#!/usr/bin/env python
"""phonebox suggest-joins — discover letter/phone joins from a lexicon.

Runs the joint-multigram (n:m) EM aligner from
``phonebox.core.multigram_align`` over a pronunciation lexicon and
emits high-probability multi-letter and multi-phone units as
``config.json``-ready suggestions.

Usage::

    phonebox suggest-joins \\
        --lexicon path/to/<lang>_ipa.tsv \\
        --max-letter-span 3 --max-phone-span 2 \\
        --min-prob 0.001 --top 50 \\
        -o path/to/joins.json

The output is a structured JSON report with two ranked lists:
``letter_joins`` (letter tuples whose total mass across phone sides is
high) and ``phone_joins`` (phone tuples whose total mass across letter
sides is high). Each entry carries the unit, its summed probability
mass, and the modal counterpart (most likely matching subsequence on
the other side, with confidence).

Sampling reduces runtime for large lexicons, at the cost of coverage.
Rare spelling and phone patterns may be absent from a sample, so its
suggested joins can differ from those found using the full lexicon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import expected_input_errors


def setup_suggest_joins_command(subparsers):
    parser = subparsers.add_parser(
        "suggest-joins",
        help="Discover letter/phone joins from a lexicon (joint-multigram EM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="TSV lexicon (word\\tphone phone ...)",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Locale tag (e.g. fr_FR): apply config.json multigram span defaults",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-letter-span",
        type=int,
        default=None,
        help="Maximum letters per unit (default: locale multigram config or 3).",
    )
    parser.add_argument(
        "--max-phone-span",
        type=int,
        default=None,
        help="Maximum phones per unit (default: locale multigram config or 2).",
    )
    parser.add_argument(
        "--min-phone-span",
        type=int,
        default=0,
        help="0 allows silent-letter units (default), 1 forbids them.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.001,
        help="Drop units below this probability (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Keep at most N top-ranked join candidates per category (default: 50).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sub-sample N entries from the lexicon (0 = no sampling, "
        "use all). Recommended: 10000-20000 for lexicons >100k entries.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --sample (default: 42 for reproducibility).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="EM iteration ceiling (default: 15; convergence usually at 5-10).",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Relative LL-change for EM convergence (default: 1e-4).",
    )
    parser.add_argument(
        "--parallel-align",
        action="store_true",
        help="Parallel multigram EM E-step on large lexicons.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=handle_suggest_joins)


def _load_pairs(path: Path) -> list[tuple[list[str], list[str]]]:
    from ...join_discovery import load_join_pairs

    return load_join_pairs(path)


@expected_input_errors
def handle_suggest_joins(args) -> int:
    from ...join_discovery import discover_joins

    result = discover_joins(
        args.lexicon,
        locale=args.locale,
        output=args.output,
        max_letter_span=args.max_letter_span,
        max_phone_span=args.max_phone_span,
        min_phone_span=args.min_phone_span,
        min_prob=args.min_prob,
        top=args.top,
        sample=args.sample,
        sample_seed=args.sample_seed,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        parallel_align=args.parallel_align,
        verbose=args.verbose,
    )
    print(
        f"Wrote {len(result.letter_joins)} letter joins and {len(result.phone_joins)} phone joins to {result.output_path}",
        file=sys.stderr,
    )
    return 0
