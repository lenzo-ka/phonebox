"""CLI adapters for pinned exemplar generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def setup_exemplar_commands(subparsers) -> None:
    parser = subparsers.add_parser(
        "exemplars",
        help="Generate or verify ICU orthographic data (requires dev extra)",
    )
    commands = parser.add_subparsers(dest="exemplar_command", required=True)
    generate = commands.add_parser(
        "generate",
        help="Generate the pinned ICU standard and auxiliary inventories",
    )
    generate.add_argument(
        "--output", type=Path, required=True, help="Destination JSON inventory"
    )
    generate.add_argument(
        "--check",
        action="store_true",
        help="Check OUTPUT without writing; stale data exits 1",
    )
    generate.set_defaults(func=handle_generate)


def handle_generate(args: argparse.Namespace) -> int:
    from ...dev.exemplars import check_exemplars, write_exemplars

    try:
        if args.check:
            if not check_exemplars(args.output):
                print(
                    f"{args.output} is stale; regenerate without --check",
                    file=sys.stderr,
                )
                return 1
        else:
            write_exemplars(args.output)
    except (OSError, RuntimeError) as exc:
        print(f"Exemplar generation failed: {exc}", file=sys.stderr)
        return 2
    return 0
