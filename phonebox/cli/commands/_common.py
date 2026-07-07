"""Shared argparse helpers for CLI subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...constants import DEFAULT_LOCALE, DEFAULT_PHONESET

# Conventional exit code for a bad-input / usage error (missing file, etc.).
EXIT_BAD_INPUT = 2


def require_file(path: str | Path, label: str = "file") -> int | None:
    """Return EXIT_BAD_INPUT (after printing a message) if *path* is missing.

    Lets command handlers guard their inputs uniformly::

        if (rc := require_file(args.dict, "dictionary")) is not None:
            return rc

    Returns None when the file exists, so callers proceed normally.
    """
    if not Path(path).is_file():
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        return EXIT_BAD_INPUT
    return None


def add_vectorizer_args(
    parser: argparse.ArgumentParser,
    *,
    cased: bool = False,
    target_first: bool = False,
) -> None:
    """Add the locale/phoneset/stress options shared by vectorizer-driven commands.

    ``cased`` and ``target_first`` add the corresponding optional flags only for
    commands that consume them, keeping each command's surface identical to its
    hand-written original while sharing one definition and one set of defaults.
    """
    parser.add_argument(
        "--locale", default=DEFAULT_LOCALE, help="Language locale (e.g. en_US)"
    )
    parser.add_argument(
        "--phoneset", default=DEFAULT_PHONESET, help="Phoneset name (e.g. cmu, ipa)"
    )
    parser.add_argument(
        "--remove-stress", action="store_true", help="Remove stress markers"
    )
    if cased:
        parser.add_argument(
            "--cased", action="store_true", help="Case-sensitive letters"
        )
    if target_first:
        parser.add_argument(
            "--target-first",
            action="store_true",
            help="Place target column first (default: last)",
        )
