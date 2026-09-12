"""Shared argparse helpers for CLI subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec

from ...constants import DEFAULT_LOCALE, DEFAULT_PHONESET
from ...utils.io import paths_refer_to_same_file

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
    definitions; context width is shared by all these consumers.
    """
    parser.add_argument(
        "--locale",
        default=DEFAULT_LOCALE,
        help=(
            "Language locale (case-insensitive; bare, hyphenated, or underscored, "
            "e.g. en, en-US)"
        ),
    )
    parser.add_argument(
        "--phoneset", default=DEFAULT_PHONESET, help="Phoneset name (e.g. cmu, ipa)"
    )
    add_width_arg(parser)
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


P = ParamSpec("P")


def expected_input_errors(handler: Callable[P, int]) -> Callable[P, int]:
    """Translate anticipated file/config/dependency failures at the CLI boundary."""

    @wraps(handler)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return handler(*args, **kwargs)
        except (ImportError, OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            if isinstance(error, ImportError) and "sklearn" in str(
                error
            ).lower().replace("scikit-learn", "sklearn"):
                print(
                    "Install the optional backend: pip install phonebox[sklearn]",
                    file=sys.stderr,
                )
            return EXIT_BAD_INPUT

    return wrapped


def require_distinct_output(
    source: str | Path, output: str | Path | None
) -> int | None:
    """Reject input/output aliases before a command opens its output."""
    if output is not None and paths_refer_to_same_file(source, output):
        print("Error: input and output must be different files", file=sys.stderr)
        return EXIT_BAD_INPUT
    return None


def add_width_arg(parser: argparse.ArgumentParser) -> None:
    """Add the context width consumed by vectorization and prepared training."""
    parser.add_argument(
        "--width", type=int, default=7, help="Odd context width (default: 7)"
    )
