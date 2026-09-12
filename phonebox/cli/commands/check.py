#!/usr/bin/env python
"""phonebox check — validate lexicon against a canonical phoneset.

Catches the bugs we keep stepping on the third time we use a language:

  * NFC vs NFD mismatches (silent killers: ``ã`` precomposed vs ``a + 0303``
    look identical but compare as different strings; common when the
    phoneset spec is hand-edited and the lexicon comes from a different
    pipeline). Detected by NFC-normalizing both sides before comparison;
    if it would change the result, the mismatch is reported as fixable.
  * Phones in the lexicon missing from the canonical phoneset (these are
    "xenophones" the downstream consumer will mark as out-of-spec).
  * Phones in the canonical phoneset that the lexicon never uses (likely
    placeholders for English-loanword handling — informational, not an
    error).
  * Lexicon words that contain non-NFC characters.

    For letter/phone join discovery use ``phonebox suggest-joins`` (joint-
    multigram EM), not this command.

Usage::

    phonebox check \\
        --lexicon path/to/<lang>_lex.tsv \\
        --phoneset path/to/<lang>.phoneset.json

    # exit code 0 = clean (or warnings only), 1 = NFC/spec problems, 2 = bad inputs

The lexicon accepts arbitrary whitespace with ``word\\tphone phone ...`` per line. The
phoneset file is a JSON array of allowed phone tokens. Both file
locations are arbitrary — pass whatever paths your project uses.

Library: phonebox.validation.validate_lexicon / validate_lexicon_file return
structured findings; format_lexicon_validation provides the report text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...constants import FILE_ENCODING
from ...validation import format_lexicon_validation, validate_lexicon_file


def setup_check_command(subparsers):
    parser = subparsers.add_parser(
        "check",
        help="Validate lexicon against canonical phoneset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--lexicon", required=True, help="Lexicon TSV file")
    parser.add_argument(
        "--phoneset",
        required=True,
        help="Canonical phoneset JSON (list of allowed phones)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing-from-spec phones as fatal (exit code 1 instead "
        "of warning).",
    )
    parser.add_argument(
        "--show-words",
        type=int,
        default=5,
        metavar="N",
        help="Show up to N example words for each lexicon→spec gap "
        "(default: 5; 0 to suppress).",
    )
    parser.set_defaults(func=handle_check)


def handle_check(args) -> int:
    try:
        spec = json.loads(Path(args.phoneset).read_text(encoding=FILE_ENCODING))
        if not isinstance(spec, list):
            raise ValueError("phoneset spec must be a JSON list of phone strings")
        result = validate_lexicon_file(args.lexicon, spec, show_words=args.show_words)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        f"=== phonebox check: {Path(args.lexicon).name} vs {Path(args.phoneset).name} ==="
    )
    print(format_lexicon_validation(result))
    return 1 if result.has_errors(strict=args.strict) else 0
