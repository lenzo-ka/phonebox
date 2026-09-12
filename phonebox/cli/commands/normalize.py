"""Normalize command: preview model-independent text tokenization."""

from __future__ import annotations

import sys

from ...constants import FILE_ENCODING
from ._common import expected_input_errors


def setup_normalize_command(subparsers):
    """Setup normalize command."""
    parser = subparsers.add_parser(
        "normalize",
        help="Preview model-independent text tokenization",
        description="""Preview text tokenization used by pronounce_text.

This does not apply a model's locale-specific grapheme transformations. It:
- NFC Unicode normalization
- Strips punctuation/symbols from token edges
- Preserves internal apostrophes and hyphens

Use --raw to see simple whitespace tokenization without normalization.

Examples:
  phonebox normalize "Hello, world!"
  phonebox normalize -f input.txt
  phonebox normalize --raw "Hello, world!"
  echo "Hello, world!" | phonebox normalize""",
    )
    parser.add_argument("text", nargs="*", help="Text to tokenize")
    parser.add_argument(
        "-f", "--file", metavar="FILE", help="Read text from file (- for stdin)"
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Just split on whitespace (no normalization)",
    )
    parser.set_defaults(func=handle_normalize)


@expected_input_errors
def handle_normalize(args):
    """Handle 'phonebox normalize' command."""
    from ...normalize import normalize_text, tokenize_raw

    def process_line(line):
        tokens = tokenize_raw(line) if args.raw else normalize_text(line)
        if tokens:
            print(" ".join(tokens))

    # Get input
    if args.file:
        f = sys.stdin if args.file == "-" else open(args.file, encoding=FILE_ENCODING)  # noqa: SIM115
        try:
            for line in f:
                process_line(line)
        finally:
            if args.file != "-":
                f.close()
    elif args.text:
        process_line(" ".join(args.text))
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            process_line(line)
    else:
        # Interactive mode
        mode_msg = "tokenize" if args.raw else "normalize"
        print(f"Enter text to {mode_msg} (Ctrl+D to exit):", file=sys.stderr)
        try:
            while True:
                line = input("> ")
                tokens = tokenize_raw(line) if args.raw else normalize_text(line)
                print(" ".join(tokens))
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0

    return 0
