#!/usr/bin/env python
"""
Vectorize command: convert alignments to feature vectors.

Pipeline phase: Alignments → Vectors (~2 sec)
"""

from __future__ import annotations

import sys

from ...constants import FILE_ENCODING
from ._common import add_vectorizer_args, require_file


def setup_vectorize_command(subparsers):
    """Setup vectorize command (alignments → vectors)."""
    vectorize_parser = subparsers.add_parser(
        "vectorize",
        help="Convert alignments to vectors",
        description="Phase 2: Convert alignments to feature vectors",
    )
    vectorize_parser.add_argument("alignments", help="Alignment file")
    vectorize_parser.add_argument(
        "-o", "--output", required=True, help="Output vector file"
    )
    add_vectorizer_args(vectorize_parser, cased=True, target_first=True)
    vectorize_parser.set_defaults(func=handle_vectorize)


def handle_vectorize(args):
    """Phase 2: Alignments → Vectors."""
    from ...core.vectorizer import Vectorizer

    if (rc := require_file(args.alignments, "alignments")) is not None:
        return rc

    print(f"Vectorizing {args.alignments}...", file=sys.stderr)

    vectorizer = Vectorizer(
        locale=args.locale,
        phoneset_name=args.phoneset,
        remove_stress=args.remove_stress,
        cased=args.cased,
        verbose=False,
        target_position="first" if args.target_first else "last",
    )

    # Read alignments and write vectors
    with (
        open(args.alignments, encoding=FILE_ENCODING) as infile,
        open(args.output, "w", encoding=FILE_ENCODING) as outfile,
    ):
        vectorizer.vectorize_file(infile, outfile, header=False)

    # Count vectors
    with open(args.output, encoding=FILE_ENCODING) as f:
        vector_count = sum(1 for line in f if line.strip())

    print(f"Done: Exported {vector_count:,} vectors to {args.output}", file=sys.stderr)
    return 0
