#!/usr/bin/env python
"""Alignment operations: align letters to phonemes."""

from __future__ import annotations

import sys

from ._common import add_vectorizer_args, require_file


def setup_align_commands(subparsers):
    """Setup align command."""
    align_parser = subparsers.add_parser(
        "align",
        help="Align letters to phonemes",
        description="Align letters to phonemes using EM algorithm",
    )
    align_parser.add_argument("dict", help="Dictionary file")
    align_parser.add_argument("-o", "--output", help="Output alignment file")
    add_vectorizer_args(align_parser)
    align_parser.add_argument("--max-iterations", type=int, help="Max EM iterations")
    align_parser.add_argument(
        "--max-combinations", type=int, help="Max combinations per alignment"
    )
    align_parser.set_defaults(func=handle_align)


def handle_align(args):
    """Handle 'phonebox align' command."""
    from ...constants import DICT_ENCODING
    from ...core.em_align import EMAlign
    from ...core.vectorizer import Vectorizer
    from ...utils.io import is_dict_comment

    if (rc := require_file(args.dict, "dictionary")) is not None:
        return rc

    print(f"Aligning dictionary: {args.dict}", file=sys.stderr)

    # Create vectorizer with same settings that will be used for vectorization
    vectorizer = Vectorizer(
        locale=args.locale,
        phoneset_name=args.phoneset,
        remove_stress=args.remove_stress,
        verbose=True,
    )

    # Create EM aligner
    em = EMAlign(
        vectorizer,
        max_iterations=args.max_iterations,
        max_combinations=args.max_combinations,
        verbose=True,
    )

    # Load dictionary
    with open(args.dict, encoding=DICT_ENCODING) as f:
        for line in f:
            if not is_dict_comment(line):
                em.add_line(line)

    # Run alignment
    em.align()

    # Write output
    if args.output:
        with open(args.output, "w", encoding=DICT_ENCODING) as f:
            em.write(f)
        print(f"Done: Alignments saved to {args.output}", file=sys.stderr)
    else:
        em.write(sys.stdout)

    return 0
