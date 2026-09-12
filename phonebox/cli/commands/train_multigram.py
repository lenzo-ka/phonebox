#!/usr/bin/env python
"""phonebox train-multigram — train and export a MultigramG2P (n:m) model."""

from __future__ import annotations

import argparse
import sys

from ._common import expected_input_errors


def setup_train_multigram_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-multigram",
        help="Train MultigramG2P (n:m joint Viterbi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train a MultigramG2P model from a pronunciation lexicon and export "
            "``<output>.units.json`` + ``<output>.lm.json`` sidecars. "
            "Pronounce with ``phonebox pronounce -m <output>`` when the sidecar "
            "is present."
        ),
    )
    parser.add_argument(
        "--locale",
        required=True,
        help="Locale tag (case-insensitive; bare, hyphenated, or underscored, e.g. it, it-IT)",
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Pronunciation dictionary TSV (word\\tphone phone ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Model stem (e.g. model.g2p.gz → model.g2p.gz.units.json)",
    )
    parser.add_argument("--phoneset", default="ipa")
    parser.add_argument(
        "--remove-stress",
        action="store_true",
        help="Opt into phoneset-specific stress removal (default: preserve stress).",
    )
    parser.add_argument("--max-letter-span", type=int, default=None)
    parser.add_argument("--max-phone-span", type=int, default=None)
    parser.add_argument("--em-iterations", type=int, default=15)
    parser.add_argument("--lm-order", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--decode-beam", type=int, default=0)
    parser.add_argument("--parallel-align", action="store_true")
    parser.add_argument(
        "--no-config-joins",
        action="store_true",
        help="Disable locale config.json letter/phone joins before training.",
    )
    parser.add_argument(
        "--spelling-rewrite",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Rewrite one post-cooking grapheme (FROM=TO, repeatable).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=handle_train_multigram)


@expected_input_errors
def handle_train_multigram(args) -> int:
    from ...multigram_training import train_multigram
    from ._common import configure_progress_logging

    configure_progress_logging(args.verbose)

    rewrites = {}
    for item in args.spelling_rewrite:
        if "=" not in item:
            print(
                f"Error: invalid --spelling-rewrite {item!r}; expected FROM=TO",
                file=sys.stderr,
            )
            return 2
        source, target = item.split("=", 1)
        rewrites[source] = target
    result = train_multigram(
        args.lexicon,
        locale=args.locale,
        phoneset=args.phoneset,
        output=args.output,
        max_letter_span=args.max_letter_span,
        max_phone_span=args.max_phone_span,
        em_iterations=args.em_iterations,
        lm_order=args.lm_order,
        decode_beam=args.decode_beam,
        parallel_align=args.parallel_align,
        no_config_joins=args.no_config_joins,
        spelling_rewrites=rewrites,
        remove_stress=args.remove_stress,
        verbose=args.verbose,
    )
    print(f"Exported {result.units_path} and {result.lm_path}", file=sys.stderr)
    return 0
