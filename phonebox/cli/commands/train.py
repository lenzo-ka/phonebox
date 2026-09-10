#!/usr/bin/env python
"""phonebox train — end-to-end G2P model build with safe defaults.

This is the "do the right thing" wrapper around the lower-level
`align` / `vectorize` / `model train` plumbing. The defaults reflect
lessons learned across several language builds:

  * native trainer (sklearn balloons memory on lexicons over ~100k
    entries — observed >200 GB RSS on a 428k-entry IPA lexicon before
    OOM-kill)
  * EMAlign.parallel_align=False (the fork pool duplicates the lexicon
    per worker; safer on big inputs)
  * EMAlign.max_combinations defaults to phonebox.constants.DEFAULT_MAX_COMBINATIONS
    (currently 10000 — keeps >99% of entries across the major European
    IPA lexicons we've tested while bounding RSS)
  * Alignments are dumped to a checkpoint file next to the model output
    BEFORE the tree-train step, so a re-train with different prune /
    criterion settings can skip the slow EM phase via
    `phonebox model train --alignments ...`.
  * prune=True with validation_split=0.05 by default — pruning reduces
    overfitting noise and roughly halves the model size with no
    measurable accuracy hit.

Usage::

    phonebox train --locale fr_FR \\
        --lexicon path/to/fr_lexicon.tsv \\
        -o path/to/fr-fr-ipa.g2p.gz

The lexicon file is expected to be TSV with one entry per line:
``word\\tphone phone phone``. Optional ``(N)`` variant suffixes on the
word (CMUdict-style) are stripped automatically. Variant pronunciations
for the same word are kept as separate training examples; the most
common rendering wins at inference time.

By default the alignments file is named ``<output-stem>_alignments.txt``
next to the model output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_train_command(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="End-to-end G2P training (native trainer, safe defaults)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Locale tag (case-insensitive; bare, hyphenated, or underscored, e.g. fr, fr-FR)",
    )
    parser.add_argument(
        "--lexicon",
        default=None,
        help="Pronunciation dictionary TSV (word\\tphone phone ...)",
    )
    parser.add_argument("-o", "--output", default=None, help="Model output (.g2p.gz)")
    parser.add_argument(
        "-c", "--config", help="YAML, TOML, or JSON training configuration"
    )
    parser.add_argument(
        "--phoneset",
        default=None,
        help="Phoneset tag — drives locale config lookups (default: ipa)",
    )
    parser.add_argument(
        "--alignments-out",
        default=None,
        help="Where to checkpoint EM alignments before training. "
        "Default: <output>_alignments.txt next to the model.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Cap on alignment combinations per word (default: 10000; 0 disables)",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prune with validation data (default: enabled)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction held out for pruning (default: 0.05).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Fraction reserved for held-out evaluation (default: 0).",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        choices=["native", "sklearn"],
        help="Tree trainer backend (default: native — safer on big lexicons).",
    )
    parser.add_argument(
        "--parallel-align",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable EMAlign multiprocessing pool. Off by default because "
        "the fork pool duplicates the lexicon per worker.",
    )
    parser.add_argument("--width", type=int, default=None, help="Odd context width")
    parser.add_argument(
        "--store-distributions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store leaf distributions for pronunciation scoring (default: enabled)",
    )
    parser.add_argument(
        "--remove-stress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Strip phoneset stress markers while loading the lexicon",
    )
    parser.add_argument("--cased", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--norm-xlit", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=None)
    parser.set_defaults(func=handle_train)


def handle_train(args) -> int:
    from ...config_loader import load_config
    from ...training import train_g2p_from_config

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("phonebox.train")

    try:
        config = load_config(args.config) if args.config else {}
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    overrides = {
        "locale": args.locale,
        "dictionary": args.lexicon,
        "output": args.output,
        "phoneset": args.phoneset,
        "alignments_out": args.alignments_out,
        "max_combinations": args.max_combinations,
        "prune": args.prune,
        "validation_split": args.validation_split,
        "test_split": args.test_split,
        "trainer": args.trainer,
        "parallel_align": args.parallel_align,
        "width": args.width,
        "store_distributions": args.store_distributions,
        "remove_stress": args.remove_stress,
        "cased": args.cased,
        "norm_xlit": args.norm_xlit,
        "max_iterations": args.max_iterations,
        "verbose": args.verbose,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if not config.get("locale"):
        print("Error: no locale specified (use --locale or --config)", file=sys.stderr)
        return 2
    lexicon = config.get("dictionary")
    if not lexicon:
        print(
            "Error: no lexicon specified (use --lexicon or --config)", file=sys.stderr
        )
        return 2
    lex = Path(lexicon)
    if not lex.is_file():
        print(f"Error: lexicon not found: {lex}", file=sys.stderr)
        return 2
    if not config.get("output"):
        print("Error: no output specified (use --output or --config)", file=sys.stderr)
        return 2

    try:
        result = train_g2p_from_config(config)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    log.info("Training metrics: %s", result.metrics)
    log.info("Model: %s", result.output_path)
    log.info("Alignments: %s", result.alignments_path)
    return 0
