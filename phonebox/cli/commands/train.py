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
import time
from pathlib import Path

from ...constants import DEFAULT_MAX_COMBINATIONS, DICT_ENCODING, FILE_ENCODING


def setup_train_command(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="End-to-end G2P training (native trainer, safe defaults)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--locale", required=True, help="Locale tag, e.g. fr_FR")
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Pronunciation dictionary TSV (word\\tphone phone ...)",
    )
    parser.add_argument("-o", "--output", required=True, help="Model output (.g2p.gz)")
    parser.add_argument(
        "--phoneset",
        default="ipa",
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
        default=DEFAULT_MAX_COMBINATIONS,
        help=f"Cap on alignment combinations per word "
        f"(default: {DEFAULT_MAX_COMBINATIONS}; set 0 to disable).",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        default=True,
        # `%%` because argparse treats `%` as a format specifier in help.
        help="Skip post-train pruning (default: prune with 5%% validation).",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.05,
        help="Fraction held out for pruning (default: 0.05).",
    )
    parser.add_argument(
        "--trainer",
        default="native",
        choices=["native", "sklearn"],
        help="Tree trainer backend (default: native — safer on big lexicons).",
    )
    parser.add_argument(
        "--parallel-align",
        action="store_true",
        help="Enable EMAlign multiprocessing pool. Off by default because "
        "the fork pool duplicates the lexicon per worker.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.set_defaults(func=handle_train)


def handle_train(args) -> int:
    # Late imports so `phonebox --help` stays fast.
    from ...core.g2p_model import G2PDecisionTree

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("phonebox.train")

    lex = Path(args.lexicon)
    if not lex.is_file():
        print(f"Error: lexicon not found: {lex}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    align_out = (
        Path(args.alignments_out)
        if args.alignments_out
        else (out.with_name(out.stem.replace(".g2p", "") + "_alignments.txt"))
    )

    t0 = time.time()

    def step(msg: str) -> None:
        log.info("[%7.1fs] %s", time.time() - t0, msg)

    step(
        f"init G2PDecisionTree(locale={args.locale!r}, phoneset={args.phoneset!r}, "
        f"trainer={args.trainer!r}, parallel_align={args.parallel_align}, "
        f"max_combinations={args.max_combinations})"
    )
    dt = G2PDecisionTree(
        locale=args.locale,
        phoneset_name=args.phoneset,
        remove_stress=False,
        verbose=True,
        trainer=args.trainer,
        parallel_align=args.parallel_align,
        max_combinations=args.max_combinations,
    )

    step(f"load_prondict {lex}")
    with lex.open(encoding=DICT_ENCODING) as f:
        dt.load_prondict(f)

    step("em.align")
    dt.em.align()

    step(f"write alignments -> {align_out}")
    with align_out.open("w", encoding=FILE_ENCODING) as f:
        dt.em.write(f)

    step("load_alignments (convert to feature vectors)")
    dt.load_alignments()

    step(f"train (prune={args.prune}, validation_split={args.validation_split})")
    metrics = dt.train(
        prune=args.prune,
        validation_split=args.validation_split if args.prune else 0.0,
    )
    step(f"train metrics: {metrics}")

    step(f"export -> {out}")
    dt.export(str(out))

    step("done")
    return 0
