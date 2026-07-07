#!/usr/bin/env python
"""phonebox compare — 1:1 vs MultigramG2P evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

from phonebox.constants import (
    DEFAULT_MAX_TEST_ENTRIES,
    DEFAULT_MULTIGRAM_PHONESET,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
)
from phonebox.eval.g2p_compare import print_results_table, run_compare
from phonebox.eval.g2p_compare_all import run_compare_all
from phonebox.experiments.equiv import equiv_for_locale


def setup_compare_commands(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Compare 1:1 G2PDecisionTree vs MultigramG2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Evaluate 1:1 vs n:m on a held-out lexicon slice. "
            "Subcommands: ``locale`` (one lexicon) or ``all`` (six IPA locales). "
            "See docs/G2P_EVAL.md."
        ),
    )
    sp = parser.add_subparsers(dest="compare_mode", required=True)

    all_p = sp.add_parser(
        "all",
        help="All six IPA locales (writes docs/G2P_COMPARE*.md)",
    )
    all_p.add_argument("--lexicon-dir", type=Path, default=None)
    all_p.add_argument("--g2p-dir", type=Path, default=None)
    all_p.add_argument("--output", type=Path, default=None)
    all_p.add_argument("--no-config-joins", action="store_true")
    all_p.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    all_p.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST_ENTRIES)
    all_p.add_argument("--em-iterations", type=int, default=15)
    all_p.add_argument("--parallel-align", action="store_true")
    all_p.add_argument("--use-exceptions", action="store_true")
    all_p.add_argument("--locales", nargs="*")
    all_p.set_defaults(func=handle_compare_all)

    loc_p = sp.add_parser("locale", help="Single locale")
    loc_p.add_argument("--lexicon", required=True, type=Path)
    loc_p.add_argument("--locale", required=True)
    loc_p.add_argument("--phoneset", default=DEFAULT_MULTIGRAM_PHONESET)
    loc_p.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    loc_p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    loc_p.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST_ENTRIES)
    loc_p.add_argument("--max-letter-span", type=int, default=2)
    loc_p.add_argument("--max-phone-span", type=int, default=2)
    loc_p.add_argument("--em-iterations", type=int, default=15)
    loc_p.add_argument("--lm-order", type=int, default=2, choices=[1, 2, 3])
    loc_p.add_argument("--decode-beam", type=int, default=0)
    loc_p.add_argument("--parallel-align", action="store_true")
    loc_p.add_argument("--parallel-viterbi", action="store_true")
    loc_p.add_argument("-v", "--verbose", action="store_true")
    loc_p.add_argument("--relaxed-per", action="store_true")
    loc_p.add_argument("--vowel-equiv", action="store_true")
    loc_p.add_argument("--train-normalize", default=None)
    loc_p.add_argument("--skip-baseline", action="store_true")
    loc_p.add_argument("--skip-multigram", action="store_true")
    loc_p.add_argument("--baseline-model", type=Path, default=None)
    loc_p.add_argument("--use-exceptions", action="store_true")
    loc_p.add_argument("--no-config-joins", action="store_true")
    loc_p.set_defaults(func=handle_compare_locale)


def handle_compare_all(args) -> int:
    lex_dir = args.lexicon_dir or os.environ.get("PHONEDECODING_LEXICON_DIR")
    g2p_dir = args.g2p_dir or os.environ.get("PHONEDECODING_G2P_DIR")
    if not lex_dir:
        print("Set PHONEDECODING_LEXICON_DIR or pass --lexicon-dir", file=sys.stderr)
        return 2
    if not args.no_config_joins and not g2p_dir:
        print(
            "Set PHONEDECODING_G2P_DIR or pass --g2p-dir (not needed for --no-config-joins)",
            file=sys.stderr,
        )
        return 2
    output = args.output
    if output is None:
        output = (
            Path("docs/G2P_COMPARE_NO_JOINS.md")
            if args.no_config_joins
            else Path("docs/G2P_COMPARE.md")
        )
    return run_compare_all(
        lexicon_dir=Path(lex_dir),
        g2p_dir=Path(g2p_dir) if g2p_dir else None,
        output=output,
        no_config_joins=args.no_config_joins,
        seed=args.seed,
        max_test=args.max_test,
        em_iterations=args.em_iterations,
        parallel_align=args.parallel_align,
        use_exceptions=args.use_exceptions,
        locales=args.locales,
    )


def handle_compare_locale(args) -> int:
    try:
        summary = run_compare(
            lexicon=args.lexicon,
            locale=args.locale,
            phoneset=args.phoneset,
            seed=args.seed,
            test_fraction=args.test_fraction,
            max_test=args.max_test,
            max_letter_span=args.max_letter_span,
            max_phone_span=args.max_phone_span,
            em_iterations=args.em_iterations,
            lm_order=args.lm_order,
            decode_beam=args.decode_beam,
            parallel_align=args.parallel_align,
            parallel_viterbi=args.parallel_viterbi,
            verbose=args.verbose,
            phone_equiv=equiv_for_locale(args.locale)
            if args.relaxed_per or args.vowel_equiv
            else None,
            train_normalize_policy=args.train_normalize,
            no_config_joins=args.no_config_joins,
            skip_baseline=args.skip_baseline,
            skip_multigram=args.skip_multigram,
            baseline_model=args.baseline_model,
            use_exceptions=args.use_exceptions,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    rows = [
        (
            cast(str, r["model"]),
            cast(float, r["train_s"]),
            cast(
                dict[str, float],
                {k: r[k] for k in r if k not in ("model", "train_s")},
            ),
        )
        for r in cast(list[dict[str, object]], summary["results"])
    ]
    print_results_table(
        rows,
        show_relaxed_per=args.relaxed_per or args.vowel_equiv,
    )
    return 0
