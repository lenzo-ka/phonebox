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
from phonebox.eval.g2p_compare_all import (
    CompareAllConfig,
    run_compare_all,
    write_compare_all,
)
from phonebox.eval.locale_registry import (
    EVALUATION_LOCALES,
    canonical_locale_paths,
    canonical_locales,
    evaluation_locale,
)
from phonebox.experiments.equiv import equiv_for_locale
from phonebox.locale_resolution import canonical_locale


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

    sweep = sp.add_parser("sweep", help="Sweep multigram span and LM order")
    sweep.add_argument(
        "--lexicon-dir",
        type=Path,
        default=None,
        help="Directory using curated lexicon filenames",
    )
    sweep.add_argument(
        "--lexicon",
        action="append",
        default=[],
        metavar="LOCALE=PATH",
        help="Explicit locale lexicon; repeat for multiple locales",
    )
    sweep.add_argument("--locales", nargs="*", default=None)
    sweep.add_argument(
        "--letter-spans",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Letter spans to evaluate",
    )
    sweep.add_argument(
        "--lm-orders",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Language-model orders to evaluate",
    )
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--max-test", type=int, default=2000)
    sweep.add_argument("--em-iterations", type=int, default=15)
    sweep.add_argument("--parallel-align", action="store_true")
    sweep.add_argument(
        "--output",
        type=Path,
        default=Path("docs/G2P_SWEEP.md"),
        help="Markdown output path",
    )
    sweep.set_defaults(func=handle_compare_sweep)

    units = sp.add_parser("units", help="Inspect learned multigram units")
    units.add_argument(
        "--lexicon-dir",
        type=Path,
        default=None,
        help="Directory using curated lexicon filenames",
    )
    units.add_argument(
        "--lexicon",
        action="append",
        default=[],
        metavar="LOCALE=PATH",
        help="Explicit locale lexicon; repeat for multiple locales",
    )
    units.add_argument("--locales", nargs="*", default=None)
    units.add_argument("--top", type=int, default=25)
    units.add_argument("--em-iterations", type=int, default=15)
    units.add_argument("--seed", type=int, default=42)
    units.add_argument("--max-test", type=int, default=2000)
    units.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output; always prints to stdout",
    )
    units.set_defaults(func=handle_compare_units)

    accuracy = sp.add_parser("accuracy", help="Train/test dictionary accuracy")
    accuracy.add_argument(
        "dictionary",
        type=Path,
        help="Pronunciation dictionary to split, train, and evaluate",
    )
    accuracy.add_argument("--locale", default="en_US")
    accuracy.add_argument("--phoneset", default="cmu")
    accuracy.add_argument("--train-fraction", type=float, default=0.95)
    accuracy.add_argument("--seed", type=int, default=42)
    accuracy.add_argument("--width", type=int, default=None)
    accuracy.add_argument("--parallel-align", action="store_true")
    accuracy.set_defaults(func=handle_compare_accuracy)

    experiments = sp.add_parser("experiments", help="Run normalization experiments")
    experiments.add_argument(
        "--lexicon-dir",
        type=Path,
        default=None,
        help="Directory using curated lexicon filenames",
    )
    experiments.add_argument(
        "--g2p-dir",
        type=Path,
        default=None,
        help="Directory using curated baseline-model filenames",
    )
    experiments.add_argument(
        "--output-dir", type=Path, default=Path("docs/experiments")
    )
    experiments.add_argument("--seed", type=int, default=42)
    experiments.add_argument("--max-test", type=int, default=2000)
    experiments.add_argument("--em-iterations", type=int, default=15)
    experiments.add_argument("--parallel-align", action="store_true")
    experiments.add_argument("--skip-error-analysis", action="store_true")
    experiments.add_argument("--locales", nargs="*")
    experiments.add_argument("--policies", nargs="*")
    experiments.add_argument(
        "--experiment",
        action="append",
        nargs=4,
        metavar=("LOCALE", "POLICY", "LEXICON", "MODEL"),
        help="Explicit experiment specification; repeat for multiple runs",
    )
    experiments.set_defaults(func=handle_compare_experiments)


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
    try:
        locales = canonical_locales(args.locales or list(EVALUATION_LOCALES))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        lexicons = _curated_paths(Path(lex_dir), locales)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    models = (
        None
        if args.no_config_joins
        else {
            locale: Path(g2p_dir) / evaluation_locale(locale).baseline_model
            for locale in locales
        }
    )
    try:
        summaries = run_compare_all(
            lexicons=lexicons,
            baseline_models=models,
            no_config_joins=args.no_config_joins,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
            use_exceptions=args.use_exceptions,
            locales=locales,
            quiet=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    write_compare_all(
        output,
        summaries,
        config=CompareAllConfig(
            no_config_joins=args.no_config_joins,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
            use_exceptions=args.use_exceptions,
        ),
    )
    print(f"Wrote {output}")
    return 0


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
    except (FileNotFoundError, ValueError) as exc:
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


def _required_dir(value, environment: str, option: str) -> Path | None:
    resolved = value or os.environ.get(environment)
    if resolved:
        return Path(resolved)
    print(f"Set {environment} or pass {option}", file=sys.stderr)
    return None


def _locale_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        locale, separator, path = value.partition("=")
        if not separator or not locale or not path:
            raise ValueError(f"invalid locale path {value!r}; expected LOCALE=PATH")
        paths[locale] = Path(path)
    return canonical_locale_paths(paths)


def _curated_paths(root: Path, locales: list[str]) -> dict[str, Path]:
    locales = canonical_locales(locales)
    return {locale: root / evaluation_locale(locale).lexicon_name for locale in locales}


def handle_compare_sweep(args) -> int:
    from phonebox.eval.g2p_sweep import format_g2p_sweep, run_g2p_sweep

    try:
        paths = _locale_paths(args.lexicon)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    locales = args.locales or list(paths) or ["fr_FR", "de_DE", "pt_BR", "en_US"]
    if not paths:
        lexicons = _required_dir(
            args.lexicon_dir, "PHONEDECODING_LEXICON_DIR", "--lexicon-dir"
        )
        if lexicons is None:
            return 2
        try:
            paths = _curated_paths(lexicons, locales)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    try:
        rows = run_g2p_sweep(
            paths,
            locales=locales,
            letter_spans=args.letter_spans,
            lm_orders=args.lm_orders,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
            relaxed_locales=frozenset(
                locale
                for locale in locales
                if evaluation_locale(locale).sweep_relaxed_per
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        format_g2p_sweep(
            rows,
            letter_spans=args.letter_spans,
            lm_orders=args.lm_orders,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
        ),
        encoding="utf-8",
    )
    return 0


def handle_compare_units(args) -> int:
    from phonebox.eval.multigram_units import (
        analyze_multigram_units,
        format_multigram_units,
    )

    try:
        paths = _locale_paths(args.lexicon)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        locales = canonical_locales(
            args.locales or list(paths) or list(EVALUATION_LOCALES)
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not paths:
        lexicons = _required_dir(
            args.lexicon_dir, "PHONEDECODING_LEXICON_DIR", "--lexicon-dir"
        )
        if lexicons is None:
            return 2
        try:
            paths = _curated_paths(lexicons, locales)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    try:
        from phonebox.eval.locale_registry import select_locale_paths

        selected_paths = select_locale_paths(paths, locales)
        text = "\n".join(
            format_multigram_units(
                analyze_multigram_units(
                    locale,
                    selected_paths[locale],
                    top=args.top,
                    em_iterations=args.em_iterations,
                    seed=args.seed,
                    max_test=args.max_test,
                )
            )
            for locale in selected_paths
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "# Top multigram units per locale\n\n```\n" + text + "```\n",
            encoding="utf-8",
        )
    return 0


def handle_compare_accuracy(args) -> int:
    from phonebox.eval.accuracy import evaluate_accuracy, load_pronunciation_entries

    try:
        entries = load_pronunciation_entries(args.dictionary)
        print(f"Loaded {len(entries):,} entries (excluding alternates)")
        result = evaluate_accuracy(
            entries,
            locale=args.locale,
            phoneset=args.phoneset,
            train_fraction=args.train_fraction,
            seed=args.seed,
            width=args.width,
            parallel_align=args.parallel_align,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Train: {result.training_entries:,}, Test: {result.test_entries:,}")
    print(f"Word accuracy: {result.word_accuracy:.1f}%")
    print(f"Phone accuracy: {result.phone_accuracy:.1f}%")
    return 0


def handle_compare_experiments(args) -> int:
    from phonebox.eval.experiments import ExperimentSpec, run_experiments

    try:
        if args.experiment:
            specs = [
                ExperimentSpec(
                    canonical_locale(locale), Path(lexicon), Path(model), policy
                )
                for locale, policy, lexicon, model in args.experiment
            ]
        else:
            lexicons = _required_dir(
                args.lexicon_dir, "PHONEDECODING_LEXICON_DIR", "--lexicon-dir"
            )
            models = _required_dir(args.g2p_dir, "PHONEDECODING_G2P_DIR", "--g2p-dir")
            if lexicons is None or models is None:
                return 2
            definitions = [
                (
                    "it_IT",
                    ("baseline", "spelling_gated", "collapse_open"),
                ),
                (
                    "pt_BR",
                    ("baseline", "surface_final", "do_du", "citation_expand"),
                ),
            ]
            specs = [
                ExperimentSpec(
                    locale,
                    lexicons / evaluation_locale(locale).lexicon_name,
                    models / evaluation_locale(locale).baseline_model,
                    policy,
                )
                for locale, policies in definitions
                for policy in policies
            ]
        run_experiments(
            specs,
            output_dir=args.output_dir,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
            skip_error_analysis=args.skip_error_analysis,
            locales=args.locales,
            policies=args.policies,
            quiet=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0
