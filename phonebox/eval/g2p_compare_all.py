#!/usr/bin/env python
"""Run 1:1 (pretrained) vs n:m (train split) G2P comparison for all IPA locales.

Also invoked as ``phonebox compare all``. Loads existing ``*-ipa.g2p.gz``
decision-tree models (tree only — embedded lexicon exceptions are disabled) and
trains MultigramG2P on the held-out train slice. Writes ``docs/G2P_COMPARE.md``.

Use ``--use-exceptions`` for hybrid eval (train-split lookup on both 1:1 and n:m).

Requires environment variables (no hard-coded paths outside this repo):

  PHONEDECODING_LEXICON_DIR  directory with ``es_ipa.tsv``, ``fr_ipa.tsv``, …
  PHONEDECODING_G2P_DIR      parent of ``es-mx/``, ``fr-fr/``, … model dirs

Example::

    export PHONEDECODING_LEXICON_DIR=/path/to/lexicons/processed
    export PHONEDECODING_G2P_DIR=/path/to/build/g2p
    python compare_g2p_all.py --parallel-align

    # Fair join-off baseline (both models train split; no G2P_DIR):
    python compare_g2p_all.py --no-config-joins --parallel-align

See ``docs/G2P_EVAL.md`` for metrics and flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from phonebox.constants import (
    DEFAULT_MAX_TEST_ENTRIES,
    DEFAULT_SPLIT_SEED,
    FILE_ENCODING,
)
from phonebox.eval.g2p_compare import run_compare
from phonebox.experiments.equiv import equiv_for_locale
from phonebox.experiments.metrics import G2P_METRICS_FOOTER

_LOCALES: list[tuple[str, str, str]] = [
    ("es_MX", "es_ipa.tsv", "es-mx/es-mx-ipa.g2p.gz"),
    ("fr_FR", "fr_ipa.tsv", "fr-fr/fr-fr-ipa.g2p.gz"),
    ("de_DE", "de_ipa.tsv", "de-de/de-de-ipa.g2p.gz"),
    ("en_US", "en_ipa.tsv", "en-us/en-us-ipa.g2p.gz"),
    ("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz"),
    ("it_IT", "it_ipa.tsv", "it-it/it-it-ipa.g2p.gz"),
]


def _metric_row(
    model: str, train_s: float, m: dict[str, float], relaxed_per: bool
) -> str:
    cells = [
        model,
        f"{train_s:.1f}",
        f"{m['wer_pct']:.2f}",
        f"{m['wer_relaxed_pct']:.2f}",
        f"{m['per_pct']:.2f}",
    ]
    if relaxed_per:
        cells.append(f"{m['per_equiv_pct']:.2f}")
    cells.append(f"{m['pos_acc_pct']:.2f}")
    return "| " + " | ".join(cells) + " |"


def _write_markdown(
    path: Path, summaries: list[dict[str, object]], *, args: argparse.Namespace
) -> None:
    no_joins = args.no_config_joins
    title = (
        "# G2P comparison: 1:1 vs n:m (no config joins)"
        if no_joins
        else "# G2P comparison: 1:1 vs n:m (MultigramG2P)"
    )
    if no_joins:
        one_one_line = "- 1:1: **G2PDecisionTree trained on train split** (same xlit, no ``config.json`` joins)"
        joins_line = (
            "- Locale ``config.json`` letter/phone joins: **off** "
            "(multigram units from EM only; xlit still on)"
        )
        baseline_note = (
            "Pretrained ``*-ipa.g2p.gz`` models are **not** used (they were trained with joins). "
            "Compare to [`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md) for join-on + pretrained 1:1."
        )
    else:
        one_one_line = (
            "- 1:1: pretrained ``*-ipa.g2p.gz`` (decision tree only at eval time)"
        )
        joins_line = "- Locale ``config.json`` joins: **on** (via ``Vectorizer``)"
        baseline_note = "Embedded full-lexicon exceptions in the 1:1 ``.g2p.gz`` files are **not** used."
    lines = [
        title,
        "",
    ]
    if no_joins:
        lines.extend(
            [
                "> Join-off fair compare. Frozen join-on baseline: "
                "[`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md). "
                "Eval guide: [`G2P_EVAL.md`](G2P_EVAL.md).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> Baseline snapshot: [`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md). "
                "Join-off compare: [`G2P_COMPARE_NO_JOINS.md`](G2P_COMPARE_NO_JOINS.md). "
                "Eval guide: [`G2P_EVAL.md`](G2P_EVAL.md).",
                "",
            ]
        )
    lines.extend(
        [
            f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Setup",
            "",
            f"- Seed: {args.seed}",
            f"- Test cap: {args.max_test} entries (10% split, shuffled)",
            one_one_line,
            f"- n:m: MultigramG2P v3 joint decode + LM, EM iterations={args.em_iterations}",
            joins_line,
            f"- Parallel align: {args.parallel_align}",
            f"- Exceptions / lexicon lookup: {'train split only' if args.use_exceptions else 'off (pure G2P)'}",
            "",
            baseline_note,
            "Multigram trains on the train split only; test words are never in its exception table.",
            "",
            "WERr = any cooked pronunciation variant in the lexicon counts as correct.",
            "",
        ]
    )
    if not no_joins:
        lines.extend(
            [
                "n:m uses ``max_letter_span=3`` for fr_FR and de_DE (locale "
                "``config.json``); other locales default to 2.",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            "| Locale | 1:1 WER% | 1:1 PER% | n:m WER% | n:m PER% | Δ PER (1:1−n:m) | Winner |",
            "|--------|----------|----------|----------|----------|-----------------|--------|",
        ]
    )
    for summary in summaries:
        by_name = {
            r["model"]: r for r in cast(list[dict[str, object]], summary["results"])
        }
        one = cast(dict[str, float], by_name["G2PDecisionTree"])
        mg = cast(dict[str, float], by_name["MultigramG2P"])
        delta = one["per_pct"] - mg["per_pct"]
        winner = "n:m" if delta > 0 else "1:1"
        lines.append(
            f"| {summary['locale']} | {one['wer_pct']:.2f} | {one['per_pct']:.2f} | "
            f"{mg['wer_pct']:.2f} | {mg['per_pct']:.2f} | {delta:+.2f} | {winner} |"
        )

    for summary in summaries:
        locale = summary["locale"]
        relaxed_per = bool(summary.get("phone_equiv"))
        lines.extend(
            [
                "",
                f"## {locale}",
                "",
                f"- Lexicon: `{summary['lexicon']}` ({summary['n_entries']} entries, "
                f"{summary['n_test']} test)",
                f"- Multi-pron words: {summary['n_multi_pron']}",
                f"- 1:1 model: `{summary.get('baseline_model_rel') or 'train-split G2PDecisionTree'}`",
                f"- Config joins: {'off' if summary.get('no_config_joins') else 'on'}",
                "",
            ]
        )
        hdr = "| Model | train_s | WER% | WERr% | PER% |"
        if relaxed_per:
            hdr += " PERr% |"
        hdr += " pos% |"
        lines.append(hdr)
        lines.append("|" + "---|" * (hdr.count("|") - 1))
        for row in cast(list[dict[str, object]], summary["results"]):
            metrics = cast(dict[str, float], row)
            lines.append(
                _metric_row(
                    cast(str, row["model"]),
                    cast(float, row["train_s"]),
                    metrics,
                    relaxed_per,
                )
            )

    lines.extend(["", G2P_METRICS_FOOTER, ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding=FILE_ENCODING)


def run_compare_all(
    *,
    lexicon_dir: Path,
    g2p_dir: Path | None,
    output: Path,
    no_config_joins: bool = False,
    seed: int = DEFAULT_SPLIT_SEED,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
    em_iterations: int = 15,
    parallel_align: bool = False,
    use_exceptions: bool = False,
    locales: list[str] | None = None,
) -> int:
    """Run six-locale comparison; write markdown to ``output``."""
    lex_root = lexicon_dir
    g2p_root = g2p_dir
    if not lex_root.is_dir():
        print(f"Invalid lexicon dir: {lex_root}", file=sys.stderr)
        return 2
    if not no_config_joins and (g2p_root is None or not g2p_root.is_dir()):
        print(f"Invalid g2p dir: {g2p_root}", file=sys.stderr)
        return 2

    selected = set(locales) if locales else None
    summaries: list[dict[str, object]] = []
    t_all = time.time()

    for locale, lex_name, model_rel in _LOCALES:
        if selected is not None and locale not in selected:
            continue
        lexicon = lex_root / lex_name
        model_path = None if no_config_joins else g2p_root / model_rel
        print(f"\n=== {locale} ===", flush=True)
        summary = run_compare(
            lexicon=lexicon,
            locale=locale,
            seed=seed,
            max_test=max_test,
            em_iterations=em_iterations,
            parallel_align=parallel_align,
            phone_equiv=equiv_for_locale(locale),
            no_config_joins=no_config_joins,
            baseline_model=model_path,
            skip_multigram=False,
            use_exceptions=use_exceptions,
        )
        summary["baseline_model_rel"] = None if no_config_joins else model_rel
        summaries.append(summary)

    if not summaries:
        print("No locales ran.", file=sys.stderr)
        return 1

    ns = argparse.Namespace(
        no_config_joins=no_config_joins,
        seed=seed,
        max_test=max_test,
        em_iterations=em_iterations,
        parallel_align=parallel_align,
        use_exceptions=use_exceptions,
    )

    _write_markdown(output, summaries, args=ns)
    print(f"\nWrote {output} ({time.time() - t_all:.0f}s total)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lexicon-dir",
        type=Path,
        default=None,
        help="Override PHONEDECODING_LEXICON_DIR",
    )
    ap.add_argument(
        "--g2p-dir",
        type=Path,
        default=None,
        help="Override PHONEDECODING_G2P_DIR",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--no-config-joins",
        action="store_true",
        help="Disable locale joins; train 1:1 on train split (fair vs n:m). Writes G2P_COMPARE_NO_JOINS.md.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST_ENTRIES)
    ap.add_argument("--em-iterations", type=int, default=15)
    ap.add_argument("--parallel-align", action="store_true")
    ap.add_argument(
        "--use-exceptions",
        action="store_true",
        help="Train-split hybrid lookup for both 1:1 and n:m (default: pure G2P).",
    )
    ap.add_argument(
        "--locales", nargs="*", help="Subset of locale tags, e.g. es_MX it_IT"
    )
    args = ap.parse_args()
    if args.output is None:
        args.output = (
            Path("docs/G2P_COMPARE_NO_JOINS.md")
            if args.no_config_joins
            else Path("docs/G2P_COMPARE.md")
        )

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

    return run_compare_all(
        lexicon_dir=Path(lex_dir),
        g2p_dir=Path(g2p_dir) if g2p_dir else None,
        output=args.output,
        no_config_joins=args.no_config_joins,
        seed=args.seed,
        max_test=args.max_test,
        em_iterations=args.em_iterations,
        parallel_align=args.parallel_align,
        use_exceptions=args.use_exceptions,
        locales=args.locales,
    )


if __name__ == "__main__":
    raise SystemExit(main())
