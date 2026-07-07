#!/usr/bin/env python
"""Sweep ``max_letter_span`` × ``lm_order`` for n:m on selected locales.

Trains MultigramG2P once per (locale, span, order) combo on the same held-out
train slice (seed-aligned with compare_g2p_all), scores against the test slice,
writes a markdown sweep table.

Example::

    export PHONEDECODING_LEXICON_DIR=…
    python compare_g2p_sweep.py \
        --locales fr_FR de_DE pt_BR en_US \
        --letter-spans 2 3 --lm-orders 2 3 --parallel-align
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import (
    build_gold_variants,
    cook_pair,
    evaluate,
    load_lexicon,
    train_multigram,
)
from phonebox.experiments.equiv import equiv_for_locale

_LOCALES = [
    ("es_MX", "es_ipa.tsv", False),
    ("fr_FR", "fr_ipa.tsv", False),
    ("de_DE", "de_ipa.tsv", False),
    ("en_US", "en_ipa.tsv", False),
    ("pt_BR", "pt_ipa.tsv", False),
    ("it_IT", "it_ipa.tsv", True),
]


def _prep(locale: str, lex_path: Path, seed: int, max_test: int):
    pairs = load_lexicon(lex_path)
    vec = Vectorizer(locale=locale, phoneset_name="ipa")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_test = min(max_test, max(1, int(len(pairs) * 0.10)))
    test_raw = pairs[:n_test]
    train_raw = pairs[n_test:]
    train_cooked = []
    test_eval = []
    for w, ph in test_raw:
        cooked = cook_pair(vec, w, ph)
        if cooked:
            test_eval.append((w, cooked[1]))
    for w, ph in train_raw:
        cooked = cook_pair(vec, w, ph)
        if cooked:
            train_cooked.append(cooked)
    return vec, train_cooked, test_eval, build_gold_variants(pairs, vec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexicon-dir", type=Path, default=None)
    ap.add_argument(
        "--locales", nargs="*", default=["fr_FR", "de_DE", "pt_BR", "en_US"]
    )
    ap.add_argument("--letter-spans", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--lm-orders", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--em-iterations", type=int, default=15)
    ap.add_argument("--parallel-align", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("docs/G2P_SWEEP.md"))
    args = ap.parse_args()

    lex_dir = args.lexicon_dir or os.environ.get("PHONEDECODING_LEXICON_DIR")
    if not lex_dir:
        print("Set PHONEDECODING_LEXICON_DIR or --lexicon-dir", file=sys.stderr)
        return 2
    lex_root = Path(lex_dir)
    selected = set(args.locales)
    locales = [(loc, name, ve) for loc, name, ve in _LOCALES if loc in selected]

    lines = [
        "# n:m sweep: letter span × LM order",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Setup",
        "",
        f"- Seed: {args.seed}, max test: {args.max_test}",
        f"- EM iterations: {args.em_iterations}, parallel align: {args.parallel_align}",
        "- LM: stdlib add-k (k=0.1) n-gram",
        f"- Locales: {', '.join(loc for loc, _, _ in locales)}",
        f"- Letter spans: {args.letter_spans}",
        f"- LM orders: {args.lm_orders}",
        "",
        "Each cell shows ``WER% / PER%`` (lower is better). PER is primary.",
        "",
    ]

    rows: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    t_all = time.time()

    for locale, lex_name, vowel_equiv in locales:
        lex_path = lex_root / lex_name
        print(f"\n=== {locale} ({lex_name}) ===", flush=True)
        vec, train_cooked, test_eval, gold_variants = _prep(
            locale, lex_path, args.seed, args.max_test
        )
        rows[locale] = {}
        for span in args.letter_spans:
            for order in args.lm_orders:
                t0 = time.time()
                print(f"  span={span} order={order} … training", flush=True)
                mg = train_multigram(
                    train_cooked,
                    span,
                    2,
                    args.em_iterations,
                    parallel_align=args.parallel_align,
                    parallel_viterbi=args.parallel_align,
                    lm_order=order,
                )

                def mg_predict(word: str, _mg=mg, _vec=vec) -> list[str]:
                    letters = _vec.cook_letters(word, g2p=True)
                    pred = _mg.pronounce_letters(letters, word=word)
                    cooked = _vec.cook_phones(pred)
                    return cooked if cooked else pred

                m = evaluate(
                    "n:m",
                    mg_predict,
                    test_eval,
                    gold_variants=gold_variants,
                    phone_equiv=equiv_for_locale(locale) if vowel_equiv else None,
                )
                m["train_s"] = time.time() - t0
                rows[locale][(span, order)] = m
                print(
                    f"    WER={m['wer_pct']:.2f}  PER={m['per_pct']:.2f}  "
                    f"({m['train_s']:.0f}s)",
                    flush=True,
                )

    for locale, _lex_name, _ in locales:
        if locale not in rows:
            continue
        lines.extend([f"## {locale}", ""])
        hdr_cells = ["letter_span \\ lm_order", *[str(o) for o in args.lm_orders]]
        lines.append("| " + " | ".join(hdr_cells) + " |")
        lines.append("|" + "---|" * len(hdr_cells))
        for span in args.letter_spans:
            cells = [str(span)]
            for order in args.lm_orders:
                m = rows[locale].get((span, order))
                if not m:
                    cells.append("—")
                    continue
                cells.append(f"{m['wer_pct']:.2f} / {m['per_pct']:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output} ({time.time() - t_all:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
