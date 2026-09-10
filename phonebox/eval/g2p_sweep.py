"""Programmatic multigram hyperparameter sweeps."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import (
    build_gold_variants,
    cook_pair,
    evaluate,
    load_lexicon,
    train_multigram,
)
from phonebox.eval.locale_registry import canonical_locales, select_locale_paths
from phonebox.experiments.equiv import equiv_for_locale
from phonebox.experiments.split import split_lexicon


def prepare_sweep_data(locale: str, lexicon: Path, *, seed: int, max_test: int):
    """Cook a deterministic train/test split for a sweep."""
    pairs = load_lexicon(lexicon)
    vec = Vectorizer(locale=locale, phoneset_name="ipa")
    test_raw, train_raw = split_lexicon(pairs, seed=seed, max_test=max_test)
    test_eval = [
        (word, cooked[1])
        for word, phones in test_raw
        if (cooked := cook_pair(vec, word, phones))
    ]
    train = [c for w, p in train_raw if (c := cook_pair(vec, w, p))]
    return vec, train, test_eval, build_gold_variants(pairs, vec)


def run_g2p_sweep(
    lexicons: Mapping[str, Path],
    *,
    locales: list[str],
    letter_spans: list[int],
    lm_orders: list[int],
    seed: int = 42,
    max_test: int = 2000,
    em_iterations: int = 15,
    parallel_align: bool = False,
    max_phone_span: int = 2,
    relaxed_locales: frozenset[str] = frozenset(),
) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    """Train and evaluate every requested locale/span/order combination."""
    lexicons = select_locale_paths(lexicons, locales)
    locales = list(lexicons)
    relaxed_locales = frozenset(canonical_locales(list(relaxed_locales)))
    rows: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for locale in locales:
        vec, train, test, gold = prepare_sweep_data(
            locale, Path(lexicons[locale]), seed=seed, max_test=max_test
        )
        rows[locale] = {}
        for span in letter_spans:
            for order in lm_orders:
                started = time.time()
                model = train_multigram(
                    train,
                    span,
                    max_phone_span,
                    em_iterations,
                    parallel_align=parallel_align,
                    parallel_viterbi=parallel_align,
                    lm_order=order,
                )

                def predict(word: str, _model=model, _vec=vec) -> list[str]:
                    phones = _model.pronounce_letters(
                        _vec.cook_letters(word, g2p=True), word=word
                    )
                    return _vec.cook_phones(phones) or phones

                metrics = evaluate(
                    "n:m",
                    predict,
                    test,
                    gold_variants=gold,
                    phone_equiv=equiv_for_locale(locale)
                    if locale in relaxed_locales
                    else None,
                )
                metrics["train_s"] = time.time() - started
                rows[locale][(span, order)] = metrics
    return rows


def format_g2p_sweep(
    rows: dict[str, dict[tuple[int, int], dict[str, float]]],
    *,
    letter_spans: list[int],
    lm_orders: list[int],
    seed: int = 42,
    max_test: int = 2000,
    em_iterations: int = 15,
    parallel_align: bool = False,
) -> str:
    """Render sweep metrics as a compact Markdown report."""
    lines = [
        "# n:m sweep: letter span × LM order",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Setup",
        "",
        f"- Seed: {seed}, max test: {max_test}",
        f"- EM iterations: {em_iterations}, parallel align: {parallel_align}",
        "- LM: stdlib add-k (k=0.1) n-gram",
        f"- Locales: {', '.join(rows)}",
        f"- Letter spans: {letter_spans}",
        f"- LM orders: {lm_orders}",
        "",
        "Each cell shows ``WER% / PER%`` (lower is better). PER is primary.",
        "",
    ]
    for locale, values in rows.items():
        lines.extend([f"## {locale}", ""])
        header = ["letter_span \\ lm_order", *map(str, lm_orders)]
        lines.extend(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
        for span in letter_spans:
            cells = [str(span)]
            for order in lm_orders:
                value = values.get((span, order))
                cells.append(
                    "—"
                    if value is None
                    else f"{value['wer_pct']:.2f} / {value['per_pct']:.2f}"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "format_g2p_sweep",
    "prepare_sweep_data",
    "run_g2p_sweep",
]
