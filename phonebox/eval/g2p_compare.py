#!/usr/bin/env python
"""Compare 1:1 G2PDecisionTree vs MultigramG2P on a held-out lexicon slice.

Library module; also exposed as ``phonebox compare`` and ``compare_g2p.py``.
See ``docs/G2P_EVAL.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import cast

from phonebox.constants import (
    DEFAULT_MAX_TEST_ENTRIES,
    DEFAULT_MULTIGRAM_PHONESET,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    DICT_ENCODING,
)
from phonebox.core.vectorizer import Vectorizer
from phonebox.dictionary import parse_dict_line
from phonebox.experiments.equiv import equiv_for_locale
from phonebox.experiments.normalize import apply_train_normalize
from phonebox.experiments.split import split_lexicon

PolicyFn = Callable[[str, list[str]], list[str]]


def load_lexicon(path: Path) -> list[tuple[str, list[str]]]:
    pairs: list[tuple[str, list[str]]] = []
    with path.open(encoding=DICT_ENCODING) as f:
        for line in f:
            parsed = parse_dict_line(line)
            if parsed:
                word, phones = parsed
                if word and phones:
                    pairs.append((word, phones))
    return pairs


def cook_pair(
    vec: Vectorizer, word: str, phones: list[str]
) -> tuple[list[str], list[str]] | None:
    letters = vec.cook_letters(word, g2p=True)
    cooked_phones = vec.cook_phones(phones)
    if letters and cooked_phones:
        return letters, cooked_phones
    return None


def build_train_exceptions(
    train_raw: list[tuple[str, list[str]]],
    vec: Vectorizer,
) -> dict[str, list[str]]:
    """Cooked-phone exceptions from the train split only (for hybrid G2P eval)."""
    out: dict[str, list[str]] = {}
    for word, phones in train_raw:
        cooked = vec.cook_phones(phones)
        if cooked:
            key = word if vec.cased else word.lower()
            out[key] = cooked
    return out


def predict_cooked_phones(
    vec: Vectorizer, predict: Callable[[str], list[str]]
) -> Callable[[str], list[str]]:
    """Wrap a predictor so outputs use the same phone cooking as test gold."""

    def wrapped(word: str) -> list[str]:
        pred = predict(word)
        if not pred:
            return []
        cooked = vec.cook_phones(pred)
        return cooked if cooked else pred

    return wrapped


def build_gold_variants(
    pairs: list[tuple[str, list[str]]],
    vec: Vectorizer,
) -> dict[str, set[tuple[str, ...]]]:
    """All cooked-phone variants per orthographic word in the lexicon."""
    out: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for word, phones in pairs:
        cooked = vec.cook_phones(phones)
        if cooked:
            out[word].add(tuple(cooked))
    return dict(out)


def _resolve_phone_equiv(
    locale: str,
    *,
    vowel_equiv: bool,
    phone_equiv: frozenset[tuple[str, str]] | None,
) -> frozenset[tuple[str, str]] | None:
    if phone_equiv is not None:
        return phone_equiv
    if vowel_equiv:
        return equiv_for_locale("it_IT")
    return equiv_for_locale(locale)


def _phone_eq(a: str, b: str, equiv: frozenset[tuple[str, str]] | None) -> bool:
    return a == b or (equiv is not None and (a, b) in equiv)


def phones_match(
    pred: list[str],
    expected: list[str],
    gold: set[tuple[str, ...]] | None,
    phone_equiv: frozenset[tuple[str, str]] | None,
) -> bool:
    if pred == expected:
        return True
    if gold is not None and tuple(pred) in gold:
        return True
    if phone_equiv is None or len(pred) != len(expected):
        return False
    return all(_phone_eq(a, b, phone_equiv) for a, b in zip(pred, expected))


def edit_distance(
    a: list[str],
    b: list[str],
    phone_equiv: frozenset[tuple[str, str]] | None = None,
) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] if _phone_eq(ca, cb, phone_equiv) else prev[j - 1] + 1
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def evaluate(
    name: str,
    predict,
    test_set: list[tuple[str, list[str]]],
    gold_variants: dict[str, set[tuple[str, ...]]] | None = None,
    phone_equiv: frozenset[tuple[str, str]] | None = None,
) -> dict[str, float]:
    word_ok = word_ok_relaxed = 0
    phone_ok = 0
    phone_den = 0
    edit_sum = edit_sum_equiv = 0
    for word, expected in test_set:
        try:
            pred = predict(word)
        except Exception as exc:
            print(f"  [{name}] {word!r}: {exc}", file=sys.stderr)
            pred = []
        gold = gold_variants.get(word) if gold_variants else None
        if pred == expected:
            word_ok += 1
        if phones_match(pred, expected, gold, phone_equiv=None) or (
            phone_equiv is not None
            and phones_match(pred, expected, None, phone_equiv=phone_equiv)
        ):
            word_ok_relaxed += 1
        n = max(len(expected), len(pred))
        phone_den += n
        for i in range(min(len(expected), len(pred))):
            a, b = expected[i], pred[i]
            if _phone_eq(a, b, phone_equiv):
                phone_ok += 1
        word_edits = edit_distance(expected, pred, phone_equiv=None)
        edit_sum += word_edits
        if phone_equiv is not None:
            edit_sum_equiv += edit_distance(expected, pred, phone_equiv=phone_equiv)
        else:
            edit_sum_equiv += word_edits
    n_test = len(test_set)
    return {
        "wer_pct": 100.0 * (1.0 - word_ok / n_test),
        "wer_relaxed_pct": 100.0 * (1.0 - word_ok_relaxed / n_test),
        "per_pct": 100.0 * edit_sum / phone_den if phone_den else 0.0,
        "per_equiv_pct": 100.0 * edit_sum_equiv / phone_den if phone_den else 0.0,
        "pos_acc_pct": 100.0 * phone_ok / phone_den if phone_den else 0.0,
        "n_test": n_test,
    }


def train_baseline(
    locale: str,
    phoneset: str,
    train_lines: list[str],
    *,
    use_dict_fallback: bool = False,
    exceptions: dict[str, list[str]] | None = None,
):
    from phonebox.core.g2p_model import G2PDecisionTree

    dt = G2PDecisionTree(
        locale=locale,
        phoneset_name=phoneset,
        remove_stress=False,
        verbose=False,
        trainer="native",
        parallel_align=False,
        max_combinations=5000,
        use_dict_fallback=use_dict_fallback,
    )
    dt.load_prondict(iter(train_lines))
    dt.align()
    dt.load_alignments()
    dt.train(prune=False)
    if exceptions is not None:
        dt.exceptions = exceptions
    return dt


def load_baseline(
    model_path: Path,
    locale: str,
    phoneset: str,
    *,
    use_dict_fallback: bool = False,
    exceptions: dict[str, list[str]] | None = None,
):
    from phonebox.core.g2p_model import G2PDecisionTree

    dt = G2PDecisionTree(
        locale=locale,
        phoneset_name=phoneset,
        remove_stress=False,
        verbose=False,
        use_dict_fallback=use_dict_fallback,
    )
    dt.load_model(str(model_path))
    # Never use the full-lexicon embedded exceptions during held-out eval.
    dt.use_dict_fallback = use_dict_fallback
    if exceptions is not None:
        dt.exceptions = exceptions
    elif not use_dict_fallback:
        dt.exceptions = {}
    return dt


def train_multigram(
    train_pairs: list[tuple[list[str], list[str]]],
    max_l: int,
    max_p: int,
    em_iters: int,
    *,
    verbose: bool = False,
    parallel_align: bool = False,
    parallel_viterbi: bool = False,
    lm_order: int = 2,
    decode_beam: int = 0,
    use_dict_fallback: bool = False,
    exceptions: dict[str, list[str]] | None = None,
):
    from phonebox.core.multigram_g2p import MultigramG2P

    mg = MultigramG2P(
        max_letter_span=max_l,
        max_phone_span=max_p,
        min_phone_span=0,
        em_max_iterations=em_iters,
        lm_order=lm_order,
        decode_beam=decode_beam,
        verbose=verbose,
        parallel_align=parallel_align,
        parallel_viterbi=parallel_viterbi or parallel_align,
    )
    mg.train_from_pairs(train_pairs)
    mg.use_dict_fallback = use_dict_fallback
    if exceptions is not None:
        mg.exceptions = exceptions
    return mg


def run_compare(
    *,
    lexicon: Path,
    locale: str,
    phoneset: str = DEFAULT_MULTIGRAM_PHONESET,
    seed: int = DEFAULT_SPLIT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
    max_letter_span: int = 2,
    max_phone_span: int = 2,
    em_iterations: int = 15,
    lm_order: int = 2,
    decode_beam: int = 0,
    parallel_align: bool = False,
    parallel_viterbi: bool = False,
    verbose: bool = False,
    vowel_equiv: bool = False,
    phone_equiv: frozenset[tuple[str, str]] | None = None,
    train_normalize_policy: str | None = None,
    experiment_label: str | None = None,
    no_config_joins: bool = False,
    skip_baseline: bool = False,
    skip_multigram: bool = False,
    baseline_model: Path | None = None,
    use_exceptions: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    """Run comparison; return metadata plus per-model metric dicts."""
    if not lexicon.is_file():
        raise FileNotFoundError(f"lexicon not found: {lexicon}")
    if baseline_model is not None and not baseline_model.is_file():
        raise FileNotFoundError(f"baseline model not found: {baseline_model}")
    if no_config_joins and baseline_model is not None:
        if not quiet:
            print(
                "  note: no_config_joins — ignoring pretrained 1:1; training tree on train split",
                flush=True,
            )
        baseline_model = None

    pairs = load_lexicon(lexicon)
    equiv = _resolve_phone_equiv(
        locale, vowel_equiv=vowel_equiv, phone_equiv=phone_equiv
    )
    vec = Vectorizer(locale=locale, phoneset_name=phoneset, remove_stress=False)
    if no_config_joins:
        vec.disable_config_joins()

    mg_cfg = vec.multigram_config()
    if "max_letter_span" in mg_cfg:
        max_letter_span = mg_cfg["max_letter_span"]
    if "max_phone_span" in mg_cfg:
        max_phone_span = mg_cfg["max_phone_span"]

    test_raw, train_raw = split_lexicon(
        pairs, seed=seed, test_fraction=test_fraction, max_test=max_test
    )

    if train_normalize_policy:
        train_raw = [
            (w, apply_train_normalize(locale, train_normalize_policy, w, p))
            for w, p in train_raw
        ]
    train_lines = [f"{w}\t{' '.join(p)}" for w, p in train_raw]
    train_cooked: list[tuple[list[str], list[str]]] = []
    test_eval: list[tuple[str, list[str]]] = []
    for word, phones in test_raw:
        cooked = cook_pair(vec, word, phones)
        if cooked:
            letters, cooked_phones = cooked
            test_eval.append((word, cooked_phones))
    for word, phones in train_raw:
        cooked = cook_pair(vec, word, phones)
        if cooked:
            train_cooked.append(cooked)

    gold_variants = build_gold_variants(pairs, vec)
    n_multi = sum(1 for v in gold_variants.values() if len(v) > 1)
    train_exceptions = (
        build_train_exceptions(train_raw, vec) if use_exceptions else None
    )

    if not quiet:
        print(f"locale={locale} lexicon={lexicon.name}", flush=True)
        print(
            f"  entries={len(pairs)} train={len(train_raw)} test={len(test_eval)}",
            flush=True,
        )
        print(f"  words with multiple lexicon prons: {n_multi}", flush=True)
        if no_config_joins:
            print(
                "  config joins: off (xlit only; EM may still learn multigram units)",
                flush=True,
            )
        if baseline_model is not None:
            print(
                f"  1:1 model: {baseline_model} (tree only, no embedded lexicon)",
                flush=True,
            )
        elif not skip_baseline:
            print("  1:1 model: train split (G2PDecisionTree)", flush=True)
        if use_exceptions:
            n_exc = len(train_exceptions) if train_exceptions else 0
            print(f"  exceptions: train-split only ({n_exc} words)", flush=True)
        else:
            print("  exceptions: off (pure G2P, no lexicon lookup)", flush=True)
        if train_normalize_policy:
            print(
                f"  train normalize: {train_normalize_policy} (test gold unchanged)",
                flush=True,
            )
        if experiment_label:
            print(f"  experiment: {experiment_label}", flush=True)
        if equiv is not None:
            print(f"  relaxed PER (PERr): {len(equiv) // 2} equiv pairs", flush=True)

    results: list[tuple[str, float, dict[str, float]]] = []

    if not skip_baseline:
        t0 = time.time()
        if baseline_model is not None:
            if not quiet:
                print("loading G2PDecisionTree (1:1)…", flush=True)
            baseline = load_baseline(
                baseline_model,
                locale,
                phoneset,
                use_dict_fallback=use_exceptions,
                exceptions=train_exceptions,
            )
        else:
            if not quiet:
                print("training G2PDecisionTree (1:1)…", flush=True)
            baseline = train_baseline(
                locale,
                phoneset,
                train_lines,
                use_dict_fallback=use_exceptions,
                exceptions=train_exceptions,
            )
        if not quiet:
            print(f"  done in {time.time() - t0:.1f}s", flush=True)
        m = evaluate(
            "1:1",
            predict_cooked_phones(vec, baseline.pronounce),
            test_eval,
            gold_variants=gold_variants,
            phone_equiv=equiv,
        )
        results.append(("G2PDecisionTree", time.time() - t0, m))

    if not skip_multigram:
        if not quiet:
            print("training MultigramG2P (n:m)…", flush=True)
        t0 = time.time()
        multigram = train_multigram(
            train_cooked,
            max_letter_span,
            max_phone_span,
            em_iterations,
            verbose=verbose,
            parallel_align=parallel_align,
            parallel_viterbi=parallel_viterbi or parallel_align,
            lm_order=lm_order,
            decode_beam=decode_beam,
            use_dict_fallback=use_exceptions,
            exceptions=train_exceptions,
        )
        if not quiet:
            print(f"  done in {time.time() - t0:.1f}s", flush=True)

        def mg_predict(word: str) -> list[str]:
            letters = vec.cook_letters(word, g2p=True)
            pred = multigram.pronounce_letters(letters, word=word)
            cooked = vec.cook_phones(pred)
            return cooked if cooked else pred

        m = evaluate(
            "multigram",
            mg_predict,
            test_eval,
            gold_variants=gold_variants,
            phone_equiv=equiv,
        )
        results.append(("MultigramG2P", time.time() - t0, m))

    out: dict[str, object] = {
        "locale": locale,
        "lexicon": lexicon.name,
        "experiment_label": experiment_label,
        "no_config_joins": no_config_joins,
        "train_normalize_policy": train_normalize_policy,
        "n_entries": len(pairs),
        "n_test": len(test_eval),
        "n_multi_pron": n_multi,
        "baseline_model": str(baseline_model) if baseline_model else None,
        "use_exceptions": use_exceptions,
        "phone_equiv": equiv is not None,
        "results": [
            {"model": name, "train_s": train_s, **metrics}
            for name, train_s, metrics in results
        ],
    }
    if len(results) == 2:
        out["per_delta_pp"] = results[0][2]["per_pct"] - results[1][2]["per_pct"]
    return out


def print_results_table(
    results: list[tuple[str, float, dict[str, float]]],
    *,
    show_relaxed_per: bool = False,
) -> None:
    hdr = f"{'model':<18} {'train_s':>8} {'WER%':>8} {'WERr%':>8} {'PER%':>8}"
    if show_relaxed_per:
        hdr += f" {'PERr%':>8}"
    hdr += f" {'pos%':>8}"
    print()
    print(hdr)
    print("-" * (len(hdr) + 2))
    for name, train_s, m in results:
        row = (
            f"{name:<18} {train_s:8.1f} {m['wer_pct']:8.2f} "
            f"{m['wer_relaxed_pct']:8.2f} {m['per_pct']:8.2f}"
        )
        if show_relaxed_per:
            row += f" {m['per_equiv_pct']:8.2f}"
        row += f" {m['pos_acc_pct']:8.2f}"
        print(row)
    print("  WERr = any lexicon variant counts as correct")
    print("  PER% = phone edit rate; PERr% = PER with locale equiv (it_IT, pt_BR)")
    if len(results) == 2:
        d_per = results[0][2]["per_pct"] - results[1][2]["per_pct"]
        winner = "MultigramG2P" if d_per > 0 else "G2PDecisionTree"
        print(
            f"\nPER delta (1:1 − multigram): {d_per:+.2f} pp → lower PER wins ({winner})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexicon", required=True, type=Path)
    ap.add_argument("--locale", required=True)
    ap.add_argument("--phoneset", default=DEFAULT_MULTIGRAM_PHONESET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"Held-out fraction (default {DEFAULT_TEST_FRACTION}).",
    )
    ap.add_argument(
        "--max-test",
        type=int,
        default=DEFAULT_MAX_TEST_ENTRIES,
        help=f"Cap test entries (default {DEFAULT_MAX_TEST_ENTRIES}).",
    )
    ap.add_argument("--max-letter-span", type=int, default=2)
    ap.add_argument("--max-phone-span", type=int, default=2)
    ap.add_argument("--em-iterations", type=int, default=15)
    ap.add_argument(
        "--lm-order",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Unit n-gram order for joint decode (default 2).",
    )
    ap.add_argument(
        "--decode-beam",
        type=int,
        default=0,
        help="Beam width for joint decode (0 = exact Viterbi).",
    )
    ap.add_argument(
        "--parallel-align",
        action="store_true",
        help="Parallel multigram EM E-step (off by default).",
    )
    ap.add_argument(
        "--parallel-viterbi",
        action="store_true",
        help="Parallel post-EM Viterbi batch (defaults on when --parallel-align).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Per-iteration EM / phase timings on stderr.",
    )
    ap.add_argument(
        "--vowel-equiv",
        action="store_true",
        help="Alias for --relaxed-per (Italian e/ɛ, o/ɔ).",
    )
    ap.add_argument(
        "--relaxed-per",
        action="store_true",
        help="PERr column: locale phone-equivalence (it_IT, pt_BR).",
    )
    ap.add_argument(
        "--train-normalize",
        default=None,
        metavar="POLICY",
        help="Train-split phone policy (see phonebox.experiments.normalize); test gold unchanged.",
    )
    ap.add_argument(
        "--experiment",
        default=None,
        help="Label stored in experiment result metadata.",
    )
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-multigram", action="store_true")
    ap.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="Load 1:1 G2PDecisionTree from disk instead of training on the train split.",
    )
    ap.add_argument(
        "--use-exceptions",
        action="store_true",
        help="Hybrid mode: train-split lexicon lookup for 1:1 and n:m (default: off).",
    )
    ap.add_argument(
        "--no-config-joins",
        action="store_true",
        help="Disable locale config.json joins; train 1:1 on train split (ignore --baseline-model).",
    )
    args = ap.parse_args()

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
            vowel_equiv=args.vowel_equiv,
            phone_equiv=equiv_for_locale(args.locale)
            if args.relaxed_per or args.vowel_equiv
            else None,
            train_normalize_policy=args.train_normalize,
            experiment_label=args.experiment,
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


if __name__ == "__main__":
    raise SystemExit(main())
