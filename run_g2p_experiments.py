#!/usr/bin/env python
"""Run it_IT / pt_BR G2P experiments (error analysis + train-normalize policies).

Does not overwrite ``docs/G2P_COMPARE.md`` or ``docs/G2P_COMPARE_BASELINE.md``.
Writes under ``docs/experiments/``.

Requires PHONEDECODING_LEXICON_DIR and PHONEDECODING_G2P_DIR (same as compare_g2p_all).

Example::

    export PHONEDECODING_LEXICON_DIR=…/processed
    export PHONEDECODING_G2P_DIR=…/build/g2p
    python run_g2p_experiments.py --parallel-align
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import (
    cook_pair,
    load_lexicon,
    predict_cooked_phones,
    run_compare,
    train_multigram,
)
from phonebox.experiments.analysis import (
    audit_normalize_delta,
    collect_phone_substitutions,
    format_substitution_table,
)
from phonebox.experiments.equiv import equiv_for_locale
from phonebox.experiments.normalize import NORMALIZE_POLICIES
from phonebox.experiments.split import split_lexicon

_EXPERIMENTS: list[tuple[str, str, str, str]] = [
    ("it_IT", "it_ipa.tsv", "it-it/it-it-ipa.g2p.gz", "baseline"),
    ("it_IT", "it_ipa.tsv", "it-it/it-it-ipa.g2p.gz", "spelling_gated"),
    ("it_IT", "it_ipa.tsv", "it-it/it-it-ipa.g2p.gz", "collapse_open"),
    ("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz", "baseline"),
    ("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz", "surface_final"),
    ("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz", "do_du"),
    ("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz", "citation_expand"),
]


def _write_locale_doc(
    path: Path,
    locale: str,
    *,
    audits: dict[str, dict[str, int]],
    runs: list[dict[str, object]],
    error_sections: list[str],
) -> None:
    lines = [
        f"# G2P experiments: {locale}",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Baseline numbers (unaltered lexicon, seed 42) are frozen in "
        "[`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md).",
        "",
        "## Train-normalize policies",
        "",
    ]
    if locale == "it_IT":
        lines.extend(
            [
                "| Policy | Train gold change | Why |",
                "|--------|-------------------|-----|",
                "| `baseline` | none | Control; matches baseline compare. |",
                "| `spelling_gated` | ɛ→e, ɔ→o only when word has no è/ò | "
                "Plain spelling dominates; open vowels stay when orthography marks them. |",
                "| `collapse_open` | always ɛ→e, ɔ→o | Upper bound; removes open/closed entirely on train. |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Policy | Train gold change | Why |",
                "|--------|-------------------|-----|",
                "| `baseline` | none | Control. |",
                "| `surface_final` | final `o`/`ʊ`→`u` when spelling ends in plain `o` | "
                "~99% of `-o` words already have `u`; fixes mixed gold. |",
                "| `do_du` | `surface_final` + `… d o`→`… d u` at `-do`/`-ado` | "
                "Targets dominant n:m confusion (`d o` vs `d u`). |",
                "| `citation_expand` | final `u`→`o` on plain `-o` words | "
                "Contrast: full vowels on train (expected to hurt n:m). |",
                "",
            ]
        )

    lines.append("## Train-split audit (phones changed before cook)")
    lines.append("")
    lines.append("| Policy | Entries changed | Phone token edits |")
    lines.append("|--------|-----------------|-------------------|")
    for policy, stats in sorted(audits.items()):
        lines.append(
            f"| `{policy}` | {stats['entries_changed']} / {stats['train_entries']} | "
            f"{stats['phone_token_changes']} |"
        )
    lines.append("")

    lines.append("## Metrics (test gold = original lexicon)")
    lines.append("")
    lines.append(
        "| Policy | 1:1 WER% | 1:1 PER% | n:m WER% | n:m PER% | Δ PER | 1:1 PERr% | n:m PERr% |"
    )
    lines.append(
        "|--------|----------|----------|----------|----------|-------|-----------|-----------|"
    )
    for run in runs:
        by_name = {r["model"]: r for r in cast(list[dict[str, Any]], run["results"])}
        one = by_name["G2PDecisionTree"]
        mg = by_name["MultigramG2P"]
        delta = one["per_pct"] - mg["per_pct"]
        lines.append(
            f"| `{run['train_normalize_policy'] or 'baseline'}` | "
            f"{one['wer_pct']:.2f} | {one['per_pct']:.2f} | "
            f"{mg['wer_pct']:.2f} | {mg['per_pct']:.2f} | {delta:+.2f} | "
            f"{one['per_equiv_pct']:.2f} | {mg['per_equiv_pct']:.2f} |"
        )
    lines.append("")
    lines.append(
        "PERr uses locale relaxed phone-equivalence (see `phonebox/experiments/equiv.py`)."
    )
    lines.append("")

    for section in error_sections:
        lines.append(section)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_error_analysis(
    locale: str,
    lexicon: Path,
    model_path: Path,
    *,
    seed: int,
    max_test: int,
    parallel_align: bool,
    em_iterations: int,
) -> tuple[str, str]:
    """Return markdown sections for 1:1 and n:m substitution tables on baseline test set."""
    pairs = load_lexicon(lexicon)
    test_raw, train_raw = split_lexicon(pairs, seed=seed, max_test=max_test)
    vec = Vectorizer(locale=locale, phoneset_name="ipa", remove_stress=False)
    test_eval: list[tuple[str, list[str]]] = []
    for word, phones in test_raw:
        cooked = cook_pair(vec, word, phones)
        if cooked:
            test_eval.append((word, cooked[1]))

    train_cooked = []
    for word, phones in train_raw:
        cooked = cook_pair(vec, word, phones)
        if cooked:
            train_cooked.append(cooked)

    from phonebox.eval.g2p_compare import load_baseline

    baseline = load_baseline(model_path, locale, "ipa", use_dict_fallback=False)
    b_pred = predict_cooked_phones(vec, baseline.pronounce)
    b_subs, _ = collect_phone_substitutions(b_pred, test_eval)
    b_table = format_substitution_table(
        b_subs, title="Top phone substitutions (1:1, baseline test)"
    )

    mg = train_multigram(
        train_cooked,
        vec.multigram_config().get("max_letter_span", 2),
        vec.multigram_config().get("max_phone_span", 2),
        em_iterations,
        parallel_align=parallel_align,
        parallel_viterbi=parallel_align,
    )

    def mg_predict(word: str) -> list[str]:
        letters = vec.cook_letters(word, g2p=True)
        pred = mg.pronounce_letters(letters, word=word)
        cooked = vec.cook_phones(pred)
        return cooked if cooked else pred

    m_subs, _ = collect_phone_substitutions(mg_predict, test_eval)
    m_table = format_substitution_table(
        m_subs, title="Top phone substitutions (n:m, baseline test)"
    )
    return b_table, m_table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lexicon-dir", type=Path, default=None)
    ap.add_argument("--g2p-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("docs/experiments"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--em-iterations", type=int, default=15)
    ap.add_argument("--parallel-align", action="store_true")
    ap.add_argument("--skip-error-analysis", action="store_true")
    ap.add_argument("--locales", nargs="*", choices=["it_IT", "pt_BR"])
    ap.add_argument(
        "--policies",
        nargs="*",
        help="Subset of normalize policy names (default: all for selected locales)",
    )
    args = ap.parse_args()

    lex_dir = args.lexicon_dir or os.environ.get("PHONEDECODING_LEXICON_DIR")
    g2p_dir = args.g2p_dir or os.environ.get("PHONEDECODING_G2P_DIR")
    if not lex_dir or not g2p_dir:
        print(
            "Set PHONEDECODING_LEXICON_DIR and PHONEDECODING_G2P_DIR", file=sys.stderr
        )
        return 2

    lex_root = Path(lex_dir)
    g2p_root = Path(g2p_dir)
    out_root = args.output_dir
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    selected_locales = set(args.locales) if args.locales else {"it_IT", "pt_BR"}
    manifest: list[dict[str, object]] = []
    t_all = time.time()

    for locale, lex_name, model_rel, policy in _EXPERIMENTS:
        if locale not in selected_locales:
            continue
        if args.policies and policy not in args.policies:
            continue

        lexicon = lex_root / lex_name
        model_path = g2p_root / model_rel
        label = f"{locale}_{policy}"
        print(f"\n=== {label} ===", flush=True)

        train_norm = None if policy == "baseline" else policy
        summary = run_compare(
            lexicon=lexicon,
            locale=locale,
            seed=args.seed,
            max_test=args.max_test,
            em_iterations=args.em_iterations,
            parallel_align=args.parallel_align,
            phone_equiv=equiv_for_locale(locale),
            train_normalize_policy=train_norm,
            experiment_label=label,
            baseline_model=model_path,
            quiet=False,
        )
        result_path = results_dir / f"{label}.json"
        result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest.append({"label": label, "path": str(result_path), **summary})

    by_locale: dict[str, list[dict[str, object]]] = {}
    for entry in manifest:
        loc = entry["locale"]
        by_locale.setdefault(str(loc), []).append(entry)

    for locale in sorted(by_locale):
        lexicon = lex_root / ("it_ipa.tsv" if locale == "it_IT" else "pt_ipa.tsv")
        model_rel = (
            "it-it/it-it-ipa.g2p.gz" if locale == "it_IT" else "pt-br/pt-br-ipa.g2p.gz"
        )
        pairs = load_lexicon(lexicon)
        _test, train_raw = split_lexicon(pairs, seed=args.seed, max_test=args.max_test)
        audits = {
            policy: audit_normalize_delta(train_raw, locale, policy)
            for policy in NORMALIZE_POLICIES[locale]
        }
        error_sections: list[str] = []
        if not args.skip_error_analysis:
            print(f"\n--- error analysis {locale} ---", flush=True)
            b_tab, m_tab = _run_error_analysis(
                locale,
                lexicon,
                g2p_root / model_rel,
                seed=args.seed,
                max_test=args.max_test,
                parallel_align=args.parallel_align,
                em_iterations=args.em_iterations,
            )
            error_sections = [b_tab, m_tab]

        runs = sorted(
            by_locale[locale], key=lambda r: str(r.get("train_normalize_policy") or "")
        )
        _write_locale_doc(
            out_root / f"{locale}.md",
            locale,
            audits=audits,
            runs=runs,
            error_sections=error_sections,
        )

    index_lines = [
        "# G2P experiments (it_IT, pt_BR)",
        "",
        f"Last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Frozen pre-experiment metrics: [`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md).",
        "",
        "Join-off fair compare (both models train split): "
        "[`G2P_COMPARE_NO_JOINS.md`](../G2P_COMPARE_NO_JOINS.md).",
        "",
        "Eval scripts and metrics: [`G2P_EVAL.md`](../G2P_EVAL.md).",
        "",
        "## Summary",
        "",
        "See **[RESULTS.md](RESULTS.md)** for outcomes.",
        "",
        "Train-split **phone** normalization policies live in "
        "`phonebox/experiments/normalize.py`. Test evaluation always uses the "
        "**original** lexicon pronunciations.",
        "",
        "## Locale reports",
        "",
        "- [it_IT.md](it_IT.md) — open/closed vowels (ɛ/e, ɔ/o)",
        "- [pt_BR.md](pt_BR.md) — final reduction and `d o`→`d u`",
        "",
        "## Raw JSON",
        "",
        "Per-run metrics: `results/<locale>_<policy>.json`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export PHONEDECODING_LEXICON_DIR=…/processed",
        "export PHONEDECODING_G2P_DIR=…/build/g2p",
        "python run_g2p_experiments.py --parallel-align",
        "```",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {out_root} in {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
