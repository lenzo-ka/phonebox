"""Run the supported Italian and Portuguese normalization experiments.

The API accepts explicit :class:`ExperimentSpec` paths and writes its report
under the requested output directory. Environment and layout conveniences are
confined to the CLI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
from phonebox.locale_resolution import resolve_locale


@dataclass(frozen=True)
class ExperimentSpec:
    """One locale, lexicon, baseline model, and normalization policy run."""

    locale: str
    lexicon: Path
    baseline_model: Path
    policy: str = "baseline"


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
        f"Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
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


def run_experiments(
    experiments: list[ExperimentSpec],
    *,
    output_dir: Path = Path("docs/experiments"),
    seed: int = 42,
    max_test: int = 2000,
    em_iterations: int = 15,
    parallel_align: bool = False,
    skip_error_analysis: bool = False,
    locales: list[str] | None = None,
    policies: list[str] | None = None,
) -> list[dict[str, object]]:
    """Run explicit Italian/Portuguese experiments and return their manifest."""
    normalized_experiments = []
    unsupported = set()
    for spec in experiments:
        resolution = resolve_locale(spec.locale, NORMALIZE_POLICIES)
        if resolution.resolved is None:
            unsupported.add(resolution.requested)
            continue
        normalized_experiments.append(
            ExperimentSpec(
                resolution.resolved,
                spec.lexicon,
                spec.baseline_model,
                spec.policy,
            )
        )
    if unsupported:
        raise ValueError(
            f"unsupported normalization experiment locales: {sorted(unsupported)}"
        )
    experiments = normalized_experiments
    out_root = Path(output_dir)
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    selected_locales = {spec.locale for spec in experiments}
    if locales:
        selected_locales = set()
        for locale in locales:
            resolution = resolve_locale(locale, NORMALIZE_POLICIES)
            if resolution.resolved is None:
                raise ValueError(
                    f"unsupported normalization experiment locale: "
                    f"{resolution.requested}"
                )
            selected_locales.add(resolution.resolved)
    manifest: list[dict[str, object]] = []
    t_all = time.time()

    selected_specs: dict[str, ExperimentSpec] = {}
    for spec in experiments:
        locale, policy = spec.locale, spec.policy
        if locale not in selected_locales:
            continue
        if policies and policy not in policies:
            continue

        lexicon = Path(spec.lexicon)
        model_path = Path(spec.baseline_model)
        selected_specs.setdefault(locale, spec)
        label = f"{locale}_{policy}"
        print(f"\n=== {label} ===", flush=True)

        train_norm = None if policy == "baseline" else policy
        summary = run_compare(
            lexicon=lexicon,
            locale=locale,
            seed=seed,
            max_test=max_test,
            em_iterations=em_iterations,
            parallel_align=parallel_align,
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
        spec = selected_specs[locale]
        lexicon = Path(spec.lexicon)
        pairs = load_lexicon(lexicon)
        _test, train_raw = split_lexicon(pairs, seed=seed, max_test=max_test)
        audits = {
            policy: audit_normalize_delta(train_raw, locale, policy)
            for policy in NORMALIZE_POLICIES[locale]
        }
        error_sections: list[str] = []
        if not skip_error_analysis:
            print(f"\n--- error analysis {locale} ---", flush=True)
            b_tab, m_tab = _run_error_analysis(
                locale,
                lexicon,
                Path(spec.baseline_model),
                seed=seed,
                max_test=max_test,
                parallel_align=parallel_align,
                em_iterations=em_iterations,
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
        f"Last run: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
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
        "phonebox compare experiments --parallel-align",
        "```",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {out_root} in {time.time() - t_all:.0f}s", flush=True)
    return manifest


__all__ = ["ExperimentSpec", "run_experiments"]
