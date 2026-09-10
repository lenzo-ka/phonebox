#!/usr/bin/env python
"""Structured multi-locale 1:1 versus n:m evaluation and report rendering.

The API accepts explicit locale-to-lexicon and locale-to-model mappings. The
``phonebox compare all`` adapter owns conventional directory and environment
defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CompareAllConfig:
    """Settings needed to render a multi-locale comparison."""

    no_config_joins: bool = False
    seed: int = DEFAULT_SPLIT_SEED
    max_test: int = DEFAULT_MAX_TEST_ENTRIES
    em_iterations: int = 15
    parallel_align: bool = False
    use_exceptions: bool = False


def write_compare_all(
    path: Path, summaries: list[dict[str, object]], *, config: CompareAllConfig
) -> None:
    """Write multi-locale summaries as the established Markdown report."""
    no_joins = config.no_config_joins
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
            f"- Seed: {config.seed}",
            f"- Test cap: {config.max_test} entries (10% split, shuffled)",
            one_one_line,
            f"- n:m: MultigramG2P v3 joint decode + LM, EM iterations={config.em_iterations}",
            joins_line,
            f"- Parallel align: {config.parallel_align}",
            f"- Exceptions / lexicon lookup: {'train split only' if config.use_exceptions else 'off (pure G2P)'}",
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
    lexicons: Mapping[str, Path],
    baseline_models: Mapping[str, Path] | None,
    no_config_joins: bool = False,
    seed: int = DEFAULT_SPLIT_SEED,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
    em_iterations: int = 15,
    parallel_align: bool = False,
    use_exceptions: bool = False,
    locales: list[str] | None = None,
    quiet: bool = True,
) -> list[dict[str, object]]:
    """Compare explicit locale lexicons/models and return structured summaries."""
    if not no_config_joins and baseline_models is None:
        raise ValueError("baseline_models is required when config joins are enabled")
    selected = locales or list(lexicons)
    summaries: list[dict[str, object]] = []

    for locale in selected:
        lexicon = Path(lexicons[locale])
        model_path = None if no_config_joins else Path(baseline_models[locale])
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
            quiet=quiet,
        )
        summary["baseline_model"] = None if no_config_joins else str(model_path)
        summaries.append(summary)
    return summaries


__all__ = ["CompareAllConfig", "run_compare_all", "write_compare_all"]
