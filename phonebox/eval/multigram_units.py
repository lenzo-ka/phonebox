"""Inspect multigram units learned from pronunciation dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phonebox.core.vectorizer import Vectorizer
from phonebox.eval.g2p_compare import cook_pair, load_lexicon, train_multigram
from phonebox.experiments.split import split_lexicon


@dataclass(frozen=True)
class MultigramUnit:
    """One learned unit with mass and configured-join membership."""

    letters: tuple[str, ...]
    phones: tuple[str, ...]
    probability: float
    configured_letter_join: bool
    configured_phone_join: bool


@dataclass(frozen=True)
class MultigramUnitAnalysis:
    """Structured learned-unit report for one locale and lexicon."""

    locale: str
    training_pairs: int
    max_letter_span: int
    max_phone_span: int
    configured_letter_joins: tuple[tuple[str, ...], ...]
    configured_phone_joins: tuple[tuple[str, ...], ...]
    units: tuple[MultigramUnit, ...]


def analyze_multigram_units(
    locale: str,
    lexicon: Path,
    *,
    top: int = 25,
    em_iterations: int = 15,
    seed: int = 42,
    max_test: int = 2000,
    parallel_align: bool = True,
) -> MultigramUnitAnalysis:
    """Train on the deterministic split and return its highest-mass n:m units."""
    pairs = load_lexicon(lexicon)
    vec = Vectorizer(locale=locale, phoneset_name="ipa")
    config = vec.multigram_config()
    max_l = config.get("max_letter_span", 2)
    max_p = config.get("max_phone_span", 2)
    _, raw_train = split_lexicon(pairs, seed=seed, max_test=max_test)
    train = [c for word, phones in raw_train if (c := cook_pair(vec, word, phones))]
    model = train_multigram(
        train,
        max_l,
        max_p,
        em_iterations,
        parallel_align=parallel_align,
        parallel_viterbi=parallel_align,
    ).model
    joins = vec.config.get("join", {})
    letter_joins = {tuple(value.split()) for value in joins.get("letters", [])}
    phone_joins = {tuple(value.split()) for value in joins.get("ipa", [])}
    found = sorted(
        (
            (letters, phones, probability)
            for (letters, phones), probability in model.aligner.q.items()
            if (len(letters) >= 2 or len(phones) >= 2) and probability >= 1e-5
        ),
        key=lambda item: -item[2],
    )[:top]
    return MultigramUnitAnalysis(
        locale,
        len(train),
        max_l,
        max_p,
        tuple(sorted(letter_joins)),
        tuple(sorted(phone_joins)),
        tuple(
            MultigramUnit(
                letters, phones, prob, letters in letter_joins, phones in phone_joins
            )
            for letters, phones, prob in found
        ),
    )


def format_multigram_units(result: MultigramUnitAnalysis) -> str:
    """Format one analysis using the historical text layout."""
    lines = [
        f"=== {result.locale} ({result.training_pairs} train pairs) ===",
        f"  multigram spans: letter={result.max_letter_span} phone={result.max_phone_span} (locale config)",
        f"  joins in config: letters={list(result.configured_letter_joins)} phones={list(result.configured_phone_joins)}",
        f"  top {len(result.units)} multigram units (n>1):",
        f"    {'letters':<14} {'phones':<22} mass     letter_join? phone_join?",
    ]
    for unit in result.units:
        lines.append(
            f"    {' '.join(unit.letters):<14} {' '.join(unit.phones):<22} {unit.probability:7.4f} {'yes' if unit.configured_letter_join else '':<12} {'yes' if unit.configured_phone_join else ''}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "MultigramUnit",
    "MultigramUnitAnalysis",
    "analyze_multigram_units",
    "format_multigram_units",
]
