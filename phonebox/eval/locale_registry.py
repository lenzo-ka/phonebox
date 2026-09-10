"""Conventional paths for the repository's curated evaluation locales."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from phonebox.locale_resolution import canonical_locale, resolve_locale


@dataclass(frozen=True)
class EvaluationLocale:
    """Conventional lexicon/model names and evaluation behavior for one locale."""

    locale: str
    lexicon_name: str
    baseline_model: str
    sweep_relaxed_per: bool = False


EVALUATION_LOCALES = {
    item.locale: item
    for item in (
        EvaluationLocale("es_MX", "es_ipa.tsv", "es-mx/es-mx-ipa.g2p.gz"),
        EvaluationLocale("fr_FR", "fr_ipa.tsv", "fr-fr/fr-fr-ipa.g2p.gz"),
        EvaluationLocale("de_DE", "de_ipa.tsv", "de-de/de-de-ipa.g2p.gz"),
        EvaluationLocale("en_US", "en_ipa.tsv", "en-us/en-us-ipa.g2p.gz"),
        EvaluationLocale("pt_BR", "pt_ipa.tsv", "pt-br/pt-br-ipa.g2p.gz"),
        EvaluationLocale(
            "it_IT", "it_ipa.tsv", "it-it/it-it-ipa.g2p.gz", sweep_relaxed_per=True
        ),
    )
}


def canonical_locale_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    """Canonicalize locale keys while preserving caller-supplied paths."""
    result: dict[str, Path] = {}
    for locale, path in paths.items():
        canonical = canonical_locale(locale)
        if canonical in result:
            raise ValueError(f"duplicate locale identity: {canonical}")
        result[canonical] = Path(path)
    return result


def canonical_locales(locales: Sequence[str]) -> list[str]:
    """Canonicalize requested locale identities and reject duplicates."""
    result = [canonical_locale(locale) for locale in locales]
    if len(result) != len(set(result)):
        raise ValueError("duplicate locale identities")
    return result


def select_locale_paths(
    paths: Mapping[str, Path], locales: Sequence[str] | None = None
) -> dict[str, Path]:
    """Select explicit paths by exact or bare-language-likely locale identity."""
    canonical_paths = canonical_locale_paths(paths)
    requested = (
        canonical_locales(locales) if locales is not None else list(canonical_paths)
    )
    selected: dict[str, Path] = {}
    for locale in requested:
        resolution = resolve_locale(locale, canonical_paths)
        if resolution.resolved is None:
            raise ValueError(
                f"no path supplied for locale {resolution.requested!r}; "
                f"available: {', '.join(canonical_paths)}"
            )
        selected[resolution.requested] = canonical_paths[resolution.resolved]
    return selected


def evaluation_locale(locale: str) -> EvaluationLocale:
    """Select an exact or bare-language-likely curated evaluation entry."""
    resolution = resolve_locale(locale, EVALUATION_LOCALES)
    if resolution.resolved is None:
        supported = ", ".join(EVALUATION_LOCALES)
        raise ValueError(
            f"unsupported curated locales [{resolution.requested!r}]; "
            f"choose from {supported}"
        )
    return EVALUATION_LOCALES[resolution.resolved]


__all__ = [
    "EVALUATION_LOCALES",
    "EvaluationLocale",
    "canonical_locale_paths",
    "canonical_locales",
    "evaluation_locale",
    "select_locale_paths",
]
