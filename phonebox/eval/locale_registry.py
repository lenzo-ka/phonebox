"""Conventional paths for the repository's curated evaluation locales."""

from dataclasses import dataclass


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


__all__ = ["EVALUATION_LOCALES", "EvaluationLocale"]
