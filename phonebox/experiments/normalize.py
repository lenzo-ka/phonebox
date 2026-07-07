"""Train-split lexicon phone normalization policies (experiments only).

Test gold is never modified: models are scored against the original lexicon phones.
"""

from __future__ import annotations

from collections.abc import Callable

_OPEN_MARKS = frozenset("\u00e8\u00c8\u00f2\u00d2")  # è, È, ò, Ò
_ACCENTED_FINAL_O = frozenset("óôõÓÔÕ")


def _replace_phones(phones: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(p, p) for p in phones]


def normalize_identity(_word: str, phones: list[str]) -> list[str]:
    return phones


def normalize_it_spelling_gated(word: str, phones: list[str]) -> list[str]:
    """Collapse open vowels to mid when spelling has no è/ò.

    Rationale: lexicon has no stress markers; plain ``e``/``o`` spellings dominate.
    Accented ``è``/``ò`` are mapped to letter ``ɛ``/``ɔ`` in ``g2p.xlit`` and should
    keep open vowels in gold. Words without those marks get ɛ→e, ɔ→o on the train
    split so n:m units are less ambiguous. Test eval still uses original gold.
    """
    if any(ch in word for ch in _OPEN_MARKS):
        return phones
    return _replace_phones(phones, {"ɛ": "e", "ɔ": "o"})


def normalize_it_collapse_open(_word: str, phones: list[str]) -> list[str]:
    """Upper bound: always ɛ→e, ɔ→o (ignores spelling)."""
    return _replace_phones(phones, {"ɛ": "e", "ɔ": "o"})


def normalize_pt_surface_final(word: str, phones: list[str]) -> list[str]:
    """Align unstressed orthographic ``-o`` with surface ``… u`` in gold.

    Rationale: ~99% of words ending in plain ``o`` already have final phone ``u`` in
    ``pt_ipa.tsv``; a handful still end in ``o``/``ʊ``. Normalizing train gold removes
    mixed targets for the same letter context (helps n:m ``d o``→``d u`` units).
    Does not change nasal ``õ``/``ão`` (different letter sequences).
    """
    if not phones or not word.endswith("o"):
        return phones
    if word[-1] == "o" and len(word) >= 2 and word[-2] in "ãõ":
        return phones
    if word[-1] == "o" and any(ch in word for ch in _ACCENTED_FINAL_O):
        return phones
    out = list(phones)
    if out[-1] in ("o", "ʊ"):
        out[-1] = "u"
    return out


def normalize_pt_do_du(word: str, phones: list[str]) -> list[str]:
    """``surface_final`` plus ``… d o`` → ``… d u`` at suffix ``-do`` / ``-ado``."""
    out = normalize_pt_surface_final(word, phones)
    if (
        len(out) >= 2
        and word.endswith(("do", "ado"))
        and out[-2] == "d"
        and out[-1] in ("o", "ʊ")
    ):
        out[-1] = "u"
    return out


def normalize_pt_citation_expand(word: str, phones: list[str]) -> list[str]:
    """Opposite experiment: final ``u`` → ``o`` when spelling ends in plain ``o``.

    Tests whether teaching full vowels (citation form) helps the 1:1 tree; expected
    to hurt n:m contextual reduction. Included for contrast only.
    """
    if not phones or not word.endswith("o"):
        return phones
    if any(ch in word for ch in _ACCENTED_FINAL_O) or (
        len(word) >= 2 and word[-2] in "ãõ"
    ):
        return phones
    out = list(phones)
    if out[-1] == "u":
        out[-1] = "o"
    return out


PolicyFn = Callable[[str, list[str]], list[str]]

NORMALIZE_POLICIES: dict[str, dict[str, PolicyFn]] = {
    "it_IT": {
        "baseline": normalize_identity,
        "spelling_gated": normalize_it_spelling_gated,
        "collapse_open": normalize_it_collapse_open,
    },
    "pt_BR": {
        "baseline": normalize_identity,
        "surface_final": normalize_pt_surface_final,
        "do_du": normalize_pt_do_du,
        "citation_expand": normalize_pt_citation_expand,
    },
}


def apply_train_normalize(
    locale: str,
    policy: str,
    word: str,
    phones: list[str],
) -> list[str]:
    policies = NORMALIZE_POLICIES.get(locale)
    if policies is None:
        raise KeyError(f"no normalize policies for locale {locale!r}")
    fn = policies.get(policy)
    if fn is None:
        raise KeyError(
            f"unknown policy {policy!r} for {locale}; choose from {sorted(policies)}"
        )
    return fn(word, phones)
