"""Phone equivalence sets for relaxed PER (PERr) scoring in experiments."""

from __future__ import annotations

from phonebox.locale_resolution import resolve_locale

# Italian open/closed (same as compare_g2p historical --vowel-equiv).
_IT_VOWEL_QUALITY = frozenset({("e", "ɛ"), ("ɛ", "e"), ("o", "ɔ"), ("ɔ", "o")})

# Brazilian Portuguese: reduction and common allophony (eval-only; not training targets).
_PT_REDUCTION = frozenset(
    {
        ("o", "u"),
        ("u", "o"),
        ("e", "i"),
        ("i", "e"),
        ("ʊ", "u"),
        ("u", "ʊ"),
        ("ʁ", "ɾ"),
        ("ɾ", "ʁ"),
        ("tʃ", "t"),
        ("t", "tʃ"),
        ("dʒ", "d"),
        ("d", "dʒ"),
    }
)

_LOCALE_EQUIV: dict[str, frozenset[tuple[str, str]]] = {
    "it_IT": _IT_VOWEL_QUALITY,
    "pt_BR": _PT_REDUCTION,
}


def equiv_for_locale(locale: str) -> frozenset[tuple[str, str]] | None:
    resolved = resolve_locale(locale, _LOCALE_EQUIV).resolved
    return _LOCALE_EQUIV.get(resolved) if resolved else None


def locale_uses_relaxed_scoring(locale: str) -> bool:
    return resolve_locale(locale, _LOCALE_EQUIV).resolved is not None
