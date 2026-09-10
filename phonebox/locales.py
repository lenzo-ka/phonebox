"""
Locale-aware defaults for downstream tooling (supplement generation,
phoneset validation, etc.).

Each locale dir under ``phonebox/config/locales/`` may declare a
``defaults.json`` next to the existing ``config.json`` / ``g2p.xlit``.
Currently only the ``supplement`` block is consumed:

    {
      "supplement": {
        "phonemic_geminates": false,
        "collapse_geminates": true,
        "vowel_like": ["a", "e", ...],
        "split_token": {"ɟʝ": ["ɟ", "ʝ"]}
      }
    }

``vowel_like`` is the allow-list of phones whose adjacent-doubles are
preserved when ``collapse_geminates=True`` (i.e. legitimate vowel hiatus
in romance languages, syllabic consonants, etc.). ``null`` means "no
collapse" (used for languages whose collapse default is false anyway).
``split_token`` post-processes G2P output to split a single multi-char
token into multiple phones — e.g. ``ɟʝ → ɟ ʝ`` for es_MX since the lexicon
emits the combined affricate but the canonical phoneset lists them as
separate phones.

A locale that doesn't ship a ``defaults.json`` falls back to ``default/``
(currently: no geminate collapse, no token splits).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import FILE_ENCODING
from .locale_resolution import (
    LocaleResolution,
    canonical_locale,
    locale_candidates,
    resolve_locale,
)
from .utils.logging_config import get_logger

logger = get_logger(__name__)


def _locales_dir() -> Path:
    return Path(__file__).parent / "config" / "locales"


def load_locale_defaults(locale: str) -> dict[str, Any]:
    """Load ``defaults.json`` for ``locale`` (with default-locale fallback).

    Returns ``{}`` if no ``defaults.json`` exists anywhere — callers
    should treat missing keys as "no preference, use built-in defaults".
    """
    base = _locales_dir()
    # Supplement defaults affect phone output, so exemplar-profile
    # compatibility is intentionally not used here.
    candidates = [
        *(
            base / candidate / "defaults.json"
            for candidate in locale_candidates(locale)
        ),
        base / "default" / "defaults.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding=FILE_ENCODING))
            except json.JSONDecodeError as e:
                logger.warning("Could not parse %s: %s", p, e)
                return {}
    return {}


def supplement_defaults(locale: str) -> dict[str, Any]:
    """Return the ``supplement`` block of the locale's defaults, or empty."""
    return load_locale_defaults(locale).get("supplement", {}) or {}


__all__ = [
    "LocaleResolution",
    "canonical_locale",
    "load_locale_defaults",
    "resolve_locale",
    "supplement_defaults",
]
