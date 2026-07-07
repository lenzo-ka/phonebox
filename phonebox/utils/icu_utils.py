"""
ICU utilities wrapper with fallback when PyICU is not available.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..constants import FILE_ENCODING

try:
    import icu

    HAS_ICU = True
except ImportError:
    HAS_ICU = False
    warnings.warn(
        "PyICU not installed. Transliteration features will be limited.",
        ImportWarning,
        stacklevel=2,
    )


class RuleTransliterator:
    """Transliterator that uses ICU if available, otherwise provides basic fallback."""

    def __init__(
        self, path: str | Path | None = None, rules: str | None = None
    ) -> None:
        self.path = path
        self.rules = rules
        self._transliterator = None

        if HAS_ICU and (rules or path):
            try:
                if path:
                    with open(path, encoding=FILE_ENCODING) as f:
                        rules = f.read()

                if rules:
                    self._transliterator = icu.Transliterator.createFromRules(
                        "Custom", rules, icu.UTransDirection.FORWARD
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed to create ICU transliterator: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def translit(self, text: str) -> str:
        """Apply transliteration to text (identity fallback when ICU is absent)."""
        if self._transliterator:
            return self._transliterator.transliterate(text)
        return text

    def __bool__(self) -> bool:
        """Check if transliterator is available."""
        return self._transliterator is not None
