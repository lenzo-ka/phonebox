"""
Text normalization for G2P processing.

Provides consistent normalization across CLI, runners, and library.
"""

from __future__ import annotations

import unicodedata

# Unicode categories to strip from token edges
# P=Punctuation, S=Symbol, C=Control, M=Mark, Z=Separator
EXCLUDE_CATEGORIES = {"P", "S", "C", "M", "Z"}


def normalize_nfc(text: str) -> str:
    """NFC-normalize and strip whitespace from a single string."""
    return unicodedata.normalize("NFC", text.strip())


def normalize_text(text: str) -> list[str]:
    """
    Normalize text for G2P processing.

    - NFC Unicode normalization
    - Splits on whitespace
    - Strips punctuation/symbols from token edges
    - Preserves internal punctuation (apostrophes, hyphens)

    Args:
        text: Input text

    Returns:
        List of normalized tokens
    """
    text = unicodedata.normalize("NFC", text)
    result = []

    for token in text.split():
        # Strip leading excluded characters
        while token and unicodedata.category(token[0])[0] in EXCLUDE_CATEGORIES:
            token = token[1:]
        # Strip trailing excluded characters
        while token and unicodedata.category(token[-1])[0] in EXCLUDE_CATEGORIES:
            token = token[:-1]
        if token:
            result.append(token)

    return result


def tokenize_raw(text: str) -> list[str]:
    """
    Simple whitespace tokenization (no normalization).

    Args:
        text: Input text

    Returns:
        List of tokens split on whitespace
    """
    return text.split()
