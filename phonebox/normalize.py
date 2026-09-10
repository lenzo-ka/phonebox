"""
Text normalization for G2P processing.

Provides consistent normalization across CLI, runners, and library.
"""

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
        start = 0
        end = len(token)
        while (
            start < end and unicodedata.category(token[start])[0] in EXCLUDE_CATEGORIES
        ):
            start += 1
        while (
            end > start
            and unicodedata.category(token[end - 1])[0] in EXCLUDE_CATEGORIES
        ):
            end -= 1
        if start < end:
            result.append(token[start:end])

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
