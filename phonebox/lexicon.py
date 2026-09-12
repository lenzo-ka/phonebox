"""Cycle-free parsing primitives for pronunciation lexicon lines."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRESS_PATTERNS = {
    "cmu": re.compile(r"[012]"),
    "xsampa": re.compile(r'["%]'),
}


@dataclass(frozen=True)
class LexiconEntry:
    """A parsed occurrence retaining physical source line and variant label."""

    word: str
    phones: tuple[str, ...]
    label: str
    variant: int | None
    line_number: int

    def to_dict(self) -> dict:
        """Return JSON-compatible source provenance without inferred senses."""
        return {
            "word": self.word,
            "phones": list(self.phones),
            "label": self.label,
            "variant": self.variant,
            "line_number": self.line_number,
        }


def parse_dict_entry(line: str, *, line_number: int = 1) -> LexiconEntry | None:
    """Parse shared dictionary grammar while retaining the original label."""
    line = line.strip()
    if not line or line.startswith(";;;") or line.startswith("#"):
        return None
    parts = line.split("#", 1)[0].split()
    if len(parts) < 2:
        return None
    match = re.search(r"\((\d+)\)$", parts[0])
    word = parts[0][: match.start()] if match else parts[0]
    return LexiconEntry(
        word,
        tuple(parts[1:]),
        parts[0],
        int(match.group(1)) if match else None,
        line_number,
    )


def parse_dict_line(line: str) -> tuple[str, list[str]] | None:
    """Parse whitespace-separated words/phones with comments and variants."""
    entry = parse_dict_entry(line)
    return (entry.word, list(entry.phones)) if entry is not None else None


def strip_phone_stress(phone: str, phoneset: str) -> str:
    """Strip stress defined by a known phoneset; preserve unknown tags."""
    pattern = _STRESS_PATTERNS.get(phoneset)
    return pattern.sub("", phone) if pattern is not None else phone
