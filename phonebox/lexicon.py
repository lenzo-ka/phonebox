"""Cycle-free parsing primitives for pronunciation lexicon lines."""

from __future__ import annotations

import re

_STRESS_PATTERNS = {
    "cmu": re.compile(r"[012]"),
    "xsampa": re.compile(r'["%]'),
}


def parse_dict_line(line: str) -> tuple[str, list[str]] | None:
    """Parse a whitespace-separated word and pronunciation with comments."""
    line = line.strip()
    if not line or line.startswith(";;;") or line.startswith("#"):
        return None
    if "#" in line:
        line = line.split("#", 1)[0].strip()
    parts = line.split()
    if len(parts) < 2:
        return None
    return re.sub(r"\(\d+\)$", "", parts[0]), parts[1:]


def strip_phone_stress(phone: str, phoneset: str) -> str:
    """Strip stress defined by a known phoneset; preserve unknown tags."""
    pattern = _STRESS_PATTERNS.get(phoneset)
    return pattern.sub("", phone) if pattern is not None else phone
