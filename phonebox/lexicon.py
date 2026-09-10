"""Cycle-free parsing primitives for pronunciation lexicon lines."""

from __future__ import annotations

import re


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
