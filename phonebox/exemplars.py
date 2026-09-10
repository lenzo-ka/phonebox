"""Read the packaged ICU locale exemplar inventory without ICU dependencies."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib.resources import files
from typing import Any

from .locale_resolution import canonical_locale

_FORMAT_VERSION = 2
_KINDS = ("standard", "auxiliary")


@dataclass(frozen=True)
class ExemplarInventory:
    """A compact set of exemplar characters and multi-codepoint strings."""

    characters: str
    ranges: tuple[tuple[int, int], ...]
    strings: tuple[str, ...]

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str) or not value:
            return False
        if len(value) != 1:
            return value in self.strings
        if value in self.characters:
            return True
        point = ord(value)
        index = bisect_right(self.ranges, (point, 0x10FFFF)) - 1
        return index >= 0 and self.ranges[index][0] <= point <= self.ranges[index][1]

    def __len__(self) -> int:
        return (
            len(self.characters)
            + len(self.strings)
            + sum(end - start + 1 for start, end in self.ranges)
        )

    def __iter__(self) -> Iterator[str]:
        yield from self.characters
        for start, end in self.ranges:
            yield from map(chr, range(start, end + 1))
        yield from self.strings

    def iter_ranges(self) -> Iterator[tuple[int, int]]:
        """Iterate inclusive code-point ranges without expanding them."""
        return iter(self.ranges)


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    resource = files("phonebox.config").joinpath("exemplars.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    if data.get("format") != _FORMAT_VERSION:
        raise RuntimeError(
            "unsupported exemplar inventory format; regenerate it with the pinned dev tools"
        )
    if data.get("kinds") != list(_KINDS):
        raise RuntimeError("invalid exemplar inventory kinds; regenerate the artifact")
    return data


def supported_locales() -> tuple[str, ...]:
    """Return every locale ID represented in the pinned ICU inventory."""
    return tuple(_data()["locales"])


@cache
def get_exemplars(locale: str, kind: str = "standard") -> ExemplarInventory:
    """Return one exact locale inventory; locale lookup has no fallback."""
    if kind not in _KINDS:
        raise ValueError(
            f"unknown exemplar kind {kind!r}; expected standard or auxiliary"
        )
    canonical = canonical_locale(locale)
    if canonical not in _data()["locales"]:
        raise KeyError(f"unknown exemplar locale: {locale}")
    data = _data()
    profile = data["profiles"][data["locales"][canonical]]
    raw = data["inventories"][profile[_KINDS.index(kind)]]
    return ExemplarInventory(
        raw["c"],
        tuple((start, end) for start, end in raw["r"]),
        tuple(raw["s"]),
    )


__all__ = ["ExemplarInventory", "get_exemplars", "supported_locales"]
