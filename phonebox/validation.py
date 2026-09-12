"""Inspect a pronunciation lexicon against a caller-supplied phone inventory.

Validation compares raw tokens as well as their NFC equivalence. It does not
cook spellings, remove stress, or modify either input.
"""

from __future__ import annotations

import unicodedata as ud
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .constants import DICT_ENCODING
from .lexicon import parse_dict_line


@dataclass(frozen=True)
class PhoneUsage:
    """One NFC-equivalent phone group with all raw forms and occurrence counts."""

    normalized: str
    raw_counts: tuple[tuple[str, int], ...]
    words: tuple[str, ...]

    @property
    def occurrences(self) -> int:
        """Total occurrences across every raw normalization form."""
        return sum(count for _, count in self.raw_counts)


@dataclass(frozen=True)
class NormalizationMismatch:
    """An absent raw lexicon token whose NFC form exists in the inventory."""

    lexicon_phone: str
    spec_forms: tuple[str, ...]
    occurrences: int


@dataclass(frozen=True)
class LexiconValidationResult:
    """Exact input counts, inventory gaps, and bounded deterministic samples."""

    entries: int
    unique_words: int
    unique_pronunciations: int
    spec_entries: int
    spec_distinct_phones: int
    phones: tuple[PhoneUsage, ...]
    normalization_mismatches: tuple[NormalizationMismatch, ...]
    missing_phones: tuple[PhoneUsage, ...]
    unused_phones: tuple[str, ...]
    non_nfc_word_entries: int
    non_nfc_words: tuple[str, ...]

    @property
    def distinct_phones(self) -> int:
        """Number of distinct raw phone tokens in the lexicon."""
        return sum(len(phone.raw_counts) for phone in self.phones)

    @property
    def duplicate_entries(self) -> int:
        """Parsed entries repeating the same base word and raw pronunciation."""
        return self.entries - self.unique_pronunciations

    def has_errors(self, *, strict: bool = False) -> bool:
        """Whether normalization gaps or strict missing-phone checks fail."""
        return bool(self.normalization_mismatches or (strict and self.missing_phones))


def validate_lexicon(
    lines: Iterable[str], phoneset: Iterable[str], *, show_words: int = 5
) -> LexiconValidationResult:
    """Validate raw dictionary lines using the shared variant/comment parser.

    ``phoneset`` supplies nonempty, whitespace-free phone strings. NFC gaps
    always count as errors; missing phones are warnings unless ``strict=True``
    is requested on the result. Unused phones and non-NFC words are information.
    ``show_words`` bounds unique word samples for each phone group and the word
    normalization finding; totals are never truncated. Invalid specifications
    or a negative/noninteger sample limit raise ``ValueError``.
    """
    if (
        isinstance(show_words, bool)
        or not isinstance(show_words, int)
        or show_words < 0
    ):
        raise ValueError("show_words must be a nonnegative integer")
    if isinstance(phoneset, (str, bytes)):
        raise ValueError("phoneset must be an iterable of phone strings")
    spec = tuple(phoneset)
    if any(
        not isinstance(p, str) or not p or any(c.isspace() for c in p) for p in spec
    ):
        raise ValueError(
            "phoneset elements must be nonempty phone strings without whitespace"
        )
    spec_raw = set(spec)
    spec_groups: dict[str, set[str]] = {}
    for p in spec:
        spec_groups.setdefault(ud.normalize("NFC", p), set()).add(p)
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    words: set[str] = set()
    pronunciations: set[tuple[str, tuple[str, ...]]] = set()
    non_nfc_samples: list[str] = []
    entries = non_nfc_entries = 0
    for line in lines:
        parsed = parse_dict_line(line)
        if parsed is None:
            continue
        word, phones = parsed
        entries += 1
        words.add(word)
        pronunciations.add((word, tuple(phones)))
        if ud.normalize("NFC", word) != word:
            non_nfc_entries += 1
            if len(non_nfc_samples) < show_words and word not in non_nfc_samples:
                non_nfc_samples.append(word)
        for p in phones:
            counts[p] += 1
            group_samples = samples.setdefault(ud.normalize("NFC", p), [])
            if len(group_samples) < show_words and word not in group_samples:
                group_samples.append(word)
    grouped: dict[str, list[tuple[str, int]]] = {}
    mismatches = []
    for p, count in sorted(counts.items()):
        nfc = ud.normalize("NFC", p)
        grouped.setdefault(nfc, []).append((p, count))
        if p not in spec_raw and nfc in spec_groups:
            mismatches.append(
                NormalizationMismatch(p, tuple(sorted(spec_groups[nfc])), count)
            )
    usages = tuple(
        PhoneUsage(nfc, tuple(raw), tuple(samples[nfc]))
        for nfc, raw in sorted(grouped.items())
    )
    return LexiconValidationResult(
        entries,
        len(words),
        len(pronunciations),
        len(spec),
        len(spec_raw),
        usages,
        tuple(mismatches),
        tuple(p for p in usages if p.normalized not in spec_groups),
        tuple(sorted(p for p in spec_raw if ud.normalize("NFC", p) not in grouped)),
        non_nfc_entries,
        tuple(non_nfc_samples),
    )


def validate_lexicon_file(
    lexicon: str | Path, phoneset: Iterable[str], *, show_words: int = 5
) -> LexiconValidationResult:
    """Validate a UTF-8 pronunciation file; filesystem errors propagate."""
    with Path(lexicon).open(encoding=DICT_ENCODING) as lines:
        return validate_lexicon(lines, phoneset, show_words=show_words)


def format_lexicon_validation(result: LexiconValidationResult) -> str:
    """Render findings without printing or deciding a command exit status."""
    lines = [
        f"lexicon: {result.entries} entries, {result.unique_words} words, "
        f"{result.distinct_phones} distinct phones; spec: {result.spec_entries} phones",
        "",
    ]
    if result.normalization_mismatches:
        lines.append(
            f"!! NFC/NFD MISMATCH ({len(result.normalization_mismatches)} raw phones): normalize both inputs consistently."
        )
        for mismatch in result.normalization_mismatches:
            lines.append(
                f"   lex {mismatch.lexicon_phone!r} ({mismatch.occurrences} uses) vs spec {mismatch.spec_forms!r}"
            )
    if result.missing_phones:
        lines.append(
            f"!! {len(result.missing_phones)} phones used in lexicon but missing from canonical spec:"
        )
        for phone in result.missing_phones:
            samples = f"  e.g. {', '.join(phone.words)}" if phone.words else ""
            lines.append(
                f"   {phone.normalized!r} ({phone.occurrences} uses; raw {phone.raw_counts!r}){samples}"
            )
    if result.unused_phones:
        lines.append(
            f"-- {len(result.unused_phones)} unused spec phones (informational): {result.unused_phones!r}"
        )
    if result.non_nfc_word_entries:
        lines.append(
            f"-- {result.non_nfc_word_entries} lexicon word entries are not NFC (informational): {result.non_nfc_words!r}"
        )
    if not result.normalization_mismatches and not result.missing_phones:
        lines.append("OK: lexicon and spec agree, no NFC/NFD mismatches.")
    return "\n".join(lines)


__all__ = [
    "PhoneUsage",
    "NormalizationMismatch",
    "LexiconValidationResult",
    "validate_lexicon",
    "validate_lexicon_file",
    "format_lexicon_validation",
]
