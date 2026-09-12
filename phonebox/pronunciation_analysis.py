"""Reusable scoring and triage for candidate pronunciations.

Suspicious-entry classifications are lightweight English/CMUdict heuristics
for prioritizing human review, not a language detector or correctness verdict.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .pronunciation_scoring import ScoreMethod

Category = Literal["REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK"]
CATEGORIES: tuple[Category, ...] = ("REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK")


class PronunciationScorer(Protocol):
    """Minimal scoring interface accepted by the batch helpers."""

    def score_pronunciation(
        self, word: str, phones: list[str], method: str = "geometric"
    ) -> float: ...


@dataclass(frozen=True)
class ZeroScoreResult:
    word: str
    pronunciations: Mapping[str, float | str]
    zero_pronunciations: tuple[str, ...]


@dataclass(frozen=True)
class LowScoreResult:
    score: float
    word: str
    pronunciations: Mapping[str, float | str]


@dataclass(frozen=True)
class ScoreGapResult:
    ratio: float
    word: str
    pronunciations: Mapping[str, float | str]


@dataclass(frozen=True)
class TriageResult:
    category: Category
    score: float
    word: str
    pronunciation: str
    reason: str


def strip_stress(pronunciation: str) -> list[str]:
    """Remove CMU stress digits from a space-separated pronunciation."""
    return re.sub(r"[012]", "", pronunciation).split()


def score_pronunciations(
    scorer: PronunciationScorer,
    word: str,
    pronunciations: Sequence[str],
    method: ScoreMethod = "geometric",
) -> dict[str, float]:
    """Score variants, retaining original spellings in descending score order."""
    scored = {
        pron: scorer.score_pronunciation(word, pron.split(), method=method)
        for pron in pronunciations
    }
    return dict(sorted(scored.items(), key=lambda item: -item[1]))


def score_entries(
    scorer: PronunciationScorer,
    entries: Iterable[Mapping[str, Any]],
    method: ScoreMethod = "geometric",
) -> Iterator[dict[str, Any]]:
    """Score JSONL-style entries while preserving their other fields."""
    for entry in entries:
        result = dict(entry)
        result["prons"] = score_pronunciations(
            scorer, str(entry["word"]), list(entry["prons"]), method
        )
        yield result


def find_zeros(entries: Iterable[Mapping[str, Any]]) -> Iterator[ZeroScoreResult]:
    """Yield entries containing at least one zero-score pronunciation."""
    for entry in entries:
        prons = entry["prons"]
        zeros = tuple(p for p, score in prons.items() if float(score) == 0)
        if zeros:
            yield ZeroScoreResult(str(entry["word"]), prons, zeros)


def find_low_scores(
    entries: Iterable[Mapping[str, Any]], threshold: float, min_len: int = 5
) -> Iterator[LowScoreResult]:
    """Yield alphabetic words whose variants all score below *threshold*."""
    for entry in entries:
        word = str(entry["word"])
        if len(word) < min_len or not re.fullmatch(r"[a-z]+", word, re.I):
            continue
        prons = entry["prons"]
        max_score = max(float(value) for value in prons.values())
        if max_score < threshold:
            yield LowScoreResult(max_score, word, prons)


def find_score_gaps(
    entries: Iterable[Mapping[str, Any]],
    high_threshold: float = 0.1,
    low_threshold: float = 0.01,
) -> Iterator[ScoreGapResult]:
    """Yield multi-variant entries spanning the low and high thresholds."""
    for entry in entries:
        prons = entry["prons"]
        if len(prons) < 2:
            continue
        scores = [float(score) for score in prons.values()]
        high, low = max(scores), min(scores)
        if high > high_threshold and low < low_threshold:
            yield ScoreGapResult(
                high / low if low > 0 else float("inf"), str(entry["word"]), prons
            )


def classify_variant(
    word: str, pronunciation: str, score: float, best_score: float
) -> tuple[Category, str]:
    """Classify one low-scoring variant using English/CMUdict heuristics."""
    phones = pronunciation.split()
    letters = re.sub(r"[^a-z]", "", word.lower())
    if score == 0:
        return "ABBREV", "spelled out (0 score)"
    if len(letters) <= 4 and len(phones) > len(letters) * 1.5:
        return "ABBREV", "spelled out"
    if word.lower() in {
        "mr",
        "mr.",
        "dr",
        "dr.",
        "jr",
        "jr.",
        "sr",
        "sr.",
        "st",
        "st.",
    }:
        return "ABBREV", "title abbreviation"
    if word.lower() in {"of", "the", "to", "is", "was", "were", "have", "has", "had"}:
        return "FUNCTION", "function word"
    if re.search(r"^xia|^zh[aeiou]|iao$|^qia|^qiu", word, re.I):
        return "FOREIGN", "Chinese"
    if re.search(r"^mc[a-z]|^mac[a-z]", word, re.I):
        return "FOREIGN", "Irish/Scottish"
    if best_score > 0 and best_score > score:
        return "REVIEW", f"score={score:.3f} (best={best_score:.3f})"
    return "REVIEW", f"score={score:.3f}"


def triage_entries(
    entries: Iterable[Mapping[str, Any]], threshold: float = 0.05
) -> dict[Category, list[TriageResult]]:
    """Group scored variants using the documented English/CMUdict heuristics."""
    results: dict[Category, list[TriageResult]] = {
        category: [] for category in CATEGORIES
    }
    for entry in entries:
        word = str(entry["word"])
        if len(re.sub(r"[^a-z]", "", word.lower())) < 2:
            continue
        scores = [(pron, float(score)) for pron, score in entry["prons"].items()]
        best = max(score for _, score in scores)
        for pron, score in scores:
            if score >= threshold:
                category: Category = "OK"
                reason = "ok"
            else:
                category, reason = classify_variant(word, pron, score, best)
            results[category].append(TriageResult(category, score, word, pron, reason))
    for category, values in results.items():
        values.sort(
            key=(lambda value: (-value.score, value.word))
            if category == "OK"
            else (lambda value: (value.score, value.word))
        )
    return results


__all__ = [
    "CATEGORIES",
    "Category",
    "LowScoreResult",
    "PronunciationScorer",
    "ScoreGapResult",
    "ScoreMethod",
    "TriageResult",
    "ZeroScoreResult",
    "classify_variant",
    "find_low_scores",
    "find_score_gaps",
    "find_zeros",
    "score_entries",
    "score_pronunciations",
    "strip_stress",
    "triage_entries",
]
