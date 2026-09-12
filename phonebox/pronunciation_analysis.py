"""Reusable scoring and triage for candidate pronunciations.

Suspicious-entry classifications are lightweight English/CMUdict heuristics
for prioritizing human review, not a language detector or correctness verdict.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .dictionary import phone_mapping_transform
from .lexicon import LexiconEntry, parse_dict_entry
from .pronunciation_scoring import (
    PronunciationScore,
    ScoreMethod,
    validate_score_method,
)

Category = Literal["REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK"]
CATEGORIES: tuple[Category, ...] = ("REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK")


class PronunciationScorer(Protocol):
    """Minimal scoring interface accepted by the batch helpers."""

    def score_pronunciation_details(
        self, word: str, phones: list[str], method: str = "geometric"
    ) -> PronunciationScore: ...


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
    """Score variants by sequence log mass, retaining source-order ties.

    Requires detailed model scores so raw probability underflow cannot turn
    unequal supported likelihoods into false ties. Returned scores are numeric.
    """
    validate_score_method(method)
    details = {
        pron: scorer.score_pronunciation_details(word, pron.split(), method=method)
        for pron in pronunciations
    }
    ordered = sorted(
        details.items(),
        key=lambda item: (
            item[1].log_probability is None,
            -(item[1].log_probability or 0.0),
        ),
    )
    return {pron: result.score for pron, result in ordered}


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
        return "REVIEW", "no model support; cause undetermined"
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


def _likelihood_key(
    log_probability: float | None,
    positions: int,
    method: str,
    *,
    within_word: bool = False,
) -> float:
    """Retain mathematical likelihood order when exponentiation underflows."""
    if log_probability is None:
        return -math.inf
    return (
        log_probability
        if within_word or method == "product"
        else log_probability / positions
    )


@dataclass(frozen=True)
class LexiconReviewRecord:
    """One effective pronunciation with dense rank and all source occurrences."""

    word: str
    phones: tuple[str, ...]
    score: float
    probability: float
    log_probability: float | None
    supported: bool
    positions: int
    method: str
    rank: int
    origins: tuple[LexiconEntry, ...]

    @property
    def entry(self) -> str:
        """Dense dictionary label, independent of the source's variant suffix."""
        return self.entry_label()

    def entry_label(self, *, number_senses: bool = True) -> str:
        """Return the ranked label, optionally repeating the bare spelling."""
        return (
            self.word
            if not number_senses or self.rank == 1
            else f"{self.word}({self.rank})"
        )

    def to_dict(self, *, number_senses: bool = True) -> dict:
        """Return numeric values and origins, optionally without sense suffixes."""
        return {
            "word": self.word,
            "entry": self.entry_label(number_senses=number_senses),
            "phones": list(self.phones),
            "score": self.score,
            "probability": self.probability,
            "log_probability": self.log_probability,
            "supported": self.supported,
            "positions": self.positions,
            "method": self.method,
            "rank": self.rank,
            "origins": [origin.to_dict() for origin in self.origins],
        }


@dataclass(frozen=True)
class LexiconReviewResult:
    """Selected ranked records plus unfiltered population and effective method."""

    records: tuple[LexiconReviewRecord, ...]
    source_entries: int
    unique_variants: int
    method: str
    order: str
    filtered: bool

    def to_dict(self, *, number_senses: bool = True) -> dict:
        """Return strict JSON values with the selected entry-label presentation."""
        return {
            "source_entries": self.source_entries,
            "unique_variants": self.unique_variants,
            "method": self.method,
            "order": self.order,
            "filtered": self.filtered,
            "records": [
                record.to_dict(number_senses=number_senses) for record in self.records
            ],
        }


def review_lexicon(
    scorer: PronunciationScorer,
    lines: Iterable[str],
    *,
    method: str = "geometric",
    phone_mapping: Mapping[str, str | list[str]] | None = None,
    phone_transform: Callable[[list[str]], list[str]] | None = None,
    threshold: float | None = None,
    limit: int | None = None,
    order: str = "worst",
) -> LexiconReviewResult:
    """Score shared dictionary entries, deduplicate effective phones, then rank.

    Saved model preprocessing owns stress; an optional caller mapping/transform
    runs first. Groups retain exact base spelling and source order. Threshold
    and limit select already-ranked records, leaving dense ranks unchanged.
    This measures model compatibility, not correctness or semantic sense order.
    """
    validate_score_method(method)
    if not callable(getattr(scorer, "score_pronunciation_details", None)):
        raise ValueError(
            "lexicon review requires detailed CART scoring; multigram scoring is unsupported"
        )
    if order not in {"worst", "variants"}:
        raise ValueError("order must be worst or variants")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and between 0 and 1")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be a nonnegative integer")
    if phone_mapping is not None and phone_transform is not None:
        raise ValueError("choose phone_mapping or phone_transform, not both")
    if phone_mapping is not None:
        phone_transform = phone_mapping_transform(phone_mapping)
    groups: dict[
        str, dict[tuple[str, ...], tuple[PronunciationScore, list[LexiconEntry]]]
    ] = {}
    entries = 0
    for number, line in enumerate(lines, 1):
        origin = parse_dict_entry(line, line_number=number)
        if origin is None:
            continue
        entries += 1
        phones = list(origin.phones)
        if phone_transform is not None:
            phones = phone_transform(phones)
        if (
            not isinstance(phones, list)
            or not phones
            or any(
                not isinstance(p, str) or not p or any(c.isspace() for c in p)
                for p in phones
            )
        ):
            raise ValueError(
                "phone transform must return nonempty whitespace-free phone strings"
            )
        details = scorer.score_pronunciation_details(origin.word, phones, method=method)
        if any(not math.isfinite(v) for v in (details.score, details.probability)) or (
            details.log_probability is not None
            and not math.isfinite(details.log_probability)
        ):
            raise ValueError("scorer returned nonfinite values")
        effective = tuple(details.phones)
        if any(
            not isinstance(p, str) or not p or any(c.isspace() for c in p)
            for p in effective
        ):
            raise ValueError("effective phones must be nonempty whitespace-free tokens")
        group = groups.setdefault(origin.word, {})
        if effective in group:
            group[effective][1].append(origin)
        else:
            group[effective] = (details, [origin])
    ranked = []
    for word, candidates in groups.items():
        ordered = sorted(
            candidates.items(),
            key=lambda item: (
                -_likelihood_key(
                    item[1][0].log_probability,
                    item[1][0].positions,
                    method,
                    within_word=True,
                )
            ),
        )
        for rank, (ranked_phones, (details, origins)) in enumerate(ordered, 1):
            ranked.append(
                LexiconReviewRecord(
                    word,
                    ranked_phones,
                    details.score,
                    details.probability,
                    details.log_probability,
                    details.supported,
                    details.positions,
                    method,
                    rank,
                    tuple(origins),
                )
            )
    if not entries:
        raise ValueError("no pronunciation entries to review")
    total = len(ranked)
    if order == "worst":
        ranked.sort(
            key=lambda record: (
                _likelihood_key(record.log_probability, record.positions, method),
                record.origins[0].line_number,
            )
        )
    if threshold is not None:
        ranked = [record for record in ranked if record.score < threshold]
    if limit is not None:
        ranked = ranked[:limit]
    return LexiconReviewResult(
        tuple(ranked),
        entries,
        total,
        method,
        order,
        threshold is not None or limit is not None,
    )


def review_lexicon_file(
    scorer: PronunciationScorer, path: str | Path, **options
) -> LexiconReviewResult:
    """Review a UTF-8 pronunciation dictionary without modifying it."""
    with Path(path).open(encoding="utf-8") as lines:
        return review_lexicon(scorer, lines, **options)


def format_lexicon_review(
    result: LexiconReviewResult,
    *,
    format: str = "tsv",
    header: bool = True,
    number_senses: bool = True,
) -> Iterator[str]:
    """Yield data-only formats with optional ranked pronunciation suffixes.

    number_senses changes entry labels only; ranks, ordering and origins remain.
    """
    if format == "dict" and (result.filtered or result.order != "variants"):
        raise ValueError("dictionary format requires unfiltered variants order")
    if format == "dict" and any(
        not record.word or not record.phones for record in result.records
    ):
        raise ValueError(
            "dictionary format requires nonempty words and effective pronunciations"
        )
    if format == "json":
        yield json.dumps(
            result.to_dict(number_senses=number_senses),
            ensure_ascii=False,
            allow_nan=False,
        )
    elif format == "jsonl":
        for record in result.records:
            yield json.dumps(
                record.to_dict(number_senses=number_senses),
                ensure_ascii=False,
                allow_nan=False,
            )
    elif format in {"tsv", "dict"}:
        if format == "tsv" and header:
            yield "score\tentry\tphones\tword\trank\tlog_probability\tstatus\tsource_lines"
        for record in result.records:
            phones = " ".join(record.phones)
            if format == "dict":
                fields = [record.entry_label(number_senses=number_senses), phones]
            else:
                fields = [
                    repr(record.score),
                    record.entry_label(number_senses=number_senses),
                    phones,
                    record.word,
                    str(record.rank),
                    ""
                    if record.log_probability is None
                    else repr(record.log_probability),
                    "supported" if record.supported else "unsupported",
                    ",".join(str(origin.line_number) for origin in record.origins),
                ]
            if any("\t" in field or "\n" in field or "\r" in field for field in fields):
                raise ValueError("review fields must not contain tabs or newlines")
            yield "\t".join(fields)
    else:
        raise ValueError("format must be tsv, json, jsonl, or dict")


__all__ += [
    "LexiconReviewRecord",
    "LexiconReviewResult",
    "review_lexicon",
    "review_lexicon_file",
    "format_lexicon_review",
]
