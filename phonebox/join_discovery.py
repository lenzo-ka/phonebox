"""Discover asymmetric spelling/phone joins from raw dictionary tokens."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import DICT_ENCODING, FILE_ENCODING
from .core.multigram_align import MultigramAligner
from .core.vectorizer import Vectorizer
from .lexicon import parse_dict_line
from .utils.io import paths_refer_to_same_file


@dataclass(frozen=True)
class JoinCandidate:
    """A join's mass and its most probable counterpart."""

    tokens: tuple[str, ...]
    total_mass: float
    modal_match: tuple[str, ...]
    modal_confidence: float
    alternatives: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class JoinDiscoveryResult:
    """Ranked candidates plus effective settings and EM history."""

    lexicon: Path
    n_entries: int
    settings: dict[str, Any]
    loglik_history: list[float]
    letter_joins: list[JoinCandidate]
    phone_joins: list[JoinCandidate]
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible report used by the CLI."""
        return {
            "lexicon": str(self.lexicon),
            "n_entries": self.n_entries,
            "settings": self.settings,
            "loglik_history": self.loglik_history,
            "letter_joins": [asdict(candidate) for candidate in self.letter_joins],
            "phone_joins": [asdict(candidate) for candidate in self.phone_joins],
        }


def load_join_pairs(path: str | Path) -> list[tuple[list[str], list[str]]]:
    """Read raw spelling/phone tokens with the shared dictionary grammar."""
    pairs = []
    with Path(path).open(encoding=DICT_ENCODING) as stream:
        for line in stream:
            parsed = parse_dict_line(line)
            if parsed is not None:
                word, phones = parsed
                if word and phones:
                    pairs.append((list(word), phones))
    return pairs


def _aggregate(side: dict, min_prob: float, top: int) -> list[JoinCandidate]:
    candidates = []
    for tokens, counterparts in side.items():
        total = sum(counterparts.values())
        if total < min_prob:
            continue
        ranked = sorted(counterparts.items(), key=lambda item: (-item[1], item[0]))
        modal, mass = ranked[0]
        candidates.append(
            JoinCandidate(
                tokens,
                total,
                modal,
                mass / total if total > 0 else 0.0,
                tuple(
                    {"tokens": list(other), "mass": probability}
                    for other, probability in ranked[1:5]
                ),
            )
        )
    candidates.sort(key=lambda candidate: (-candidate.total_mass, candidate.tokens))
    return candidates[:top]


def discover_joins(
    lexicon: str | Path,
    *,
    locale: str | None = None,
    output: str | Path | None = None,
    max_letter_span: int | None = None,
    max_phone_span: int | None = None,
    min_phone_span: int = 0,
    min_prob: float = 0.001,
    top: int = 50,
    sample: int = 0,
    sample_seed: int = 42,
    max_iterations: int = 15,
    convergence_threshold: float = 1e-4,
    parallel_align: bool = False,
    verbose: bool = False,
) -> JoinDiscoveryResult:
    """Fit raw-token EM and rank joins; locale supplies only span defaults.

    No locale grapheme/phone cooking is applied, so suggestions can discover
    joins before config joins compress the input. No files are written unless
    output is supplied.
    """
    lexicon = Path(lexicon)
    if output is not None:
        if not str(output).strip():
            raise ValueError("output must be a nonempty report path")
        if paths_refer_to_same_file(lexicon, output):
            raise ValueError("join report must differ from the input dictionary")
    if sample < 0 or top < 0:
        raise ValueError("sample and top must be nonnegative")
    vectorizer = Vectorizer(locale=locale, phoneset_name="ipa") if locale else None
    config = vectorizer.multigram_config() if vectorizer else {}
    max_l = (
        max_letter_span
        if max_letter_span is not None
        else config.get("max_letter_span", 3)
    )
    max_p = (
        max_phone_span
        if max_phone_span is not None
        else config.get("max_phone_span", 2)
    )
    if max_l <= 0 or max_p <= 0 or max_iterations <= 0:
        raise ValueError("spans and EM iteration count must be positive")
    if min_phone_span < 0 or min_phone_span > max_p:
        raise ValueError("minimum phone span must be between zero and maximum")
    pairs = load_join_pairs(lexicon)
    if sample and len(pairs) > sample:
        pairs = random.Random(sample_seed).sample(pairs, sample)
    if not pairs:
        raise ValueError("no trainable dictionary entries")
    aligner = MultigramAligner(
        max_letter_span=max_l,
        max_phone_span=max_p,
        min_phone_span=min_phone_span,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
        parallel=parallel_align,
    )
    aligner.fit(pairs)
    letters: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    phones: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for (letter_tokens, phone_tokens), probability in aligner.q.items():
        if len(letter_tokens) > len(phone_tokens):
            letters[letter_tokens][phone_tokens] = probability
        if len(phone_tokens) > len(letter_tokens):
            phones[phone_tokens][letter_tokens] = probability
    result = JoinDiscoveryResult(
        lexicon,
        len(pairs),
        {
            "locale": vectorizer.locale if vectorizer else None,
            "max_letter_span": max_l,
            "max_phone_span": max_p,
            "min_phone_span": min_phone_span,
            "min_prob": min_prob,
            "top": top,
            "max_iterations": max_iterations,
            "iterations_run": len(aligner.loglik_history),
            "sample": sample or None,
            "sample_seed": sample_seed if sample else None,
        },
        list(aligner.loglik_history),
        _aggregate(letters, min_prob, top),
        _aggregate(phones, min_prob, top),
        Path(output) if output is not None else None,
    )
    if result.output_path is not None:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.output_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding=FILE_ENCODING,
        )
    return result
