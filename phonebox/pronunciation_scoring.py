"""Exact ordered pronunciation mass under CART position emissions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

ScoreMethod = Literal["geometric", "product"]
SCORE_METHODS: tuple[ScoreMethod, ...] = ("geometric", "product")


@dataclass(frozen=True)
class PronunciationScore:
    """Model compatibility, with finite log mass retained after underflow."""

    score: float
    probability: float
    log_probability: float | None
    positions: int
    phones: tuple[str, ...]
    supported: bool
    method: ScoreMethod

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON-compatible values, including null for no path."""
        return {
            "score": self.score,
            "probability": self.probability,
            "log_probability": self.log_probability,
            "positions": self.positions,
            "phones": list(self.phones),
            "supported": self.supported,
            "method": self.method,
        }


def validate_score_method(method: str) -> ScoreMethod:
    """Reject unsupported aggregation before preprocessing or model inference."""
    if method not in SCORE_METHODS:
        raise ValueError("method must be 'geometric' or 'product'")
    return cast(ScoreMethod, method)


def score_sequence(
    distributions: Sequence[str | dict[str, float]],
    phones: Sequence[str],
    expand: Callable[[str], list[str]],
    method: str = "geometric",
) -> PronunciationScore:
    """Sum complete matching paths, consuming one label per ordered position."""
    selected = validate_score_method(method)
    target = tuple(phones)
    positions = len(distributions)
    masses = {0: 0.0} if positions else {}
    for distribution in distributions:
        labels = {distribution: 1.0} if isinstance(distribution, str) else distribution
        following: dict[int, float] = {}
        for label, probability in labels.items():
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "model emission probabilities must be finite in [0, 1]"
                )
            if probability == 0.0:
                continue
            emission = tuple(expand(label))
            log_probability = math.log(probability)
            for offset, mass in masses.items():
                end = offset + len(emission)
                if target[offset:end] != emission:
                    continue
                candidate = mass + log_probability
                previous = following.get(end)
                if previous is None:
                    following[end] = candidate
                else:
                    high, low = max(previous, candidate), min(previous, candidate)
                    following[end] = high + math.log1p(math.exp(low - high))
        masses = following
    log_mass = masses.get(len(target))
    if log_mass is None:
        return PronunciationScore(0.0, 0.0, None, positions, target, False, selected)
    # Native CART stores probabilities as float32; allow only accumulation
    # rounding above unit mass, never report a probability greater than one.
    if log_mass > 1e-6 * positions:
        raise ValueError("model sequence probability exceeds one")
    log_mass = min(0.0, log_mass)
    probability = math.exp(log_mass)
    score = math.exp(log_mass / positions) if selected == "geometric" else probability
    return PronunciationScore(
        score, probability, log_mass, positions, target, True, selected
    )
