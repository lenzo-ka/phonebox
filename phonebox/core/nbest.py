#!/usr/bin/env python
"""
N-best and confidence scoring for g2p models.

Optional feature: Models can store probability distributions at leaves
to provide confidence scores and n-best lists.
"""

from __future__ import annotations

from ..constants import DEFAULT_NBEST_COUNT, NBEST_TOP_K_PER_POSITION


def generate_nbest(
    distributions: list[dict[str, float]], n: int = DEFAULT_NBEST_COUNT
) -> list[tuple[list[str], float]]:
    """
    Generate n-best pronunciations from per-phoneme distributions.

    Args:
        distributions: List of dicts (one per letter position)
        n: Number of alternatives to return

    Returns:
        List of (pronunciation, score) tuples, sorted by score

    Example:
        distributions = [
            {"HH": 0.9, "K": 0.1},
            {"AH": 0.8, "EH": 0.2},
            {"L": 0.95, "R": 0.05},
        ]

        Returns top n combinations with scores
    """
    if not distributions:
        return []

    # Handle mix of string leaves (deterministic) and dict leaves (probabilistic)
    normalized_dists = []
    for dist in distributions:
        if isinstance(dist, str):
            normalized_dists.append({dist: 1.0})
        elif isinstance(dist, dict):
            normalized_dists.append(dist)
        else:
            normalized_dists.append({})

    # Beam search over positions. Each position multiplies the running score by
    # a probability <= 1, so a partial hypothesis can never overtake a
    # higher-scoring one as it is extended. Keeping the top-n partials at each
    # step therefore yields the exact top-n full pronunciations in O(P * n * k)
    # time, instead of materialising all k^P combinations.
    k = min(NBEST_TOP_K_PER_POSITION, n)

    beam: list[tuple[list[str], float]] = [([], 1.0)]
    for dist in normalized_dists:
        if not dist:
            continue
        top_k = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
        extended = [
            (phones + [phone], score * prob)
            for phones, score in beam
            for phone, prob in top_k
        ]
        extended.sort(key=lambda x: x[1], reverse=True)
        beam = extended[:n]

    return beam
