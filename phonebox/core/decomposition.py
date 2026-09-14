"""Posterior-weighted joint-unit targets on ordinary CART letter positions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .multigram_align import EPS, MultigramAligner, _enum_outgoing
from .vectorizer import Vectorizer


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def decomposition_targets(letters, phones, aligner, vectorizer):
    """Return per-position label posteriors, or None without a supported path.

    Gold phones constrain every path. Scores are products of alignment q,
    conditional on both sequences, not downstream joint-LM probabilities.
    Support and the probability floor match q-Viterbi. Log arithmetic avoids
    underflow in the complete-path partition function.
    """
    letters, phones = tuple(letters), tuple(phones)
    if not letters:
        return None
    if not vectorizer.join_char:
        raise ValueError("decomposition targets require a phone join character")
    if any(
        not p
        or p == vectorizer.epsilon
        or vectorizer.join_char in p
        or any(c.isspace() for c in p)
        for p in phones
    ):
        raise ValueError("phone tokens contain reserved decomposition target syntax")
    edges = defaultdict(list)
    for i, j, length, width, unit in _enum_outgoing(
        letters, phones, aligner.max_l, aligner.max_p, aligner.min_p
    ):
        probability = aligner.q.get(unit, 0.0)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("invalid joint-unit probability")
        if probability > EPS:
            target = (
                vectorizer.join_char.join(unit[1]) if unit[1] else vectorizer.epsilon
            )
            # Validate the actual runtime codec, without recooking targets.
            if vectorizer.uncook([target]) != list(unit[1]):
                raise ValueError("decomposition target does not round trip")
            edges[i, j].append((i + length, j + width, math.log(probability), target))
    n, m = len(letters), len(phones)
    alpha = [[-math.inf] * (m + 1) for _ in range(n + 1)]
    beta = [[-math.inf] * (m + 1) for _ in range(n + 1)]
    alpha[0][0] = 0.0
    for i in range(n):
        for j in range(m + 1):
            if alpha[i][j] == -math.inf:
                continue
            for end_i, end_j, score, _ in edges[i, j]:
                alpha[end_i][end_j] = _logadd(alpha[end_i][end_j], alpha[i][j] + score)
    partition = alpha[n][m]
    if partition == -math.inf:
        return None
    beta[n][m] = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(m + 1):
            for end_i, end_j, score, _ in edges[i, j]:
                beta[i][j] = _logadd(beta[i][j], score + beta[end_i][end_j])
    targets: list[defaultdict[str, float]] = [defaultdict(float) for _ in letters]
    for (i, j), outgoing in edges.items():
        for end_i, end_j, score, target in outgoing:
            weight = math.exp(alpha[i][j] + score + beta[end_i][end_j] - partition)
            if not weight:
                continue
            targets[i][target] += weight
            for position in range(i + 1, end_i):
                targets[position][vectorizer.epsilon] += weight
    for distribution in targets:
        total = math.fsum(distribution.values())
        if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("decomposition posterior does not conserve letter mass")
        # Remove only floating-point accumulation drift.
        for target in distribution:
            distribution[target] /= total
    return [dict(distribution) for distribution in targets]


@dataclass
class DecompositionPreparation:
    X: list[list[str]]
    y: list[str]
    counts: list[float]
    admitted_pairs: list
    metadata: dict


def prepare_decomposition_vectors(
    pairs, vectorizer: Vectorizer, aligner: MultigramAligner
):
    """Prepare cooked pairs using a fitted aligner; repeated pairs retain weight.

    Letter contexts are the same cooked positions used in runtime inference.
    Callers must freeze preprocessing consistently; static joins still remove
    their corresponding unmerged alternatives before this function sees data.
    """
    pairs = list(pairs)
    unique_pairs = Counter((tuple(letters), tuple(phones)) for letters, phones in pairs)
    counts: defaultdict[tuple[tuple[str, ...], str], float] = defaultdict(float)
    vector_cache = {}
    admitted = []
    skipped: defaultdict[str, int] = defaultdict(int)
    ambiguous_positions = 0
    for (letters, phones), multiplicity in unique_pairs.items():
        try:
            targets = decomposition_targets(letters, phones, aligner, vectorizer)
        except ValueError as error:
            if "reserved decomposition target syntax" not in str(error):
                raise
            skipped["reserved_phone_syntax"] += multiplicity
            continue
        if targets is None:
            skipped["unsupported_alignment"] += multiplicity
            continue
        admitted.extend([[list(letters), list(phones)] for _ in range(multiplicity)])
        if letters not in vector_cache:
            vector_cache[letters] = list(
                vectorizer.next_letter_vector(list(letters), g2p=True, cooked=True)
            )
        vectors = vector_cache[letters]
        for vector, distribution in zip(vectors, targets, strict=True):
            ambiguous_positions += multiplicity * (len(distribution) > 1)
            for target, weight in distribution.items():
                counts[tuple(vector), target] += multiplicity * weight
    if not admitted:
        raise ValueError("no supported decomposition training pairs")
    X, y, weights = [], [], []
    for (vector, target), weight in counts.items():
        X.append(list(vector))
        y.append(target)
        weights.append(weight)
    metadata = {
        "alignment_method": "decomposition-posterior",
        "objective": "q posterior conditioned on gold letters and phones",
        "projection": "left-anchor; epsilon at remaining unit letters",
        "search": "all supported paths; no beam",
        "supplied_entries": len(pairs),
        "retained_entries": len(admitted),
        "skipped_entries": sum(skipped.values()),
        "skip_reasons": dict(skipped),
        "ambiguous_positions": ambiguous_positions,
        "unit_inventory": len(aligner.q),
        "unique_source_pairs": len(unique_pairs),
        "context_spellings": len(vector_cache),
        "alignment_history": aligner.loglik_history,
        "alignment_cap": aligner.max_iter,
        "alignment_threshold": aligner.conv,
        "max_letter_span": aligner.max_l,
        "max_phone_span": aligner.max_p,
        "min_phone_span": aligner.min_p,
        "min_unit_mass": aligner.min_unit_mass,
        "unit_probability_floor": EPS,
        "letter_preprocessing": vectorizer.export_letter_preprocessing(),
    }
    return DecompositionPreparation(X, y, weights, admitted, metadata)
