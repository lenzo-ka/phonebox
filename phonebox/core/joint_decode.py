"""
Joint Viterbi decode: letters in, phones out, using the unit n-gram LM.

Pure Python — no FST. Scores complete unit sequences, including their end transition, in
one pass instead of greedy ``segment_letters`` + per-unit prediction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from .multigram_align import EPS, Unit
from .multigram_lm import SOS, MultigramLM, decode_unit_id, unit_id


def _index_units(q: dict[Unit, float], max_l: int) -> dict[str, list[Unit]]:
    """Index supported units by first letter, preserving candidate order."""
    index: dict[str, list[Unit]] = {}
    for (letters, phones), probability in q.items():
        if probability <= EPS or not 1 <= len(letters) <= max_l:
            continue
        index.setdefault(letters[0], []).append((letters, phones))
    return index


# DP state is a tuple of the last ``order - 1`` unit ids ending at a position;
# this is exactly the history the n-gram LM conditions on. Each backpointer
# records the unit id consumed to reach it so the path can be reconstructed
# even for the unigram case (where the state is always empty).
_State = tuple[str, ...]
_Cell = tuple[float, int, _State, str]  # (score, back_position, back_state, uid)


def validate_decode_beam(beam: int) -> int:
    """Validate the explicit hypothesis beam: zero is exact, positive approximate."""
    if type(beam) is not int or beam < 0:
        raise ValueError("decode beam must be a nonnegative integer")
    return beam


def joint_decode(
    letters: list[str],
    q: dict[Unit, float],
    lm: MultigramLM,
    max_letter_span: int,
    beam: int = 0,
) -> list[str] | None:
    """Best phone sequence for a cooked letter-token list.

    Dynamic programming over positions ``0 … len(letters)``. The state is the
    last ``lm.order - 1`` unit ids (the n-gram history), so the LM's full order
    is used rather than silently backing off. Each transition consumes a
    multigram unit from ``q`` whose letter side matches the next span.
    Complete paths also score the LM end-of-sequence transition. Alignment
    probabilities determine candidate support, not an additional path weight.

    Args:
        beam: Zero searches exactly. A positive value expands only the best
            ``beam`` histories per position, an approximation that can discard
            the optimal pronunciation. Higher LM orders retain longer histories
            and can create many more states; choose a beam explicitly if needed.
    """
    validate_decode_beam(beam)
    n = len(letters)
    if n == 0:
        return []

    return _decode_prepared(letters, _index_units(q, max_letter_span), lm, beam)


def _decode_prepared(
    letters: list[str],
    index: Mapping[str, Sequence[Unit]],
    lm: MultigramLM | None,
    beam: int,
    unit_ids: Mapping[Unit, str] | None = None,
    edge_score: Callable[[int, Unit], float] | None = None,
) -> list[str] | None:
    """Decode using ordered prepared candidates and an owned language model."""
    n = len(letters)
    if not n:
        return []
    ctx_width = lm.order - 1 if lm is not None else 0

    # best[i][state] = (log_score, back_position, back_state, uid_ending_here)
    best: list[dict[_State, _Cell]] = [{} for _ in range(n + 1)]
    best[0][()] = (0.0, -1, (), SOS)

    for i in range(n):
        layer = best[i]
        if not layer:
            continue
        if beam > 0:
            layer = dict(
                sorted(layer.items(), key=lambda item: item[1][0], reverse=True)[:beam]
            )
        for state, (score, _prev_i, _prev_state, _uid) in layer.items():
            history = list(state)
            for unit in index.get(letters[i], ()):
                (L, P) = unit
                j = i + len(L)
                if j > n or list(L) != letters[i : i + len(L)]:
                    continue
                uid = unit_ids[unit] if unit_ids is not None else unit_id(unit)
                new_state = (state + (uid,))[-ctx_width:] if ctx_width else ()
                new_score = score
                if lm is not None:
                    new_score += lm.log_prob_unit_id(uid, history)
                if edge_score is not None:
                    new_score += edge_score(i, unit)
                    if new_score == -math.inf:
                        continue
                prev = best[j].get(new_state)
                if prev is None or new_score > prev[0]:
                    best[j][new_state] = (new_score, i, state, uid)

    if not best[n]:
        return None
    end_state = max(
        best[n],
        key=lambda s: (
            best[n][s][0] + (lm.log_end_prob(list(s)) if lm is not None else 0.0)
        ),
    )

    phones: list[str] = []
    i, state = n, end_state
    while i > 0:
        _score, prev_i, prev_state, uid = best[i][state]
        _L, P = decode_unit_id(uid)
        phones = list(P) + phones
        if prev_i < 0:
            break
        i, state = prev_i, prev_state

    return phones


__all__ = ["joint_decode", "validate_decode_beam"]
