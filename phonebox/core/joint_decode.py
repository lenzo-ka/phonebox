"""
Joint Viterbi decode: letters in, phones out, using ``q`` + unit n-gram LM.

Pure Python — no FST. Scores full paths (segmentation + units + LM) in
one pass instead of greedy ``segment_letters`` + per-unit prediction.
"""

from __future__ import annotations

import math

from .multigram_align import EPS, Unit
from .multigram_lm import SOS, MultigramLM, decode_unit_id, unit_id


def _index_units(
    q: dict[Unit, float], max_l: int
) -> dict[str, list[tuple[Unit, float]]]:
    """Index scorable units by their first letter.

    Each decode position then inspects only the handful of units that could
    start there, instead of scanning the whole ``q`` table once per position.
    Units with non-positive probability or a letter span outside ``1..max_l``
    are dropped up front.
    """
    index: dict[str, list[tuple[Unit, float]]] = {}
    for (L, P), prob in q.items():
        if prob <= EPS or not 1 <= len(L) <= max_l:
            continue
        index.setdefault(L[0], []).append(((L, P), prob))
    return index


# DP state is a tuple of the last ``order - 1`` unit ids ending at a position;
# this is exactly the history the n-gram LM conditions on. Each backpointer
# records the unit id consumed to reach it so the path can be reconstructed
# even for the unigram case (where the state is always empty).
_State = tuple[str, ...]
_Cell = tuple[float, int, _State, str]  # (score, back_position, back_state, uid)


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

    Args:
        beam: If > 0, keep only the top-``beam`` hypotheses per position.
    """
    n = len(letters)
    if n == 0:
        return []

    index = _index_units(q, max_letter_span)
    ctx_width = lm.order - 1

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
            for unit, prob in index.get(letters[i], ()):
                (L, P) = unit
                j = i + len(L)
                if j > n or list(L) != letters[i : i + len(L)]:
                    continue
                uid = unit_id(unit)
                new_state = (state + (uid,))[-ctx_width:] if ctx_width else ()
                new_score = (
                    score + math.log(max(prob, EPS)) + lm.log_prob(unit, history)
                )
                prev = best[j].get(new_state)
                if prev is None or new_score > prev[0]:
                    best[j][new_state] = (new_score, i, state, uid)

    if not best[n]:
        return None
    end_state = max(best[n], key=lambda s: best[n][s][0])

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


__all__ = ["joint_decode"]
