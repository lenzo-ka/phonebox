"""Tests for joint Viterbi decode."""

from __future__ import annotations

import pytest

from phonebox.core.joint_decode import joint_decode
from phonebox.core.multigram_align import MultigramAligner, Unit
from phonebox.core.multigram_lm import MultigramLM


def _train_toy():
    pairs = [
        (list("chat"), ["tʃ", "a", "t"]),
        (list("chin"), ["tʃ", "i", "n"]),
        (list("rich"), ["r", "i", "tʃ"]),
        (list("cat"), ["k", "a", "t"]),
        (list("cap"), ["k", "a", "p"]),
    ]
    aligner = MultigramAligner(
        max_letter_span=2,
        max_phone_span=1,
        min_phone_span=1,
        max_iterations=25,
    )
    aligner.fit(pairs)
    units_list = [aligner.viterbi_align(L, P) for L, P in pairs]
    lm = MultigramLM(order=2)
    lm.train([u for u in units_list if u])
    return aligner, lm


def test_joint_decode_chat():
    aligner, lm = _train_toy()
    phones = joint_decode(list("chat"), aligner.q, lm, max_letter_span=2)
    assert phones == ["tʃ", "a", "t"]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_complete_path_scores_end_transition(order, roundtrip):
    continued: Unit = (("x",), ("A",))
    terminal: Unit = (("x",), ("B",))
    following: Unit = (("y",), ("C",))
    lm = MultigramLM(order=order)
    lm.train([[continued, following]] * 10 + [[terminal]] * 3)
    if roundtrip:
        lm = MultigramLM.from_dict(lm.to_dict())
    # The common prefix wins without EOS, but its training paths continue.
    assert lm.log_prob(continued, []) > lm.log_prob(terminal, [])
    assert joint_decode(["x"], {continued: 0.5, terminal: 0.5}, lm, 1) == ["B"]


def test_unigram_end_transition_does_not_change_path_order():
    a: Unit = (("x",), ("A",))
    b: Unit = (("x",), ("B",))
    lm = MultigramLM(order=1)
    lm.train([[a]] * 3 + [[b]])
    assert joint_decode(["x"], {a: 0.5, b: 0.5}, lm, 1) == ["A"]
