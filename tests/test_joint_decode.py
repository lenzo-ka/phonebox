"""Tests for joint Viterbi decode."""

from __future__ import annotations

from phonebox.core.joint_decode import joint_decode
from phonebox.core.multigram_align import MultigramAligner
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
