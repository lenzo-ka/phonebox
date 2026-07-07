"""Tests for unit n-gram LM."""

from __future__ import annotations

from typing import cast

from phonebox.core.multigram_lm import MultigramLM, unit_id


def test_lm_bigram_prefers_seen_bigram():
    seqs = [[(("c", "h"), ("tʃ",)), (("a",), ("a",)), (("t",), ("t",))]] * 20
    lm = MultigramLM(order=2, add_k=0.01)
    lm.train(cast(list[list[tuple[tuple[str, ...], tuple[str, ...]]]], seqs))
    uid_ch = unit_id((("c", "h"), ("tʃ",)))
    log_after_ch = lm.log_prob((("a",), ("a",)), [uid_ch])
    log_cold = lm.log_prob((("a",), ("a",)), [])
    assert log_after_ch > log_cold
