"""Tests for unit n-gram LM."""

from __future__ import annotations

import math
from typing import cast

import pytest

from phonebox.core.multigram_lm import EOS, SOS, MultigramLM, unit_id


def test_lm_bigram_prefers_seen_bigram():
    seqs = [[(("c", "h"), ("tʃ",)), (("a",), ("a",)), (("t",), ("t",))]] * 20
    lm = MultigramLM(order=2, add_k=0.01)
    lm.train(cast(list[list[tuple[tuple[str, ...], tuple[str, ...]]]], seqs))
    uid_ch = unit_id((("c", "h"), ("tʃ",)))
    log_after_ch = lm.log_prob((("a",), ("a",)), [uid_ch])
    log_cold = lm.log_prob((("a",), ("a",)), [])
    assert log_after_ch > log_cold


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_prediction_event_probabilities_sum_to_one(order, roundtrip):
    a = (("a",), ("A",))
    b = (("b",), ("B",))
    silent = (("h",), ())  # q-supported but never chosen in a Viterbi path.
    lm = MultigramLM(order=order)
    lm.train([[a, b], [a], [b]], vocabulary=[silent])
    if roundtrip:
        lm = MultigramLM.from_dict(lm.to_dict())
    for history in (
        [],
        [unit_id(a)],
        [unit_id(a), unit_id(b)],
        [unit_id(silent)],
        [unit_id(silent), unit_id(a)],
    ):
        mass = sum(math.exp(lm.log_prob(u, history)) for u in (a, b, silent))
        mass += math.exp(lm.log_end_prob(history))
        assert mass == pytest.approx(1.0)
    assert lm.supports_unit(silent)
    assert SOS not in lm.to_dict()["events"]
    assert EOS in lm.to_dict()["events"]
    with pytest.raises(ValueError, match="outside the declared"):
        lm.log_prob((("z",), ("Z",)), [])


def test_old_lm_scoring_format_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="unsupported multigram LM scoring version"):
        MultigramLM.from_dict({"order": 2, "add_k": 0.1, "vocab": []})


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_bi",
        "unknown_tri",
        "sos_target",
        "eos_context",
        "negative",
        "fractional",
        "inconsistent",
    ],
)
def test_loaded_counts_must_match_declared_event_space(mutation):
    a = (("a",), ("A",))
    lm = MultigramLM()
    lm.train([[a]])
    data = lm.to_dict()
    uid = unit_id(a)
    if mutation == "unknown_bi":
        data["bi"][SOS + "\tunknown"] = 1
    elif mutation == "unknown_tri":
        data["tri"][SOS + "\t" + uid + "\tunknown"] = 1
    elif mutation == "sos_target":
        data["bi"][uid + "\t" + SOS] = 1
    elif mutation == "eos_context":
        data["bi"][EOS + "\t" + uid] = 1
    elif mutation == "negative":
        data["uni"][uid] = -1
    elif mutation == "fractional":
        data["uni"][uid] = 1.5
    else:
        data["bi"][SOS + "\t" + uid] += 1
    with pytest.raises(ValueError, match="multigram LM"):
        MultigramLM.from_dict(data)


def test_untrained_end_score_is_neutral():
    assert MultigramLM().log_end_prob([]) == 0.0
