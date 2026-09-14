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


@pytest.mark.parametrize("order", range(1, 9))
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
    lm = MultigramLM(order=3)
    lm.train([[a]])
    data = lm.to_dict()
    uid = unit_id(a)
    if mutation == "unknown_bi":
        data["counts"][1].append([[SOS, "unknown"], 1])
    elif mutation == "unknown_tri":
        data["counts"][2].append([[SOS, uid, "unknown"], 1])
    elif mutation == "sos_target":
        data["counts"][1].append([[uid, SOS], 1])
    elif mutation == "eos_context":
        data["counts"][1].append([[EOS, uid], 1])
    elif mutation == "negative":
        data["counts"][0][0][1] = -1
    elif mutation == "fractional":
        data["counts"][0][0][1] = 1.5
    else:
        data["counts"][1][0][1] += 1
    with pytest.raises(ValueError, match="multigram LM"):
        MultigramLM.from_dict(data)


def test_untrained_end_score_is_neutral():
    assert MultigramLM().log_end_prob([]) == 0.0


@pytest.mark.parametrize("order", range(1, 9))
def test_only_requested_sparse_count_levels_are_allocated(order):
    unit = (("a",), ("A",))
    lm = MultigramLM(order=order)
    lm.train([[unit]])
    saved = lm.to_dict()
    assert len(saved["counts"]) == order
    assert all(not level for level in saved["counts"][3:])
    assert not {"uni", "bi", "tri"} & saved.keys()
    assert MultigramLM.from_dict(saved).to_dict() == saved


@pytest.mark.parametrize("order", [0, 9, -1, True, 2.5, "8", None])
def test_order_range_is_explicit_and_integral(order):
    with pytest.raises(ValueError, match="order must be an integer from 1 to 8"):
        MultigramLM(order=order)


def test_lower_order_normalized_scores_are_preserved():
    a = (("a",), ("A",))
    b = (("b",), ("B",))
    silent = (("h",), ())
    for order, probability in [(1, 2.1 / 7.4), (2, 2.1 / 3.4), (3, 2.1 / 3.4)]:
        lm = MultigramLM(order=order)
        lm.train([[a, b], [a], [b]], vocabulary=[silent])
        assert math.exp(lm.log_prob(a, [])) == pytest.approx(probability)


@pytest.mark.parametrize(
    "malformation",
    ["level_count", "duplicate", "width", "inconsistent_high_order", "old_version"],
)
def test_generalized_serialization_rejects_malformed_sparse_tables(malformation):
    unit = (("a",), ("A",))
    lm = MultigramLM(order=8)
    lm.train([[unit] * 9])
    data = lm.to_dict()
    if malformation == "level_count":
        data["counts"].pop()
    elif malformation == "duplicate":
        data["counts"][7].append(data["counts"][7][0])
    elif malformation == "width":
        data["counts"][7][0][0].pop()
    elif malformation == "inconsistent_high_order":
        data["counts"][7][0][1] += 1
    else:
        data["version"] = 2
    with pytest.raises(ValueError, match="multigram LM"):
        MultigramLM.from_dict(data)
