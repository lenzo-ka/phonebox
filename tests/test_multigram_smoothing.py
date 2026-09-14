"""Finite positive smoothing must remain scorable at floating-point boundaries."""

import math
import sys
from decimal import Decimal, localcontext

import pytest

from phonebox.core.multigram_lm import MultigramLM, unit_id

A = (("a",), ("A",))
B = (("b",), ())


def reference(count, total, k, size=3):
    with localcontext() as context:
        context.prec = 90
        smoothing = Decimal.from_float(k)
        return float(
            ((Decimal(count) + smoothing) / (Decimal(total) + smoothing * size)).ln()
        )


@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize(
    "k", [math.ulp(0.0), 1e-322, sys.float_info.min, 0.1, 1e308, sys.float_info.max]
)
def test_scores_match_decimal_oracle_and_normalize_after_reload(order, k):
    lm = MultigramLM(order=order, add_k=k)
    lm.train([[A]] * 100, vocabulary=[A, B])
    lm = MultigramLM.from_dict(lm.to_dict())
    for history in ([], [unit_id(A)], [unit_id(B)], ["unknown"] * 9):
        # These paths have only SOS->A->EOS. Unknown histories back off to
        # unigram counts, while order>=2 can use either observed context.
        if order > 1 and not history:
            counts, total = (100, 0, 0), 100
        elif order > 1 and history == [unit_id(A)]:
            counts, total = (0, 0, 100), 100
        else:
            counts, total = (100, 0, 100), 200
        scores = [
            lm.log_prob(A, history),
            lm.log_prob(B, history),
            lm.log_end_prob(history),
        ]
        for score, count in zip(scores, counts, strict=True):
            assert math.isfinite(score)
            assert score <= 0
            assert score == pytest.approx(reference(count, total, k), rel=0, abs=3e-13)
        assert math.fsum(math.exp(score) for score in scores) == pytest.approx(
            1, rel=0, abs=3e-13
        )


def test_ordinary_range_retains_exact_previous_arithmetic():
    for k in (0.01, 0.1, 1.0, 100.0):
        lm = MultigramLM(order=2, add_k=k)
        lm.train([[A]] * 100, vocabulary=[A, B])
        assert lm.log_prob(A, []) == math.log((100 + k) / (100 + 3 * k))
        assert lm.log_prob(B, []) == math.log(k / (100 + 3 * k))
        assert lm.log_end_prob([]) == math.log(k / (100 + 3 * k))


def test_valid_serialized_integer_counts_beyond_float_range():
    lm = MultigramLM(order=8)
    lm.train([[A]], vocabulary=[A, B])
    payload = lm.to_dict()
    scale = 10**400
    for level in payload["counts"]:
        for record in level:
            record[1] *= scale
    restored = MultigramLM.from_dict(payload)
    assert restored.log_prob(B, []) == pytest.approx(
        reference(0, scale, restored.add_k), rel=0, abs=3e-13
    )
    assert restored.log_prob(A, []) == pytest.approx(0, rel=0, abs=3e-13)
