"""Public convergence history follows the actual CART alignment loop."""

import json

import pytest

from phonebox.core.em_align import EMAlign
from phonebox.core.vectorizer import Vectorizer


def _aligner():
    model = EMAlign(
        locale=Vectorizer(phoneset_name="ipa"), parallel=False, verbose=False
    )
    model.load_prondict(["cat k a t", "cap k a p", "chat tʃ a t"])
    return model


def test_actual_alignment_history_is_serializable_reset_and_defensive():
    model = _aligner()
    assert model.alignment_history == []
    model.align()
    history = model.alignment_history
    assert history
    assert history[-1]["changed"] == 0
    assert [row["iteration"] for row in history] == list(range(1, len(history) + 1))
    for row in history:
        assert row["ratio"] == row["changed"] / len(model.em_data)
    json.dumps(history, allow_nan=False)
    history[0]["changed"] = -1
    assert model.alignment_history[0]["changed"] >= 0
    model.align(init=False)
    assert model.alignment_history == [{"iteration": 1, "changed": 0, "ratio": 0.0}]
    model.em_data = []
    with pytest.raises(ValueError, match="No admissible"):
        model.align(init=False)
    assert model.alignment_history == []


def test_capped_history_records_observed_changes_without_claiming_convergence(
    monkeypatch,
):
    model = _aligner()
    model.initialize()
    model.max_iterations = 2
    monkeypatch.setattr(model, "align_once", lambda iteration: (1, 0.5))
    model.align(init=False)
    assert model.alignment_history == [
        {"iteration": 1, "changed": 1, "ratio": 0.5},
        {"iteration": 2, "changed": 1, "ratio": 0.5},
    ]
