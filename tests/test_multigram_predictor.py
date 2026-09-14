"""Reusable prediction snapshots preserve inference while isolating mutations."""

from itertools import product

import pytest

from phonebox import MultigramG2P, MultigramPredictor
from phonebox.core.multigram_lm import unit_id
from phonebox.core.vectorizer import Vectorizer


def model(order=2, beam=0):
    a = (("a",), ("AA1",))
    b = (("a",), ("ɛ̃",))
    silent = (("b",), ())
    joined = (("a", "b"), ("t͡ʃ",))
    result = MultigramG2P(max_letter_span=2, lm_order=order, decode_beam=beam)
    result.aligner.q = {a: 0.25, b: 0.25, silent: 0.25, joined: 0.25}
    result.lm.train([[a, silent], [b], [joined], [a, a]], vocabulary=result.aligner.q)
    return result


@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize("beam", [0, 1, 3])
def test_snapshot_exactly_matches_live_predictions(order, beam):
    source = model(order, beam)
    prepared = source.prepare_predictor()
    assert isinstance(prepared, MultigramPredictor)
    for size in range(5):
        for letters in product("ab", repeat=size):
            word = "".join(letters)
            assert prepared.pronounce(word) == source.pronounce(word)
    assert prepared.pronounce("unknown") == []


def test_snapshot_is_independent_of_q_lm_and_beam_changes():
    source = model()
    before = source.prepare_predictor()
    expected = before.pronounce("ab")
    source.aligner.q.clear()
    source.lm.train([])
    source.decode_beam = -1
    assert before.pronounce("ab") == expected
    with pytest.raises(RuntimeError, match="not trained"):
        source.prepare_predictor()
    source.lm = model().lm
    source.decode_beam = 0
    assert source.pronounce("ab") == []
    assert source.prepare_predictor().pronounce("ab") == []
    assert before.pronounce("ab") == expected


def test_snapshot_ties_preserve_original_candidate_order():
    a = (("x",), ("A",))
    b = (("x",), ("B",))
    source = MultigramG2P(lm_order=1)
    source.aligner.q = {a: 0.5, b: 0.5}
    source.lm.train([[a], [b]])
    prepared = source.prepare_predictor()
    assert prepared.pronounce("x") == ["A"]
    source.aligner.q = {b: 0.5, a: 0.5}
    assert source.pronounce("x") == ["B"]
    assert source.prepare_predictor().pronounce("x") == ["B"]
    assert prepared.pronounce("x") == ["A"]


def test_snapshot_owns_preprocessing_and_exception_values():
    source = model()
    source.set_preprocessor(Vectorizer(locale="en_US", spelling_rewrites={"z": "a"}))
    source.use_dict_fallback = True
    source.exceptions = {"special": ["EXCEPTION"]}
    prepared = source.prepare_predictor()
    expected = prepared.pronounce("z")
    assert expected == source.pronounce("z")
    source.preprocessor.spelling_rewrites["z"] = "b"
    assert prepared.pronounce("z") == expected
    assert source.pronounce("z") != expected
    source.preprocessor = Vectorizer(locale="en_US", spelling_rewrites={"z": "b"})
    source.exceptions["special"].append("CHANGED")
    source.use_dict_fallback = False
    assert prepared.pronounce("z") == expected
    assert prepared.pronounce("special") == ["EXCEPTION"]
    returned = prepared.pronounce("special")
    returned.append("NO ALIAS")
    assert prepared.pronounce("special") == ["EXCEPTION"]
    assert source.prepare_predictor().pronounce("special") != ["EXCEPTION"]


def test_snapshot_survives_retraining_and_export_load(tmp_path):
    source = model()
    prepared = source.prepare_predictor()
    expected = prepared.pronounce("ab")
    source.train_from_pairs([(list("a"), ["NEW"])])
    assert source.pronounce("a") == ["NEW"]
    assert prepared.pronounce("ab") == expected
    path = tmp_path / "model.g2p"
    source.export(path)
    restored = MultigramG2P.load(path)
    assert restored.prepare_predictor().pronounce("a") == ["NEW"]


def test_encoded_unit_scoring_matches_unit_api_and_checks_support():
    source = model(8)
    history: list[str] = []
    for unit in source.aligner.q:
        uid = unit_id(unit)
        assert source.lm.log_prob_unit_id(uid, history) == source.lm.log_prob(
            unit, history
        )
        history.append(uid)
    with pytest.raises(ValueError, match="outside"):
        source.lm.log_prob_unit_id("unknown", [])


def test_cli_and_evaluation_reuse_prepared_predictor(tmp_path, monkeypatch, capsys):
    from phonebox.cli.main import main
    from phonebox.eval.g2p_compare import run_compare

    source = model()
    source.set_preprocessor(Vectorizer(locale="en_US"))
    path = tmp_path / "model.g2p"
    source.export(path)
    calls = []
    original = MultigramG2P.prepare_predictor

    def prepare(instance):
        calls.append(instance)
        return original(instance)

    def forbidden_live_call(instance, word):
        raise AssertionError("batch consumer called live model per word")

    monkeypatch.setattr(MultigramG2P, "prepare_predictor", prepare)
    monkeypatch.setattr(MultigramG2P, "pronounce", forbidden_live_call)
    monkeypatch.setattr(
        "sys.argv", ["phonebox", "pronounce", "a", "ab", "-m", str(path)]
    )
    assert main() in (None, 0)
    assert len(calls) == 1
    assert capsys.readouterr().out == "a\tɛ̃\nab\tt͡ʃ\n"
    lexicon = tmp_path / "toy.dict"
    lexicon.write_text(
        "".join(
            f"{word} a b\n"
            for word in [
                "ab",
                "ac",
                "ad",
                "ae",
                "af",
                "ag",
                "ah",
                "ai",
                "aj",
                "ak",
            ]
        )
    )
    calls.clear()
    report = run_compare(
        lexicon=lexicon,
        locale="en_US",
        skip_baseline=True,
        em_iterations=2,
        no_config_joins=True,
        test_fraction=0.3,
    )
    assert len(calls) == 1
    assert report
