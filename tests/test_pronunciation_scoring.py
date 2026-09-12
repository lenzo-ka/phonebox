"""Sequence likelihood uses complete ordered saved-model emissions."""

import json

import pytest

from phonebox import G2P, PronunciationScore, Vectorizer, train_g2p
from phonebox.cli.main import main
from phonebox.pronunciation_scoring import score_sequence


def test_order_repetition_and_complete_consumption():
    v = Vectorizer(locale="en", phoneset_name="cmu")

    def expand(label: str) -> list[str]:
        return v.uncook([label])

    dists = [{"K": 1.0}, {"AE1": 1.0}, {"T": 1.0}]
    result = score_sequence(dists, ["K", "AE1", "T"], expand)
    assert result.score == result.probability == 1.0
    assert result.log_probability == 0.0 and result.supported
    for phones in (
        ["T", "AE1", "K"],
        ["K", "K", "K"],
        ["K", "AE1"],
        ["K", "AE1", "T", "K"],
    ):
        score = score_sequence(dists, phones, expand)
        assert (
            score.score == 0 and not score.supported and score.log_probability is None
        )


def test_sum_equivalent_epsilon_paths_join_expansion_and_deterministic_leaves():
    v = Vectorizer(locale="en", phoneset_name="cmu")

    def expand(label: str) -> list[str]:
        return v.uncook([label])

    result = score_sequence([{v.epsilon: 0.5, "A": 0.5}] * 2, ["A"], expand)
    assert result.probability == pytest.approx(0.5)
    assert result.score == pytest.approx(0.5**0.5)
    assert (
        score_sequence(
            [v.epsilon, "K" + v.join_char + "S"], ["K", "S"], expand
        ).probability
        == 1
    )
    assert score_sequence([], [], expand).supported is False


def test_underflow_remains_supported_and_json_finite():
    result = score_sequence([{"A": 0.001}] * 200, ["A"] * 200, lambda label: [label])
    assert result.supported and result.probability == 0.0
    assert result.log_probability is not None and result.log_probability < -1000
    assert result.score == pytest.approx(0.001)
    assert (
        json.loads(json.dumps(result.to_dict(), allow_nan=False))["supported"] is True
    )


@pytest.mark.parametrize("method", ["arithmetic", "min", "harmonic", "typo"])
def test_invalid_method_rejected(method):
    with pytest.raises(ValueError, match="method"):
        score_sequence([], [], lambda label: [label], method)


@pytest.mark.parametrize("suffix", [".g2p.gz", ".jsonl", ".cart"])
@pytest.mark.parametrize("remove_stress", [False, True])
def test_real_training_and_saved_score_fidelity(tmp_path, suffix, remove_stress):
    dictionary = tmp_path / "tiny.dict"
    dictionary.write_text(
        "cat K AE1 T\nbat B AE1 T\ncap K AE1 P\ncab K AE1 B\ncat(2) T AE1 K\n"
    )
    dt = train_g2p(
        dictionary,
        locale="en",
        phoneset="cmu",
        width=1,
        prune=False,
        remove_stress=remove_stress,
        use_dict_fallback=False,
    ).model
    dt.exceptions["cat"] = ["NOT", "MODEL"]
    correct = dt.score_pronunciation_details("cat", ["K", "AE1", "T"])
    reverse = dt.score_pronunciation_details("cat", ["T", "AE1", "K"])
    assert correct.probability > reverse.probability > 0
    assert isinstance(correct, PronunciationScore)
    assert correct.positions == 3
    assert correct.phones == ("K", "AE" if remove_stress else "AE1", "T")
    output = tmp_path / ("model" + suffix)
    dt.export(str(output))
    loaded = G2P(model=output, use_dict_fallback=True)
    saved = loaded.score_pronunciation_details("cat", ["K", "AE1", "T"])
    assert loaded._dt.vectorizer.width == 1
    assert saved.positions == correct.positions and saved.phones == correct.phones
    assert saved.probability == pytest.approx(correct.probability, rel=1e-6)
    assert loaded.score_pronunciation("cat", ["K", "AE1", "T"]) == saved.score
    assert loaded.score_pronunciation_details("cat", ["K", "K", "K"]).supported is False
    assert (
        loaded.score_pronunciation("cat", ["K", "AE1", "T"], method="product")
        == saved.probability
    )


def test_cli_numeric_json_and_mg_capability_error(tmp_path, capsys):
    dictionary = tmp_path / "tiny.dict"
    dictionary.write_text("cat K AE1 T\nbat B AE1 T\n")
    output = tmp_path / "model.g2p.gz"
    train_g2p(dictionary, locale="en", phoneset="cmu", prune=False, output=output)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps({"word": "cat", "prons": ["K AE1 T", "T AE1 K"]}) + "\n"
    )
    assert main(["score-prons", str(candidates), "-m", str(output)]) == 0
    values = json.loads(capsys.readouterr().out)["prons"]
    assert values == {"K AE1 T": 1.0, "T AE1 K": 0.0}
    stem = tmp_path / "mg"
    (tmp_path / "mg.units.json").write_text("{}")
    assert main(["score-prons", str(candidates), "-m", str(stem)]) == 2
    assert "MultigramG2P scoring is unsupported" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".g2p.gz", ".cart"])
def test_saved_letter_rules_rewrites_and_phone_expansion(tmp_path, monkeypatch, suffix):
    from phonebox import DecisionTree

    dictionary = tmp_path / "italian.dict"
    dictionary.write_text("caffè k a f f ɛ\n", encoding="utf-8")
    dt = DecisionTree(
        locale="it",
        phoneset_name="ipa",
        width=1,
        parallel_align=False,
        use_dict_fallback=False,
    )
    dt.vectorizer.spelling_rewrites = {"c": "g"}
    dt.train_from_dict(str(dictionary))
    before = dt.score_pronunciation_details("caffè", ["k", "a", "f", "f", "ɛ"])
    assert before.supported
    output = tmp_path / ("model" + suffix)
    dt.export(str(output))
    monkeypatch.setattr(
        Vectorizer, "_load_transliterators", lambda self, locale_dir: None
    )
    loaded = G2P(model=output, use_dict_fallback=False)
    assert loaded._dt.vectorizer.cook_letters("caffè", g2p=True) == [
        "g",
        "a",
        "f",
        "f",
        "ɛ",
    ]
    after = loaded.score_pronunciation_details("caffè", ["k", "a", "f", "f", "ɛ"])
    assert after.probability == pytest.approx(before.probability, rel=1e-6)


def test_unknown_method_fails_before_untrained_inference():
    from phonebox import DecisionTree

    with pytest.raises(ValueError, match="method"):
        DecisionTree(locale="en").score_pronunciation_details("cat", ["K"], "min")


def test_batch_orders_underflowed_products_by_log_probability():
    from phonebox.pronunciation_analysis import score_pronunciations

    class DetailedScorer:
        def score_pronunciation_details(self, word, phones, method="geometric"):
            return score_sequence(
                [{"A": 0.001 if phones[0] == "A" else 0.0001}] * 200,
                ["A"] * 200,
                lambda label: [label],
                method,
            )

    assert list(
        score_pronunciations(DetailedScorer(), "word", ["B", "A"], "product")
    ) == ["A", "B"]


@pytest.mark.parametrize("suffix", [".g2p.gz", ".jsonl", ".cart"])
def test_unknown_letter_policy_agrees_across_saved_prediction_surfaces(
    tmp_path, suffix
):
    dictionary = tmp_path / "tiny.dict"
    dictionary.write_text("a A\nb B\n")
    trained = train_g2p(
        dictionary,
        locale="en",
        phoneset="cmu",
        width=1,
        prune=False,
        use_dict_fallback=False,
    ).model
    output = tmp_path / ("model" + suffix)
    trained.export(str(output))
    loaded = G2P(model=output, use_dict_fallback=False)._dt
    for model in (trained, loaded):
        assert model.pronounce("z") == []
        assert model.pronounce_with_confidence("z") == ([], [])
        assert model.pronounce_nbest("z") == [([], 1.0)]
        assert not model.score_pronunciation_details("z", ["B"]).supported
        assert model.score_pronunciation_details("z", []).probability == 1
        assert model.score_pronunciation_details("az", ["A"]).probability == 1
