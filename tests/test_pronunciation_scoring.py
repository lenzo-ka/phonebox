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
