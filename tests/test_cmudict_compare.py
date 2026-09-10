import json

import pytest

from phonebox.cli.main import main
from phonebox.eval.cmudict_compare import render_markdown, validate_cmudict
from phonebox.eval.g2p_compare import evaluate
from phonebox.experiments.split import split_lexicon_by_key


def test_word_group_split_prevents_cooked_identity_leakage():
    pairs = [
        ("READ", ["R", "IY1", "D"]),
        ("read", ["R", "EH1", "D"]),
        ("reed", ["R", "IY1", "D"]),
        ("red", ["R", "EH1", "D"]),
    ]
    test, train = split_lexicon_by_key(
        pairs, key=str.lower, seed=4, test_fraction=0.5, max_test=2
    )
    assert {word.lower() for word, _ in test}.isdisjoint(
        word.lower() for word, _ in train
    )
    assert len(test) + len(train) == len(pairs)


def test_variant_per_uses_same_best_gold_policy_for_predictor():
    metrics = evaluate(
        "model",
        lambda _word: ["R", "EH", "D"],
        [("read", ["R", "IY", "D"])],
        gold_variants={"read": {("R", "IY", "D"), ("R", "EH", "D")}},
    )
    assert metrics["per_pct"] > 0
    assert metrics["per_variant_pct"] == 0
    assert metrics["wer_relaxed_pct"] == 0

    insertion_metrics = evaluate(
        "model",
        lambda _word: ["A", "B", "C"],
        [("x", ["A"])],
        gold_variants={"x": {("A",)}},
    )
    assert insertion_metrics["per_variant_edits"] == 2
    assert insertion_metrics["per_variant_reference_phones"] == 1
    assert insertion_metrics["per_variant_pct"] == 200
    assert insertion_metrics["per_reference_pct"] == 200


def test_validate_cmudict_rejects_unpinned_content(tmp_path):
    path = tmp_path / "cmudict.dict"
    path.write_text("word W ER D\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_cmudict(path)


def test_renderer_reads_metrics_from_json_snapshot():
    result = {
        "cmudict": {
            "repository": "https://example.test/cmudict",
            "commit": "abc",
            "sha256": "def",
            "license": "https://example.test/license",
        },
        "parameters": {
            "seed": 7,
            "test_fraction": 0.1,
            "max_test_words": 10,
            "em_iterations": 2,
            "max_letter_span": 2,
            "max_phone_span": 2,
        },
        "phonebox": {"revision": "123", "dirty": False, "code_sha256": "456"},
        "runtime": {
            "implementation": "CPython",
            "python": "3.12",
            "platform": "test",
            "dependencies": {"cartlet": "1", "icukit": "1"},
        },
        "conditions": [
            {
                "stress": "preserved",
                "train_words": 90,
                "test_words": 10,
                "models": [
                    {
                        "model": "G2PDecisionTree",
                        "train_seconds": 1.25,
                        "artifact_bytes": 42,
                        "wer_pct": 20.0,
                        "wer_relaxed_pct": 10.0,
                        "per_pct": 5.0,
                        "per_reference_pct": 5.0,
                        "per_variant_pct": 4.0,
                    }
                ],
            }
        ],
    }
    assert json.loads(json.dumps(result)) == result
    markdown = render_markdown(result)
    assert "| preserved | G2PDecisionTree | 90 | 10 | 1.25 | 42 |" in markdown
    assert "snapshot, not a promise" in markdown


def test_cli_invalid_input_is_status_two_without_traceback(tmp_path, capsys):
    assert (
        main(
            [
                "compare",
                "cmudict",
                "--refresh",
                str(tmp_path / "result.json"),
                "--lexicon",
                str(tmp_path / "missing.dict"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "CMUdict comparison failed" in captured.err
    assert "Traceback" not in captured.err
