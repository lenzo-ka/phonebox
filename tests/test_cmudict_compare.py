import json

import pytest

from phonebox.cli.main import main
from phonebox.eval.cmudict_compare import (
    _training_accounting,
    render_markdown,
    validate_cmudict,
)
from phonebox.eval.g2p_compare import evaluate, train_baseline, train_multigram
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


def test_evaluate_counts_prediction_failures_and_empty_outputs():
    def fail(_word):
        raise RuntimeError("broken predictor")

    failed = evaluate("model", fail, [("x", ["A"])], quiet=True)
    assert failed["prediction_errors"] == 1
    assert failed["empty_predictions"] == 1

    empty = evaluate("model", lambda _word: [], [("x", ["A"])], quiet=True)
    assert empty["prediction_errors"] == 0
    assert empty["empty_predictions"] == 1


def test_cart_helper_loads_aligned_vectors_once(monkeypatch):
    from phonebox.core.g2p_model import G2PDecisionTree

    calls = 0
    original = G2PDecisionTree.load_alignments

    def counted(self, infile=None):
        nonlocal calls
        calls += 1
        return original(self, infile)

    monkeypatch.setattr(G2PDecisionTree, "load_alignments", counted)
    train_baseline("en_US", "cmu", ["cat\tK AE1 T"])
    assert calls == 1


def test_training_accounting_exposes_asymmetric_eligibility():
    from phonebox.core.g2p_model import G2PDecisionTree

    cart = G2PDecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)
    cart.vectorizer.disable_config_joins()
    cart.load_prondict(iter(["x\tK S"]))
    assert len(cart.em.init_data) == 0
    multigram = train_multigram([(["x"], ["K", "S"])], 2, 2, 2)
    accounting = _training_accounting(cart, [(["x"], ["K", "S"])], multigram.metrics, 1)
    assert accounting["G2PDecisionTree"] == {
        "unique_cooked_candidates": 1,
        "retained_entries": 0,
        "skipped_after_dedup": 1,
    }
    assert accounting["MultigramG2P"] == {
        "unique_cooked_candidates": 1,
        "aligned_entries": 1,
        "skipped_entries": 0,
    }


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
                "training_accounting": {
                    "raw_entries_supplied_each": 91,
                    "G2PDecisionTree": {
                        "unique_cooked_candidates": 90,
                        "retained_entries": 89,
                        "skipped_after_dedup": 1,
                    },
                    "MultigramG2P": {
                        "unique_cooked_candidates": 90,
                        "aligned_entries": 90,
                        "skipped_entries": 0,
                    },
                },
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
                        "prediction_errors": 0,
                        "empty_predictions": 0,
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
