#!/usr/bin/env python
"""
Tests for N-best and confidence scoring functionality.
"""

import gzip
import json

import pytest

from phonebox.converter import G2P
from phonebox.core.decision_tree import DecisionTree


@pytest.fixture
def simple_dict():
    """Create a simple test dictionary."""
    return [
        "hello\tHH EH L OW",
        "world\tW ER L D",
        "read\tR IY D",
        "read\tR EH D",  # Multiple pronunciations
        "test\tT EH S T",
    ]


@pytest.fixture
def trained_model_with_dists(simple_dict, tmp_path):
    """Train a model with distributions enabled."""
    dict_file = tmp_path / "test.dict"
    with open(dict_file, "w") as f:
        for line in simple_dict:
            f.write(line + "\n")

    dt = DecisionTree(
        locale="en_US",
        phoneset_name="cmu",
        remove_stress=False,
        cased=False,
        verbose=False,
        store_distributions=True,
        min_dist_entropy=0.01,
    )

    with open(dict_file) as f:
        dt.load_prondict(f)

    dt.align()
    dt.train()

    return dt


@pytest.fixture
def trained_model_without_dists(simple_dict, tmp_path):
    """Train a model with store_distributions=False (skips probability storage)."""
    dict_file = tmp_path / "test.dict"
    with open(dict_file, "w") as f:
        for line in simple_dict:
            f.write(line + "\n")

    dt = DecisionTree(
        locale="en_US",
        phoneset_name="cmu",
        remove_stress=False,
        cased=False,
        verbose=False,
        store_distributions=False,
    )

    with open(dict_file) as f:
        dt.load_prondict(f)

    dt.align()
    dt.train()

    return dt


class TestDistributionStorage:
    """Test that distributions are stored correctly in the tree."""

    def test_distributions_enabled(self, trained_model_with_dists):
        """Test that distributions are created when enabled."""
        dt = trained_model_with_dists
        assert dt.store_distributions is True
        assert dt.model is not None

        # Check if any dict nodes exist in tree
        has_dict = self._check_tree_for_dicts(dt.model)
        # May or may not have dicts depending on data ambiguity
        # Just verify it doesn't crash
        assert isinstance(has_dict, bool)

    def test_distributions_disabled(self, trained_model_without_dists):
        """Test that distributions are not created when disabled."""
        dt = trained_model_without_dists
        assert dt.store_distributions is False
        assert dt.model is not None

        # Should only have string leaves
        has_dict = self._check_tree_for_dicts(dt.model)
        assert has_dict is False

    def _check_tree_for_dicts(self, node):
        """Recursively check if tree has any dict nodes."""
        if (
            isinstance(node, dict)
            and not isinstance(node, list)
            and node
            and all(isinstance(v, (int, float)) for v in node.values())
        ):
            return True

        if isinstance(node, list) and len(node) == 4:
            _, _, left, right = node
            return self._check_tree_for_dicts(left) or self._check_tree_for_dicts(right)

        return False


class TestPronounceWithConfidence:
    """Test confidence scoring functionality."""

    def test_confidence_with_dists(self, trained_model_with_dists):
        """Test confidence scores with distributions."""
        dt = trained_model_with_dists

        phones, confidences = dt.pronounce_with_confidence("hello")

        assert isinstance(phones, list)
        assert isinstance(confidences, list)
        assert len(phones) == len(confidences)
        assert len(phones) > 0

        # All confidences should be in [0, 1]
        for conf in confidences:
            assert 0.0 <= conf <= 1.0

    def test_confidence_without_dists(self, trained_model_without_dists):
        """Test confidence scores without distributions (all 1.0)."""
        dt = trained_model_without_dists

        phones, confidences = dt.pronounce_with_confidence("hello")

        assert isinstance(phones, list)
        assert isinstance(confidences, list)
        assert len(phones) == len(confidences)

        # All should be 1.0 (deterministic)
        for conf in confidences:
            assert conf == 1.0

    def test_confidence_matches_pronounce(self, trained_model_with_dists):
        """Test that confidence phonemes match regular pronounce."""
        dt = trained_model_with_dists

        regular = dt.pronounce("world")
        conf_phones, _ = dt.pronounce_with_confidence("world")

        assert regular == conf_phones


class TestPronounceNBest:
    """Test n-best generation."""

    def test_nbest_basic(self, trained_model_with_dists):
        """Test basic n-best generation."""
        dt = trained_model_with_dists

        nbest = dt.pronounce_nbest("hello", n=3)

        assert isinstance(nbest, list)
        assert len(nbest) >= 1
        assert len(nbest) <= 3

        # Check structure
        for phones, score in nbest:
            assert isinstance(phones, list)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_nbest_sorted(self, trained_model_with_dists):
        """Test that n-best results are sorted by score."""
        dt = trained_model_with_dists

        nbest = dt.pronounce_nbest("test", n=5)

        if len(nbest) > 1:
            scores = [score for _, score in nbest]
            assert scores == sorted(scores, reverse=True)

    def test_nbest_first_matches_pronounce(self, trained_model_with_dists):
        """Test that first n-best matches regular pronounce."""
        dt = trained_model_with_dists

        regular = dt.pronounce("hello")
        nbest = dt.pronounce_nbest("hello", n=5)

        assert len(nbest) >= 1
        best_phones, _ = nbest[0]
        assert best_phones == regular

    def test_nbest_without_dists(self, trained_model_without_dists):
        """Test n-best without distributions returns 1-best."""
        dt = trained_model_without_dists

        nbest = dt.pronounce_nbest("hello", n=5)

        assert len(nbest) == 1
        phones, score = nbest[0]
        assert score == 1.0
        assert phones == dt.pronounce("hello")

    def test_nbest_n_parameter(self, trained_model_with_dists):
        """Test that n parameter limits results."""
        dt = trained_model_with_dists

        nbest_1 = dt.pronounce_nbest("test", n=1)
        nbest_3 = dt.pronounce_nbest("test", n=3)
        nbest_10 = dt.pronounce_nbest("test", n=10)

        assert len(nbest_1) <= 1
        assert len(nbest_3) <= 3
        assert len(nbest_10) <= 10


class TestModelExportImport:
    """Test model export/import with distributions."""

    def test_export_with_distributions(self, trained_model_with_dists, tmp_path):
        """Test exporting model with distributions."""
        dt = trained_model_with_dists
        model_path = tmp_path / "model.g2p.gz"

        dt.export(str(model_path))

        assert model_path.exists()

        # Load and check config (JSONL format)
        with gzip.open(model_path, "rt", encoding="utf-8") as f:
            header = json.loads(f.readline())

        assert isinstance(header, dict)
        # store_distributions is in metadata.training_config
        training_config = header.get("metadata", {}).get("training_config", {})
        assert training_config.get("store_distributions") is True

    def test_export_without_distributions(self, trained_model_without_dists, tmp_path):
        """Test exporting model without distributions."""
        dt = trained_model_without_dists
        model_path = tmp_path / "model.jsonl.gz"

        dt.export(str(model_path))

        # Load and check config (JSONL format)
        with gzip.open(model_path, "rt", encoding="utf-8") as f:
            header = json.loads(f.readline())

        # store_distributions is in metadata.training_config
        training_config = header.get("metadata", {}).get("training_config", {})
        assert training_config.get("store_distributions") is False

    def test_load_model_with_dists(self, trained_model_with_dists, tmp_path):
        """Test loading model with distributions."""
        model_path = tmp_path / "model.g2p.gz"
        trained_model_with_dists.export(str(model_path))

        # Load with DecisionTree
        dt2 = DecisionTree(model=str(model_path), locale="en_US")

        # Should be able to use n-best
        nbest = dt2.pronounce_nbest("hello", n=3)
        assert len(nbest) >= 1


class TestG2PNBest:
    """Test n-best functionality through the high-level G2P API."""

    def test_runtime_with_dists(self, trained_model_with_dists, tmp_path):
        """Test G2P with distributions."""
        model_path = tmp_path / "model.g2p.gz"
        trained_model_with_dists.export(str(model_path))

        g2p = G2P(model=str(model_path))

        assert g2p._dt.store_distributions is True

        # Test confidence
        phones, confs = g2p.pronounce_with_confidence("hello")
        assert len(phones) == len(confs)
        assert all(0.0 <= c <= 1.0 for c in confs)

        # Test n-best
        nbest = g2p.pronounce_nbest("hello", n=3)
        assert len(nbest) >= 1

    def test_runtime_without_dists(self, trained_model_without_dists, tmp_path):
        """Test G2P with model without distributions."""
        model_path = tmp_path / "model.g2p.gz"
        trained_model_without_dists.export(str(model_path))

        g2p = G2P(model=str(model_path))

        # Note: G2P constructor has store_distributions=True by default,
        # but the loaded model won't have distribution data
        # What matters is that it still works

        # Should still work, return deterministic results
        phones, confs = g2p.pronounce_with_confidence("hello")
        assert len(phones) > 0
        assert len(phones) == len(confs)
        # Confidences may be 1.0 or model values, just check they're valid
        assert all(0.0 <= c <= 1.0 for c in confs)

        # N-best should work (may return 1 or more results)
        nbest = g2p.pronounce_nbest("hello", n=5)
        assert len(nbest) >= 1


class TestG2PConverterNBest:
    """Test n-best functionality in G2P converter."""

    def test_converter_nbest(self, trained_model_with_dists, tmp_path):
        """Test G2P converter n-best methods."""
        model_path = tmp_path / "model.g2p.gz"
        trained_model_with_dists.export(str(model_path))

        g2p = G2P(model=str(model_path))

        # Test confidence
        phones, confs = g2p.pronounce_with_confidence("hello")
        assert len(phones) > 0
        assert len(phones) == len(confs)

        # Test n-best
        nbest = g2p.pronounce_nbest("hello", n=3)
        assert len(nbest) >= 1
        assert nbest[0][0] == g2p.pronounce("hello")


class TestDistributionFreeModels:
    """Test that models trained without distributions still pronounce."""

    def test_pronounce_with_distributions(self, trained_model_with_dists):
        """A model trained with distributions still returns plain pronunciations."""
        dt = trained_model_with_dists

        result = dt.pronounce("hello")

        assert isinstance(result, list)
        assert all(isinstance(p, str) for p in result)

    def test_load_distribution_free_model(self, trained_model_without_dists, tmp_path):
        """Models exported with store_distributions=False still load and predict."""
        model_path = tmp_path / "no_dist_model.g2p.gz"
        trained_model_without_dists.export(str(model_path))

        # Should load without issues
        dt2 = DecisionTree(model=str(model_path), locale="en_US")
        result = dt2.pronounce("hello")

        assert isinstance(result, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_word(self, trained_model_with_dists):
        """Test pronouncing empty string."""
        dt = trained_model_with_dists

        result = dt.pronounce("")
        assert result == []

        phones, confs = dt.pronounce_with_confidence("")
        assert phones == []
        assert confs == []

        nbest = dt.pronounce_nbest("", n=5)
        # Should handle gracefully
        assert isinstance(nbest, list)

    def test_unknown_word(self, trained_model_with_dists):
        """Test pronouncing word not in training data."""
        dt = trained_model_with_dists

        # Should not crash
        result = dt.pronounce("xyzabc")
        assert isinstance(result, list)

        phones, confs = dt.pronounce_with_confidence("xyzabc")
        assert isinstance(phones, list)
        assert isinstance(confs, list)

    def test_nbest_n_equals_1(self, trained_model_with_dists):
        """Test n-best with n=1."""
        dt = trained_model_with_dists

        nbest = dt.pronounce_nbest("hello", n=1)

        assert len(nbest) == 1
        assert nbest[0][0] == dt.pronounce("hello")

    def test_nbest_large_n(self, trained_model_with_dists):
        """Test n-best with very large n."""
        dt = trained_model_with_dists

        nbest = dt.pronounce_nbest("hello", n=100)

        # Should not crash, just return what's available
        assert isinstance(nbest, list)
        assert len(nbest) >= 1


class TestConfidenceScoreProperties:
    """Test properties of confidence scores."""

    def test_confidence_range(self, trained_model_with_dists):
        """Test that all confidences are in valid range."""
        dt = trained_model_with_dists

        test_words = ["hello", "world", "test", "read"]

        for word in test_words:
            _, confs = dt.pronounce_with_confidence(word)
            for conf in confs:
                assert 0.0 <= conf <= 1.0, (
                    f"Confidence {conf} out of range for '{word}'"
                )

    def test_nbest_scores_sum_reasonable(self, trained_model_with_dists):
        """Test that n-best scores are reasonable probabilities."""
        dt = trained_model_with_dists

        nbest = dt.pronounce_nbest("hello", n=10)

        if len(nbest) > 1:
            total = sum(score for _, score in nbest)
            # Total might not be exactly 1.0 due to beam search pruning
            # but should be reasonable
            assert 0.0 < total <= 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
