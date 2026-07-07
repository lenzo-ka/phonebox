"""Tests for DecisionTree core functionality."""

from pathlib import Path

import pytest

from phonebox import DecisionTree


class TestDecisionTree:
    """Test DecisionTree class."""

    def test_init_basic(self):
        """Test basic initialization."""
        dt = DecisionTree(locale="en_US", phoneset_name="cmu")
        assert dt.vectorizer.locale == "en_US"
        assert dt.vectorizer.phoneset_name == "cmu"

    def test_load_model(self):
        """Test loading a pre-trained model."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        dt = DecisionTree(model=model_path, locale="en_US")
        assert dt.model is not None

    def test_pronounce(self):
        """Test pronunciation."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        dt = DecisionTree(model=model_path, locale="en_US")
        result = dt.pronounce("hello")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(p, str) for p in result)

    def test_train_simple(self, tmp_path):
        """Test training on a small dictionary."""
        # Create mini dictionary
        dict_file = tmp_path / "mini.dict"
        dict_content = """
cat K AE T
dog D AO G
hat HH AE T
bat B AE T
mat M AE T
        """.strip()

        dict_file.write_text(dict_content)

        # Train
        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        dt.align()
        dt.train()

        # Test pronunciation
        result = dt.pronounce("cat")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_export_and_load(self, tmp_path):
        """Test exporting and loading a model."""
        # Create mini dictionary
        dict_file = tmp_path / "mini.dict"
        dict_file.write_text("cat K AE T\ndog D AO G")

        # Train
        dt1 = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)
        with open(dict_file) as f:
            dt1.load_prondict(f)
        dt1.align()
        dt1.train()

        # Export
        model_file = tmp_path / "test.g2p.gz"
        dt1.export(str(model_file))
        assert model_file.exists()

        # Load exported model
        dt2 = DecisionTree(model=str(model_file), locale="en_US")

        # Should give same results
        result1 = dt1.pronounce("cat")
        result2 = dt2.pronounce("cat")
        assert result1 == result2


class TestDecisionTreeOptions:
    """Test DecisionTree configuration options."""

    def test_remove_stress_option(self, tmp_path):
        """Test remove_stress parameter."""
        dict_file = tmp_path / "stress.dict"
        dict_file.write_text("cat K AE1 T\ndog D AO1 G")

        # With stress removal
        dt = DecisionTree(
            locale="en_US", phoneset_name="cmu", remove_stress=True, verbose=False
        )

        with open(dict_file) as f:
            dt.load_prondict(f)

        assert dt.vectorizer.remove_stress is True

    def test_cased_option(self, tmp_path):
        """Test cased parameter."""
        dt_uncased = DecisionTree(locale="en_US", cased=False)
        assert dt_uncased.vectorizer.cased is False

        dt_cased = DecisionTree(locale="en_US", cased=True)
        assert dt_cased.vectorizer.cased is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
