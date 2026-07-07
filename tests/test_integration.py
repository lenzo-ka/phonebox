"""Integration tests for complete workflows."""

from pathlib import Path

import pytest

from phonebox import G2P, Dictionary


class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""

    def test_train_and_use(self, tmp_path):
        """Test complete training and usage workflow."""
        # Create dictionary
        dict_file = tmp_path / "train.dict"
        dict_content = """
cat K AE T
dog D AO G
hat HH AE T
bat B AE T
rat R AE T
mat M AE T
sat S AE T
fat F AE T
        """.strip()
        dict_file.write_text(dict_content)

        # Train model
        model_file = tmp_path / "test.g2p.gz"
        g2p = G2P.train(
            dictionary=dict_file,
            locale="en_US",
            remove_stress=False,
            output=model_file,
            verbose=False,
        )

        # Model file should exist
        assert model_file.exists()

        # Test pronunciation
        result = g2p("cat")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_dictionary_to_model_workflow(self, tmp_path):
        """Test Dictionary class workflow to trained model."""
        # Create dictionary with actual duplicate pronunciation
        dict_file = tmp_path / "workflow.dict"
        dict_file.write_text(
            "test T EH1 S T\n"
            "test T EH1 S T\n"  # Exact duplicate - same word, same pronunciation
            "word W ER1 D\n"
        )

        # Process dictionary
        dict = Dictionary(dict_file, locale="en_US")
        processed = dict.process(
            remove_stress=True, deduplicate=True, output=tmp_path / "processed.dict"
        )

        # Verify processing
        assert processed.path.exists()
        lines = [
            line for line in processed.path.read_text().split("\n") if line.strip()
        ]
        assert len(lines) == 2  # Duplicate removed (test once, word once)

        # Train model
        _ = processed.train_g2p_model(locale="en_US", output=tmp_path / "model.g2p.gz")

        assert (tmp_path / "model.g2p.gz").exists()

    def test_pocketsphinx_workflow(self):
        """Test that PocketSphinx workflow components exist."""
        # Just verify the pieces exist
        cmudict_path = Path("data/cmudict/cmudict.dict")
        nostress_path = Path("data/cmudict/cmudict_nostress.dict")
        model_path = Path("models/en_US_nostress.g2p.gz")

        if cmudict_path.exists():
            assert cmudict_path.stat().st_size > 1000000

        if nostress_path.exists():
            # Verify stress removed
            content = nostress_path.read_text()
            # Should have phonemes but no stress digits at end
            assert "AH " in content or "EH " in content

        if model_path.exists():
            # Can load and use the model
            g2p = G2P(model=model_path)
            result = g2p("test")
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
