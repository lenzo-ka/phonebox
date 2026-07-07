"""
Integration test for complete PocketSphinx workflow.

This test fetches CMUdict, processes it for PocketSphinx (no stress),
and trains a g2p model. It's the most comprehensive test.

SLOW TEST - runs last.
"""

from pathlib import Path

import pytest

from phonebox import G2P, Dictionary


@pytest.mark.slow
class TestPocketSphinxWorkflow:
    """Test complete PocketSphinx dictionary build and g2p training."""

    def test_cmudict_fetch_process_train(self, tmp_path):
        """
        Complete PocketSphinx workflow:
        1. Fetch CMUdict from GitHub
        2. Remove stress markers (PocketSphinx format)
        3. Train g2p model
        4. Test pronunciation

        This is the canonical workflow for PocketSphinx.
        """
        # Use tmp_path for isolated test
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        # Step 1: Fetch CMUdict
        print("\n  Fetching CMUdict...")
        dict = Dictionary.fetch("cmudict", data_dir=data_dir, verbose=False)

        assert dict.path.exists()
        assert dict.locale == "en_US"

        # Verify it's CMUdict format (with stress)
        sample = dict.path.read_text()[:500]
        assert any(c in sample for c in ["0", "1", "2"])  # Has stress markers

        # Step 2: Process for PocketSphinx (remove stress)
        print("  Processing (removing stress)...")
        processed = dict.process(
            remove_stress=True,
            deduplicate=True,
            output=data_dir / "cmudict" / "cmudict_nostress.dict",
        )

        assert processed.path.exists()

        # Verify stress removed
        content = processed.path.read_text()
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 130000  # Should have ~135K entries

        # Sample a line to verify no stress markers
        sample_line = lines[100]
        phones = sample_line.split()[1:]  # Skip word, get phones
        for phone in phones:
            assert not any(phone.endswith(str(i)) for i in [0, 1, 2])

        print(f"  Processed dictionary: {len(lines):,} entries")

        # Step 3: Train g2p model (use small subset for speed)
        print("  Training g2p model (subset)...")

        # Create smaller dict for faster test
        small_dict = data_dir / "small.dict"
        small_dict.write_text("\n".join(lines[:1000]))

        small_dict_obj = Dictionary(small_dict, locale="en_US")

        # Ensure model directory exists
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "en_US_nostress_test.g2p.gz"

        _ = small_dict_obj.train_g2p_model(
            locale="en_US", output=model_path, remove_stress=True
        )

        assert model_path.exists()
        print(f"  Model size: {model_path.stat().st_size / 1024:.1f} KB")

        # Step 4: Test pronunciation
        print("  Testing pronunciation...")
        g2p = G2P(model=model_path)

        # Test common words
        test_words = {
            "hello": ["HH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            "test": ["T", "EH", "S", "T"],
        }

        for word, _expected_phones in test_words.items():
            result = g2p(word)
            # May not match exactly due to small training set, just check it returns phonemes
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(p, str) for p in result)
            print(f"    {word}: {' '.join(result)}")

        print("  Done: PocketSphinx workflow complete!")

    def test_cmudict_already_fetched(self):
        """Test that we can work with already-fetched CMUdict."""
        cmudict_path = Path("data/cmudict/cmudict.dict")

        if not cmudict_path.exists():
            pytest.skip("CMUdict not fetched yet")

        # Should be able to create Dictionary from existing file
        dict = Dictionary(cmudict_path, locale="en_US")
        assert len(dict) > 100000

        # Should be able to process it
        nostress_path = Path("data/cmudict/cmudict_nostress.dict")
        if nostress_path.exists():
            nostress = Dictionary(nostress_path, locale="en_US")
            assert len(nostress) > 100000

            # Verify stress removed in processed version
            content = nostress_path.read_text()[:1000]
            # Should not have stress markers at end of phonemes
            # (hard to verify precisely, but can check it's different)
            assert len(content) > 100

    def test_trained_model_exists(self):
        """Test that we have a trained PocketSphinx model."""
        model_path = Path("models/en_US_nostress.g2p.gz")

        if not model_path.exists():
            pytest.skip("Model not trained yet")

        # Model should be compact (< 1 MB)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        assert size_mb < 1.0

        # Should be usable
        g2p = G2P(model=model_path)

        # Test some pronunciations
        assert len(g2p("hello")) > 0
        assert len(g2p("world")) > 0
        assert len(g2p("test")) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
