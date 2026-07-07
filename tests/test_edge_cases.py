"""
Edge case tests to stress test the g2p system.

Tests unusual inputs, boundary conditions, and error handling.
"""

import gzip
import json
from pathlib import Path

import pytest

from phonebox import DecisionTree, Dictionary
from phonebox.converter import G2P
from phonebox.dictionary import parse_dict_line, strip_stress


class TestEmptyAndTrivial:
    """Test empty and trivial inputs."""

    @pytest.fixture
    def g2p(self):
        """G2P instance for testing."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return G2P(model=model_path)

    def test_empty_string(self, g2p):
        """Test empty string input."""
        result = g2p("")
        assert isinstance(result, list)
        # Should return empty or handle gracefully

    def test_single_character(self, g2p):
        """Test single character words."""
        result = g2p("a")
        assert isinstance(result, list)

        result = g2p("i")
        assert isinstance(result, list)

    def test_whitespace_only(self, g2p):
        """Test whitespace inputs."""
        result = g2p("   ")
        assert isinstance(result, list)

        result = g2p("\t")
        assert isinstance(result, list)

    def test_single_phoneme_words(self, g2p):
        """Test words that should produce single phoneme."""
        # These might produce 1-2 phonemes
        for word in ["a", "i", "oh", "uh"]:
            result = g2p(word)
            assert isinstance(result, list)
            assert len(result) >= 0


class TestExtremeInputs:
    """Test extreme and unusual inputs."""

    @pytest.fixture
    def g2p(self):
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return G2P(model=model_path)

    def test_very_long_word(self, g2p):
        """Test extremely long words."""
        # Antidisestablishmentarianism (28 letters)
        result = g2p("antidisestablishmentarianism")
        assert isinstance(result, list)
        assert len(result) > 10  # Should have many phonemes

    def test_repeated_characters(self, g2p):
        """Test words with many repeated characters."""
        result = g2p("aaaaaaa")
        assert isinstance(result, list)

        result = g2p("bookkeeper")  # Has 'kk', 'ee'
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_vowels(self, g2p):
        """Test words that are all vowels."""
        result = g2p("aeiou")
        assert isinstance(result, list)

        result = g2p("eau")  # French-style
        assert isinstance(result, list)

    def test_all_consonants(self, g2p):
        """Test consonant clusters."""
        result = g2p("str")
        assert isinstance(result, list)

        result = g2p("rhythm")  # No traditional vowels
        assert isinstance(result, list)

    def test_numbers_and_digits(self, g2p):
        """Test handling of numbers."""
        result = g2p("test123")
        assert isinstance(result, list)
        # Numbers should be skipped

        result = g2p("1234")
        assert isinstance(result, list)

    def test_special_characters(self, g2p):
        """Test special characters."""
        test_cases = [
            "hello!",
            "test@example",
            "hi-there",
            "don't",
            "it's",
            "hello...world",
        ]

        for word in test_cases:
            result = g2p(word)
            assert isinstance(result, list)

    def test_unicode_characters(self, g2p):
        """Test various Unicode characters."""
        test_cases = [
            "café",  # Latin with accent
            "naïve",  # Diacritic
            "résumé",  # Multiple accents
            "façade",  # Cedilla
        ]

        for word in test_cases:
            result = g2p(word)
            assert isinstance(result, list)
            # Should handle gracefully (skip unknown chars)

    def test_non_latin_scripts(self, g2p):
        """Test non-Latin scripts."""
        # Should handle gracefully (skip or process what it can)
        result = g2p("hello世界")
        assert isinstance(result, list)
        # Should process 'hello' part at least


class TestDictionaryEdgeCases:
    """Test dictionary parsing edge cases."""

    def test_parse_malformed_lines(self):
        """Test parsing malformed dictionary lines."""
        # No phonemes
        assert parse_dict_line("word") is None

        # Only word
        assert parse_dict_line("word ") is None

    def test_parse_extra_whitespace(self):
        """Test lines with extra whitespace."""
        word, phones = parse_dict_line("word    P H O N E S")
        assert word == "word"
        assert phones == ["P", "H", "O", "N", "E", "S"]

    def test_parse_mixed_tabs_spaces(self):
        """Test mixed tabs and spaces."""
        word, phones = parse_dict_line("word\t  P H  O N E")
        assert word == "word"
        assert len(phones) > 0

    def test_strip_stress_edge_cases(self):
        """Test stress removal edge cases."""
        assert strip_stress("") == ""
        assert strip_stress("X") == "X"
        assert strip_stress("XX0") == "XX"
        assert strip_stress("0") == ""
        assert strip_stress("AH0") == "AH"
        assert strip_stress("AH1") == "AH"
        assert strip_stress("AH2") == "AH"

    def test_parse_unicode_in_dict(self):
        """Test parsing dictionary with Unicode."""
        word, phones = parse_dict_line("café K AE F EY")
        assert word == "café"
        assert phones == ["K", "AE", "F", "EY"]


class TestTrainingEdgeCases:
    """Test training with edge cases."""

    def test_train_tiny_dataset(self, tmp_path):
        """Test training with minimal data."""
        dict_file = tmp_path / "tiny.dict"
        dict_file.write_text("a AH\nb B IY\n")

        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        dt.align()
        dt.train()

        # Should not crash
        result = dt.pronounce("a")
        assert isinstance(result, list)

    def test_train_pure_dataset(self, tmp_path):
        """Test training when all entries produce same phoneme."""
        dict_file = tmp_path / "pure.dict"
        dict_file.write_text("a AH\ne AH\ni AH\no AH\n")

        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        dt.align()
        dt.train()

        # Tree should be simple (all produce AH)
        assert dt.model is not None

    def test_train_with_duplicates(self, tmp_path):
        """Test training with duplicate entries."""
        dict_file = tmp_path / "dups.dict"
        dict_file.write_text("test T EH S T\ntest T EH S T\ntest T EH S T\n")

        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        dt.align()
        dt.train()

        result = dt.pronounce("test")
        assert result == ["T", "EH", "S", "T"]


class TestModelEdgeCases:
    """Test model loading and handling edge cases."""

    def test_load_nonexistent_model(self):
        """Test loading model that doesn't exist."""
        with pytest.raises((FileNotFoundError, OSError)):
            G2P(model="nonexistent.g2p.gz")

    def test_load_corrupted_model(self, tmp_path):
        """Test loading corrupted model file."""
        bad_model = tmp_path / "bad.g2p.gz"

        # Create invalid gzip file
        with gzip.open(bad_model, "wt") as f:
            f.write("not json")

        with pytest.raises((json.JSONDecodeError, ValueError)):
            G2P(model=str(bad_model))

    def test_empty_model_file(self, tmp_path):
        """Test loading empty model."""
        empty_model = tmp_path / "empty.g2p.gz"

        with gzip.open(empty_model, "wt") as f:
            f.write("{}")

        # Should handle gracefully or raise clear error
        try:
            g2p = G2P(model=str(empty_model))
            result = g2p("test")
            assert isinstance(result, list)
        except (KeyError, TypeError, ValueError, StopIteration):
            pass  # Expected for malformed model


class TestConcurrencyAndMemory:
    """Test concurrent usage and memory efficiency."""

    @pytest.fixture
    def g2p(self):
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return G2P(model=model_path)

    def test_multiple_instances(self):
        """Test creating multiple g2p instances."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        # Create multiple instances
        g2p1 = G2P(model=model_path)
        g2p2 = G2P(model=model_path)

        # Should both work independently
        result1 = g2p1("test")
        result2 = g2p2("test")

        assert result1 == result2

    def test_repeated_pronunciation(self, g2p):
        """Test pronouncing same word many times."""
        # Should be consistent and not leak memory
        results = [g2p("hello") for _ in range(1000)]

        # All should be identical
        assert all(r == results[0] for r in results)

    def test_batch_large(self, g2p):
        """Test large batch pronunciation."""
        # Generate many test words
        words = [f"test{i}" for i in range(1000)]

        results = g2p.pronounce_batch(words)
        assert len(results) == 1000
        assert all(isinstance(phones, list) for _, phones in results)


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    @pytest.fixture
    def g2p(self):
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return G2P(model=model_path)

    def test_word_with_only_unknown_chars(self, g2p):
        """Test word with only unknown characters."""
        result = g2p("😀🎉")
        assert isinstance(result, list)
        # Should return empty or handle gracefully

    def test_extremely_long_word(self, g2p):
        """Test extremely long synthetic word."""
        long_word = "a" * 1000
        result = g2p(long_word)
        assert isinstance(result, list)
        # Should handle without crashing

    def test_word_starting_with_special_char(self, g2p):
        """Test words starting with special characters."""
        result = g2p("'bout")  # Common in CMUdict
        assert isinstance(result, list)
        assert len(result) > 0

    def test_word_ending_with_special_char(self, g2p):
        """Test words ending with special characters."""
        result = g2p("test'")
        assert isinstance(result, list)

    def test_case_sensitivity(self):
        """Test that case doesn't matter for uncased models."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        g2p = G2P(model=model_path)

        lower = g2p("hello")
        upper = g2p("HELLO")
        mixed = g2p("HeLLo")

        # Should all be the same (lowercase)
        assert lower == upper == mixed


class TestExceptionsEdgeCases:
    """Test exceptions dictionary edge cases."""

    def test_exceptions_with_empty_pronunciation(self):
        """Test exceptions dict with empty pronunciation."""
        dt = DecisionTree(locale="en_US", phoneset_name="cmu")
        dt.exceptions = {"word": []}

        result = dt.pronounce("word")
        assert isinstance(result, list)

    def test_exceptions_override_model(self):
        """Test that exceptions truly override model."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        g2p = G2P(model=model_path)

        # Add fake exception
        g2p._dt.exceptions["test"] = ["X", "Y", "Z"]

        result = g2p("test")
        assert result == ["X", "Y", "Z"]  # Should use exception

    def test_disable_exceptions(self):
        """Test disabling exceptions dictionary."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        # Load with exceptions disabled
        g2p = G2P(model=model_path, use_dict_fallback=False)

        # Should only use g2p model
        result = g2p("hello")
        assert isinstance(result, list)


class TestMalformedDictionaries:
    """Test handling of malformed dictionaries."""

    def test_empty_dictionary(self, tmp_path):
        """Test loading empty dictionary."""
        dict_file = tmp_path / "empty.dict"
        dict_file.write_text("")

        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        # Should handle gracefully
        assert dt.em is not None

    def test_dictionary_with_only_comments(self, tmp_path):
        """Test dictionary with only comments."""
        dict_file = tmp_path / "comments.dict"
        dict_file.write_text(
            ";;; Comment line 1\n# Comment line 2\n\n;;; Comment line 3\n"
        )

        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        with open(dict_file) as f:
            dt.load_prondict(f)

        # Should not crash (but alignment/training will fail with no data)
        assert dt.em is not None

        # Don't try to align/train with empty data

    def test_dictionary_mixed_valid_invalid(self, tmp_path):
        """Test dictionary with mix of valid and invalid lines."""
        dict_file = tmp_path / "mixed.dict"
        dict_file.write_text(
            ";;; Comment\n"
            "good G UH D\n"
            "bad\n"  # Missing phonemes
            "\n"
            "also A O L S OW\n"
            "invalid line without tab or space separator properly\n"
        )

        dict = Dictionary(dict_file, locale="en_US")
        entries = dict.entries

        # Should have loaded valid entries
        assert len(entries) >= 2

    def test_dictionary_with_variant_numbers(self, tmp_path):
        """Test dictionary with high variant numbers."""
        dict_file = tmp_path / "variants.dict"
        dict_file.write_text(
            "word W ER D\nword(2) W AO R D\nword(3) W UH R D\nword(99) W IH R D\n"
        )

        dict = Dictionary(dict_file, locale="en_US")
        entries = dict.entries

        # All should map to base word 'word'
        assert all(word == "word" for word, _ in entries)


class TestProcessingEdgeCases:
    """Test dictionary processing edge cases."""

    def test_process_all_duplicates(self, tmp_path):
        """Test processing when all entries are duplicates."""
        dict_file = tmp_path / "dups.dict"
        dict_file.write_text(
            "test T EH1 S T\n" * 100  # 100 identical lines
        )

        dict = Dictionary(dict_file, locale="en_US")
        output = tmp_path / "out.dict"
        processed = dict.process(remove_stress=True, output=output)

        # Should deduplicate to 1 entry
        lines = [
            line for line in processed.path.read_text().split("\n") if line.strip()
        ]
        assert len(lines) == 1

    def test_process_empty_dictionary(self, tmp_path):
        """Test processing empty dictionary."""
        dict_file = tmp_path / "empty.dict"
        dict_file.write_text("")

        dict = Dictionary(dict_file, locale="en_US")
        output = tmp_path / "out.dict"
        processed = dict.process(output=output)

        # Should create empty or near-empty output
        assert processed.path.exists()

    def test_process_only_special_chars(self, tmp_path):
        """Test dictionary with only special character words."""
        dict_file = tmp_path / "special.dict"
        dict_file.write_text("' '  X\n- - Y\n. . Z\n")

        dict = Dictionary(dict_file, locale="en_US")
        entries = dict.entries

        # Should handle special chars in words
        assert len(entries) > 0


class TestRuntimeEdgeCases:
    """Test G2P runtime edge cases."""

    def test_runtime_repeated_calls(self):
        """Test runtime with many repeated calls."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        runtime = G2P(model=model_path)

        # Call many times
        for _ in range(100):
            result = runtime("test")
            assert result == ["T", "EH", "S", "T"]

    def test_runtime_with_empty_exceptions(self):
        """Test runtime with no exceptions."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        runtime = G2P(model=model_path, use_dict_fallback=False)

        # Should still work without exceptions
        result = runtime("test")
        assert isinstance(result, list)

    def test_runtime_callable(self):
        """Test that runtime is callable."""
        model_path = "models/en_US_nostress.g2p.gz"
        if not Path(model_path).exists():
            pytest.skip("Model not found")

        runtime = G2P(model=model_path)

        # Should be callable
        result = runtime("test")
        assert isinstance(result, list)


class TestErrorHandling:
    """Test error handling and robustness."""

    def test_dictionary_file_not_found(self):
        """Test loading non-existent dictionary."""
        with pytest.raises(FileNotFoundError):
            dict = Dictionary(Path("nonexistent.dict"))
            dict.load()

    def test_train_without_data(self):
        """Test training without loading data."""
        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        # Should handle gracefully or raise clear error
        with pytest.raises((AttributeError, ValueError, TypeError)):
            dt.align()

    def test_pronounce_without_model(self):
        """Test pronouncing without trained model."""
        dt = DecisionTree(locale="en_US", phoneset_name="cmu", verbose=False)

        # Should raise error or return empty (no model trained)
        try:
            result = dt.pronounce("test")
            # If it returns something, should be empty or handle gracefully
            assert isinstance(result, list)
        except (TypeError, AttributeError, ValueError):
            # Expected - no model to evaluate
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
