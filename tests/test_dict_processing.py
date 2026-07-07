"""Tests for dictionary processing functionality."""

import pytest

from phonebox import Dictionary
from phonebox.dictionary import parse_dict_line, strip_stress


class TestStripStress:
    """Test stress marker removal."""

    def test_strip_vowel_stress(self):
        """Test removing stress from vowels."""
        assert strip_stress("AH0") == "AH"
        assert strip_stress("AH1") == "AH"
        assert strip_stress("AH2") == "AH"
        assert strip_stress("EH0") == "EH"
        assert strip_stress("IY1") == "IY"

    def test_strip_consonant(self):
        """Test consonants remain unchanged."""
        assert strip_stress("K") == "K"
        assert strip_stress("T") == "T"
        assert strip_stress("SH") == "SH"
        assert strip_stress("CH") == "CH"


class TestParseDictLine:
    """Test dictionary line parsing."""

    def test_basic_space_separated(self):
        """Test basic space-separated format."""
        word, phones = parse_dict_line("hello HH AH L OW")
        assert word == "hello"
        assert phones == ["HH", "AH", "L", "OW"]

    def test_tab_separated(self):
        """Test tab-separated format."""
        word, phones = parse_dict_line("hello\tHH AH L OW")
        assert word == "hello"
        assert phones == ["HH", "AH", "L", "OW"]

    def test_variant_markers(self):
        """Test handling of variant markers like (2)."""
        word, phones = parse_dict_line("word(2) W ER D")
        assert word == "word"

        word, phones = parse_dict_line("test(3) T EH S T")
        assert word == "test"

    def test_inline_comments(self):
        """Test handling inline comments."""
        word, phones = parse_dict_line("test T EH S T # example")
        assert word == "test"
        assert phones == ["T", "EH", "S", "T"]

    def test_comment_lines(self):
        """Test that full comment lines are skipped."""
        assert parse_dict_line(";;; CMUdict comment") is None
        assert parse_dict_line("# Regular comment") is None

    def test_empty_lines(self):
        """Test that empty lines are skipped."""
        assert parse_dict_line("") is None
        assert parse_dict_line("   ") is None


class TestDictionaryProcessing:
    """Test dictionary processing operations."""

    def test_process_remove_stress(self, tmp_path):
        """Test stress removal in processing."""
        input_file = tmp_path / "input.dict"
        input_file.write_text("hello HH AH0 L OW1\nworld W ER1 L D\n")

        dict = Dictionary(input_file, locale="en_US")
        output_file = tmp_path / "output.dict"
        processed = dict.process(remove_stress=True, output=output_file)

        assert processed.path.exists()
        content = processed.path.read_text()

        # Check stress markers removed
        assert "0" not in content
        assert "1" not in content
        assert "2" not in content

        # Check phonemes present
        assert "HH" in content
        assert "AH" in content

    def test_process_lowercase(self, tmp_path):
        """Test lowercase conversion."""
        input_file = tmp_path / "input.dict"
        input_file.write_text("HELLO HH L OW\nWORLD W ER L D\n")

        dict = Dictionary(input_file)
        output_file = tmp_path / "output.dict"
        processed = dict.process(lowercase=True, output=output_file)

        content = processed.path.read_text()
        assert "hello" in content
        assert "world" in content
        assert "HELLO" not in content

    def test_process_deduplication(self, tmp_path):
        """Test deduplication of identical pronunciations."""
        input_file = tmp_path / "input.dict"
        input_file.write_text(
            "hello HH AH0 L OW1\nhello HH AH0 L OW1\nworld W ER1 L D\n"
        )

        dict = Dictionary(input_file)
        output_file = tmp_path / "output.dict"
        processed = dict.process(
            remove_stress=True, deduplicate=True, output=output_file
        )

        lines = [
            line for line in processed.path.read_text().strip().split("\n") if line
        ]
        assert len(lines) == 2  # Duplicate removed

    def test_process_default_output_path(self, tmp_path):
        """Test default output path generation."""
        input_file = tmp_path / "mydict.dict"
        input_file.write_text("test T EH S T\n")

        dict = Dictionary(input_file)
        processed = dict.process(remove_stress=True)

        assert processed.path.name == "mydict_nostress.dict"
        assert processed.path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
