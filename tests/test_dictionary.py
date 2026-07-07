"""Tests for the Dictionary class."""

from pathlib import Path

import pytest

from phonebox import Dictionary
from phonebox.dictionary import parse_dict_line, strip_stress


class TestDictionaryParsing:
    """Test dictionary parsing functions."""

    def test_strip_stress(self):
        """Test stress marker removal."""
        assert strip_stress("AH0") == "AH"
        assert strip_stress("AH1") == "AH"
        assert strip_stress("AH2") == "AH"
        assert strip_stress("K") == "K"
        assert strip_stress("T") == "T"

    def test_parse_dict_line_space_separated(self):
        """Test parsing space-separated dictionary lines."""
        word, phones = parse_dict_line("hello HH AH0 L OW1")
        assert word == "hello"
        assert phones == ["HH", "AH0", "L", "OW1"]

    def test_parse_dict_line_tab_separated(self):
        """Test parsing tab-separated dictionary lines."""
        word, phones = parse_dict_line("hello\tHH AH0 L OW1")
        assert word == "hello"
        assert phones == ["HH", "AH0", "L", "OW1"]

    def test_parse_dict_line_with_variant(self):
        """Test parsing lines with variant markers."""
        word, phones = parse_dict_line("word(2) W ER D")
        assert word == "word"
        assert phones == ["W", "ER", "D"]

    def test_parse_dict_line_with_comment(self):
        """Test parsing lines with inline comments."""
        word, phones = parse_dict_line("hello HH L OW # greeting")
        assert word == "hello"
        assert phones == ["HH", "L", "OW"]

    def test_parse_dict_line_skips_comments(self):
        """Test that comment lines are skipped."""
        assert parse_dict_line(";;; Comment line") is None
        assert parse_dict_line("# Another comment") is None
        assert parse_dict_line("") is None


class TestDictionary:
    """Test Dictionary class."""

    def test_init(self):
        """Test Dictionary initialization."""
        dict = Dictionary(path="test.dict", locale="en_US")
        assert dict.locale == "en_US"
        assert dict.dict_format == "auto"
        assert dict.path == Path("test.dict")

    def test_load_from_cmudict(self):
        """Test loading CMUdict."""
        cmudict_path = Path("data/cmudict/cmudict.dict")
        if not cmudict_path.exists():
            pytest.skip("CMUdict not available")

        dict = Dictionary(cmudict_path, locale="en_US")
        entries = dict.entries

        assert len(entries) > 100000
        assert all(isinstance(e, tuple) and len(e) == 2 for e in entries[:10])

    def test_size_property(self):
        """Test dictionary size property."""
        cmudict_path = Path("data/cmudict/cmudict.dict")
        if not cmudict_path.exists():
            pytest.skip("CMUdict not available")

        dict = Dictionary(cmudict_path)
        assert dict.size > 100000
        assert len(dict) == dict.size

    def test_repr(self):
        """Test string representation."""
        dict = Dictionary(path="test.dict", locale="en_US")
        repr_str = repr(dict)
        assert "Dictionary" in repr_str
        assert "en_US" in repr_str

    def test_process(self, tmp_path):
        """Test dictionary processing."""
        # Create test dictionary
        input_file = tmp_path / "input.dict"
        input_file.write_text("hello HH AH0 L OW1\nworld W ER1 L D")

        dict = Dictionary(input_file, locale="en_US")
        processed = dict.process(remove_stress=True, output=tmp_path / "output.dict")

        assert processed.path.exists()

        # Check processed content
        content = processed.path.read_text()
        assert "AH0" not in content  # Stress removed
        assert "AH" in content


class TestDictionaryFetching:
    """Test dictionary fetching (requires internet)."""

    @pytest.mark.skip(reason="Requires internet connection")
    def test_fetch_cmudict(self, tmp_path):
        """Test fetching CMUdict."""
        dict = Dictionary.fetch("cmudict", data_dir=tmp_path)
        assert dict.path.exists()
        assert dict.locale == "en_US"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
