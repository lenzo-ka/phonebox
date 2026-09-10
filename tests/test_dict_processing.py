"""Tests for dictionary processing functionality."""

import pytest

from phonebox import Dictionary
from phonebox.cli.commands.suggest_joins import _load_pairs
from phonebox.cli.main import main
from phonebox.core.vectorizer import Vectorizer
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

    def test_arbitrary_whitespace_and_inline_comment(self):
        word, phones = parse_dict_line("word(2)\tW\tER1   D # sense note")
        assert word == "word"
        assert phones == ["W", "ER1", "D"]

        letters, cooked = Vectorizer(
            locale="en_US", phoneset_name="cmu"
        ).letters_and_phones("word(2)\tW\tER1   D # sense note")
        assert "".join(letters) == "word"
        assert cooked == ["W", "ER1", "D"]

    def test_dictionary_letters_cooked_flag_controls_rewrites(self):
        vectorizer = Vectorizer(
            locale="default",
            phoneset_name="cmu",
            cased=True,
            spelling_rewrites={"a": "b"},
        )
        raw_letters, _ = vectorizer.letters_and_phones("a AH")
        cooked_letters, _ = vectorizer.letters_and_phones("a AH", letters_cooked=True)
        assert raw_letters == ["b"]
        assert cooked_letters == ["a"]

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

    def test_phone_mapping_dedup_renumbers_remaining_variants(self, tmp_path):
        input_file = tmp_path / "input.dict"
        input_file.write_text(
            "read R IY1 D\n"
            "read(2)\tR\tIY2 D # collapses without stress\n"
            "read(3) R EH1 D\n"
        )

        preserved = Dictionary(input_file).process(output=tmp_path / "preserved.dict")
        assert preserved.path.read_text().splitlines() == [
            "read R IY1 D",
            "read(2) R IY2 D",
            "read(3) R EH1 D",
        ]

        mapped = Dictionary(input_file).process(
            phone_transform=lambda phones: [
                {"IY1": "IY", "IY2": "IY", "EH1": "EH"}.get(phone, phone)
                for phone in phones
            ],
            output=tmp_path / "mapped.dict",
        )
        assert mapped.path.read_text().splitlines() == [
            "read R IY D",
            "read(2) R EH D",
        ]

    def test_phone_mapping_precedes_stress_and_deduplication(self, tmp_path):
        source = tmp_path / "input.dict"
        source.write_text("read R IY1 D\nread(2) R X1 D\n")
        processed = Dictionary(source).process(
            phone_mapping={"IY1": "X1"},
            remove_stress=True,
            phoneset="cmu",
            output=tmp_path / "output.dict",
        )
        assert processed.path.read_text() == "read R X D\n"

    def test_unknown_phoneset_preserves_trailing_digits(self, tmp_path):
        source = tmp_path / "input.dict"
        source.write_text("word TONE1 X2 A3\n")
        processed = Dictionary(source).process(
            remove_stress=True,
            phoneset="custom",
            output=tmp_path / "output.dict",
        )
        assert processed.path.read_text() == "word TONE1 X2 A3\n"

    def test_phone_mapping_requires_phone_list(self, tmp_path):
        input_file = tmp_path / "input.dict"
        input_file.write_text("word W ER D\n")
        with pytest.raises(TypeError, match="list of nonempty strings"):
            Dictionary(input_file).process(
                phone_transform=lambda _phones: "W ER D",  # type: ignore[arg-type,return-value]
                output=tmp_path / "bad.dict",
            )
        with pytest.raises(TypeError, match="list of nonempty strings"):
            Dictionary(input_file).process(
                phone_transform=lambda _phones: [],
                output=tmp_path / "empty.dict",
            )

    def test_process_rejects_same_file_before_truncating(self, tmp_path):
        source = tmp_path / "input.dict"
        original = "word W ER D\n"
        source.write_text(original)
        with pytest.raises(ValueError, match="must be different"):
            Dictionary(source).process(output=source)
        assert source.read_text() == original

        alias = tmp_path / "alias.dict"
        alias.hardlink_to(source)
        with pytest.raises(ValueError, match="must be different"):
            Dictionary(source).process(output=alias)
        assert source.read_text() == original

    def test_all_public_lexicon_readers_share_whitespace_grammar(self, tmp_path):
        source = tmp_path / "input.dict"
        source.write_text("word(2) W  ER1\tD # note\n")
        assert _load_pairs(source) == [(["w", "o", "r", "d"], ["W", "ER1", "D"])]
        letters, phones = Vectorizer(
            locale="en_US", phoneset_name="cmu", remove_stress=True
        ).letters_and_phones(source.read_text(), phones_cooked=True)
        assert "".join(letters) == "word"
        assert phones == ["W", "ER1", "D"]

    def test_process_cli_delegates_phone_mapping(self, tmp_path):
        source = tmp_path / "input.dict"
        output = tmp_path / "output.dict"
        mapping = tmp_path / "phones.json"
        source.write_text("read R IY1 D\nread(2) R IY2 D\n")
        mapping.write_text('{"IY1": "IY", "IY2": "IY"}')
        assert (
            main(
                [
                    "dict",
                    "process",
                    str(source),
                    "-o",
                    str(output),
                    "--phone-map",
                    str(mapping),
                ]
            )
            == 0
        )
        assert output.read_text() == "read R IY D\n"

    def test_check_cli_uses_shared_whitespace_comment_grammar(self, tmp_path, capsys):
        lexicon = tmp_path / "input.dict"
        phoneset = tmp_path / "phones.json"
        lexicon.write_text("word(2)   W BAD # note\n")
        phoneset.write_text('["W"]')
        assert (
            main(
                [
                    "check",
                    "--lexicon",
                    str(lexicon),
                    "--phoneset",
                    str(phoneset),
                    "--strict",
                ]
            )
            == 1
        )
        output = capsys.readouterr().out
        assert "lexicon: 1 entries" in output
        assert "BAD" in output

    def test_alignment_letters_are_not_recooked(self):
        vectorizer = Vectorizer(
            locale="default",
            phoneset_name="cmu",
            cased=True,
            spelling_rewrites={"a": "b", "b": "c"},
        )
        vector = next(vectorizer.next_alignment_vector("b\tAH1"))
        assert vector.split()[vectorizer.width // 2] == "b"

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
