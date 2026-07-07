"""Tests for the Vectorizer module."""

import pytest

from phonebox.core.vectorizer import Vectorizer, join_seq, make_join_re


class TestVectorizer:
    """Test cases for the Vectorizer class."""

    def test_vectorizer_init(self):
        """Test that Vectorizer can be initialized."""
        v = Vectorizer(locale="en_US", phoneset_name="cmu")
        assert v.locale == "en_US"
        assert v.phoneset_name == "cmu"

    def test_canonical_locale(self):
        """Test locale canonicalization."""
        assert Vectorizer.canonical_locale_for("en_us") == "en_US"
        assert Vectorizer.canonical_locale_for("EN_US") == "en_US"
        assert Vectorizer.canonical_locale_for("de_de") == "de_DE"

    def test_default_fallback(self):
        """Test that non-existent locale falls back to default."""
        # This should use the default configuration without error
        v = Vectorizer(locale="xx_YY", phoneset_name="cmu", norm_xlit=True)
        assert v.locale == "xx_YY"
        # The config should still be loaded from default
        assert v.config is not None

    def test_letter_cooking(self):
        """Test letter processing."""
        v = Vectorizer(locale="en_US", phoneset_name="cmu", cased=False)

        # Test lowercase conversion
        letters = v.cook_letters(["H", "E", "L", "L", "O"], g2p=True)
        assert letters == ["h", "e", "l", "l", "o"]

    def test_filter_non_letters(self):
        """Test that non-letters are filtered when filter_non_letters is enabled."""
        v = Vectorizer(locale="en_US", phoneset_name="cmu", filter_non_letters=True)

        # Numbers should be filtered out
        letters = v.cook_letters(["t", "e", "s", "t", "3"], g2p=True)
        assert letters == ["t", "e", "s", "t"]

        # Parentheses with numbers should be filtered
        letters = v.cook_letters(list("word(3)"), g2p=True)
        assert letters == ["w", "o", "r", "d"]

        # Hyphens and apostrophes should be kept
        letters = v.cook_letters(list("don't"), g2p=True)
        assert letters == ["d", "o", "n", "'", "t"]

    def test_arbitrary_phoneset_name(self):
        """phoneset_name is a free-form tag - non-CMU/xsampa names work."""
        # Test against `default/`, which is guaranteed to have empty
        # `ipa` join list (the shipped en_US now declares ipa joins for
        # English digraphs, which is what we want at runtime but defeats
        # the purpose of testing the "no rule defined" fallback path).
        v = Vectorizer(locale="default", phoneset_name="ipa")
        assert v.phoneset_name == "ipa"
        assert v.phon_join_re is None

    def test_unknown_phoneset_name_falls_back(self):
        """Phoneset tags not present in the locale's join dict don't raise."""
        v = Vectorizer(locale="en_US", phoneset_name="some_made_up_phoneset")
        assert v.phon_join_re is None

    def test_remove_stress_unknown_phoneset_is_noop(self):
        """remove_stress=True with an unknown phoneset leaves phones intact."""
        v = Vectorizer(locale="en_US", phoneset_name="ipa", remove_stress=True)
        # CMU-style "AH1" wouldn't normally appear in IPA, but if it did, the
        # vectorizer must not silently strip digits using CMU's regex.
        assert v.cook_phones(["AH1", "ˈa", "e"]) == ["AH1", "ˈa", "e"]

    def test_export_config_with_arbitrary_phoneset_roundtrips(self):
        """export_config must work for any phoneset_name, not just cmu/xsampa."""
        v = Vectorizer(locale="en_US", phoneset_name="ipa")
        cfg = v.export_config()
        assert cfg["phoneset_name"] == "ipa"
        assert "join" in cfg
        assert "letters" in cfg["join"]

    def test_multigram_config_reads_locale_section(self):
        v = Vectorizer(locale="fr_FR", phoneset_name="ipa")
        assert v.multigram_config().get("max_letter_span") == 3

    def test_multigram_config_empty_when_not_set(self):
        v = Vectorizer(locale="es_MX", phoneset_name="ipa")
        assert v.multigram_config() == {}

    def test_join_seq_noop_single_token_pattern(self):
        """Single-token join patterns must not loop forever."""
        regex = make_join_re(["dʒ"])
        seq = ["a", "l", "dʒ", "e", "r", "i", "a"]
        assert join_seq(regex, seq, "\u208a") == seq


if __name__ == "__main__":
    pytest.main([__file__])
