"""Tests for the G2P high-level API."""

from pathlib import Path

import pytest

from phonebox import G2P


class TestG2P:
    """Test the G2P class."""

    @pytest.fixture
    def model_path(self):
        """Path to test model."""
        return "models/en_US_nostress.g2p.gz"

    @pytest.fixture
    def g2p(self, model_path):
        """Create G2P instance for testing."""
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")
        return G2P(model=model_path)

    def test_init_with_model(self, model_path):
        """Test G2P initialization with model."""
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        g2p = G2P(model=model_path)
        assert g2p.locale is not None
        assert g2p.phoneset is not None

    def test_pronounce_single_word(self, g2p):
        """Test pronunciation of single words."""
        result = g2p.pronounce("hello")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(p, str) for p in result)

    def test_callable_interface(self, g2p):
        """Test that G2P instance is callable."""
        result = g2p("world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_pronounce_batch(self, g2p):
        """Test batch pronunciation."""
        words = ["hello", "world", "test"]
        results = g2p.pronounce_batch(words)

        assert len(results) == len(words)
        for word, phones in results:
            assert isinstance(word, str)
            assert isinstance(phones, list)
            assert len(phones) > 0

    def test_pronounce_text(self, g2p):
        """Test pronouncing text with multiple words."""
        text = "hello world"
        results = g2p.pronounce_text(text)

        assert len(results) == 2
        assert results[0][0] == "hello"
        assert results[1][0] == "world"

    def test_consistency(self, g2p):
        """Test that same word gives same pronunciation."""
        word = "test"
        result1 = g2p(word)
        result2 = g2p(word)
        assert result1 == result2

    def test_repr(self, g2p):
        """Test string representation."""
        repr_str = repr(g2p)
        assert "G2P" in repr_str
        assert g2p.locale in repr_str


class TestG2PTrain:
    """Test training functionality."""

    def test_train_from_dict(self, tmp_path):
        """Test training a model from a small dictionary."""
        # Create mini dictionary
        dict_file = tmp_path / "mini.dict"
        dict_content = """
cat K AE T
dog D AO G
hat HH AE T
        """.strip()

        dict_file.write_text(dict_content)

        # Train model
        g2p = G2P.train(
            dictionary=dict_file, locale="en_US", phoneset="cmu", verbose=False
        )

        assert g2p is not None
        assert g2p.locale == "en_US"

        # Test pronunciation
        result = g2p("cat")
        assert isinstance(result, list)

    def test_train_with_pruning(self, tmp_path):
        """Pruning + held-out splits should run and produce a usable model."""
        # Larger toy dict so the validation/test splits aren't empty after rounding.
        dict_file = tmp_path / "prune.dict"
        dict_file.write_text(
            "\n".join(
                [
                    "cat K AE T",
                    "bat B AE T",
                    "hat HH AE T",
                    "mat M AE T",
                    "rat R AE T",
                    "sat S AE T",
                    "fat F AE T",
                    "pat P AE T",
                    "dog D AO G",
                    "fog F AO G",
                    "log L AO G",
                    "bog B AO G",
                    "cog K AO G",
                    "hog HH AO G",
                    "jog JH AO G",
                    "frog F R AO G",
                    "pig P IH G",
                    "big B IH G",
                    "dig D IH G",
                    "wig W IH G",
                ]
            )
        )

        g2p = G2P.train(
            dictionary=dict_file,
            locale="en_US",
            phoneset="cmu",
            prune=True,
            validation_split=0.2,
            test_split=0.1,
            verbose=False,
        )

        result = g2p("cat")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_save_load_roundtrip_preserves_vectorizer_state(self, tmp_path):
        """Saving a model and reloading it must yield identical pronunciations.

        Guards against load_model() silently dropping vectorizer flags
        (remove_accents, letter-joining regex, aether/epsilon/join_char, etc.)
        that were configured at training time. Before this fix, retraining
        Spanish with digraph joining + remove_accents produced great results
        in-memory but identical-to-baseline output after the model was
        round-tripped to disk.
        """
        from phonebox.core.vectorizer import make_join_re

        dict_file = tmp_path / "tiny.dict"
        dict_file.write_text(
            "chat CH AE T\ncat K AE T\nmuch M AH CH\nrich R IH CH\nbatch B AE CH\n"
        )

        g2p = G2P.train(
            dictionary=dict_file,
            locale="en_US",
            phoneset="ipa",
            verbose=False,
        )

        # Inject post-train state changes: letter join + accent stripping.
        v = g2p._dt.vectorizer
        v.remove_accents = True
        v.config["join"]["letters"] = ["c h"]
        v.lett_join_re = make_join_re(["c h"])
        g2p._dt.align()
        g2p._dt.train()

        words = ["chat", "cat", "rich", "much", "café"]
        in_memory = [g2p(w) for w in words]

        model_path = tmp_path / "rt.g2p.gz"
        g2p.save(model_path)
        loaded = G2P(model=str(model_path))
        after_load = [loaded(w) for w in words]

        assert in_memory == after_load, (
            f"Reload diverged: in-memory={in_memory!r} loaded={after_load!r}"
        )
        # Sanity-check that the round-tripped vectorizer carries the same flags.
        lv = loaded._dt.vectorizer
        assert lv.remove_accents is True
        assert lv.phoneset_name == "ipa"
        assert lv.lett_join_re is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
