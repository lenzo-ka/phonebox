"""Tests for the lightweight G2PRunner and the bundled `.py` template.

Specifically exercises letter-joining (digraph) round-trip: a model trained
with `join.letters` must produce matching pronunciations when reloaded
through the runner or executed as a bundled standalone script.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from cartlet import read_cart_metadata

from phonebox import G2P
from phonebox.bundler import bundle_g2p
from phonebox.core.vectorizer import make_join_re
from phonebox.runner import G2PRunner


def _train_with_letter_join(tmp_path, joinings: list[str]):
    """Train a tiny G2P model with custom letter joinings and save to .cart."""
    dict_file = tmp_path / "tiny.dict"
    dict_file.write_text(
        "chat CH AE T\n"
        "chin CH IH N\n"
        "chip CH IH P\n"
        "cat K AE T\n"
        "cake K EY K\n"
        "cool K UW L\n"
        "much M AH CH\n"
        "rich R IH CH\n"
        "batch B AE CH\n"
    )

    g2p = G2P.train(
        dictionary=dict_file,
        locale="en_US",
        output=tmp_path / "model.g2p.gz",
        verbose=False,
    )

    v = g2p._dt.vectorizer
    v.config["join"]["letters"] = list(joinings)
    v.lett_join_re = make_join_re(joinings)
    g2p._dt.align()
    g2p._dt.train()

    cart_path = tmp_path / "model.cart"
    g2p.save(str(cart_path))
    return g2p, cart_path


class TestCartMetadata:
    """The `.cart` writer appends a JSON metadata block; verify we can read it back."""

    def test_read_metadata_roundtrip(self, tmp_path):
        _, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        meta = read_cart_metadata(str(cart_path))

        assert meta["width"] == 7
        assert meta["join_char"] == "\u208a"
        assert meta["join"]["letters"] == ["c h"]

    def test_read_metadata_rejects_non_cart_file(self, tmp_path):
        bogus = tmp_path / "not_a_cart.bin"
        bogus.write_bytes(b"NOPE" + b"\x00" * 64)
        with pytest.raises(ValueError):
            read_cart_metadata(str(bogus))


class TestRunnerLetterJoining:
    """The runner must apply letter-joining to match the heavy G2P pipeline."""

    @pytest.mark.parametrize("word", ["cat", "chat", "rich", "cool", "much", "batch"])
    def test_runner_matches_heavy(self, tmp_path, word):
        g2p, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        runner = G2PRunner(str(cart_path))
        assert runner.pronounce(word) == g2p(word)

    def test_runner_oov_matches_heavy(self, tmp_path):
        """Letters outside the training vocab must epsilon-out (no nonsense phones)."""
        g2p, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        runner = G2PRunner(str(cart_path))
        assert runner.pronounce("xyz") == g2p("xyz") == []

    def test_runner_picks_up_letter_join_re(self, tmp_path):
        _, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        runner = G2PRunner(str(cart_path))

        assert runner.lett_join_re is not None
        # vectorize_word should emit 3 cooked tokens for "chat" (c+h, a, t),
        # not 4 raw chars.
        vectors = runner.vectorize_word("chat")
        assert len(vectors) == 3
        center_letters = [v[runner.center_position] for v in vectors]
        assert center_letters[0] == "c" + runner.join_char + "h"

    def test_runner_without_letter_joins_unchanged(self, tmp_path):
        # No letter joinings: vectorize_word splits one-letter-per-vector.
        # We force an empty join list explicitly because en_US's shipped
        # config now declares English digraph joins (s h, c h, t h, ...);
        # the "unchanged" path needs to be exercised by stripping them
        # before saving the model.
        dict_file = tmp_path / "tiny.dict"
        dict_file.write_text("cat K AE T\ndog D AO G\nfox F AA K S\n")
        g2p = G2P.train(
            dictionary=dict_file,
            locale="en_US",
            output=tmp_path / "model.g2p.gz",
            verbose=False,
        )
        v = g2p._dt.vectorizer
        v.config["join"]["letters"] = []
        v.lett_join_re = None

        cart_path = tmp_path / "model.cart"
        g2p.save(str(cart_path))

        runner = G2PRunner(str(cart_path))
        assert runner.lett_join_re is None
        assert runner.pronounce("cat") == g2p("cat")


class TestBundledStandalone:
    """The bundled standalone .py must reproduce the same predictions."""

    @pytest.mark.parametrize("word", ["cat", "chat", "rich", "cool", "much"])
    def test_bundle_matches_heavy(self, tmp_path, word):
        g2p, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        bundle_path = tmp_path / "g2p.py"
        bundle_g2p(str(cart_path), str(bundle_path))

        result = subprocess.run(
            [sys.executable, str(bundle_path), word],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output format: "word\tP1 P2 P3"
        out_word, _, phones_str = result.stdout.strip().partition("\t")
        assert out_word == word
        assert phones_str.split() == g2p(word)

    def test_bundle_oov_letters_match_heavy(self, tmp_path):
        """Bundled runner must epsilon-out OOV center letters like the heavy path.

        Without OOV checking the tree falls through default branches and emits
        a phoneme for an unseen letter. Verify the bundle agrees with G2P.
        """
        g2p, cart_path = _train_with_letter_join(tmp_path, ["c h"])
        bundle_path = tmp_path / "g2p.py"
        bundle_g2p(str(cart_path), str(bundle_path))

        # 'x', 'y', 'z' are not in the toy training vocab; expect empty output.
        result = subprocess.run(
            [sys.executable, str(bundle_path), "xyz"],
            capture_output=True,
            text=True,
            check=True,
        )
        _, _, phones_str = result.stdout.strip().partition("\t")
        assert phones_str.split() == g2p("xyz")
        assert g2p("xyz") == []  # sanity check on the heavy path
