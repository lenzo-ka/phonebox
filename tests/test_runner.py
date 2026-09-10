"""Tests for the lightweight G2PRunner and the bundled `.py` template.

Specifically exercises letter-joining (digraph) round-trip: a model trained
with `join.letters` must produce matching pronunciations when reloaded
through the runner or executed as a bundled standalone script.
"""

from __future__ import annotations

import gzip
import json
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

    @pytest.mark.parametrize(
        "snapshot",
        [None, {"version": 1, "source": {"norm_rules": None}}],
    )
    def test_runner_rejects_present_malformed_snapshot(self, tmp_path, snapshot):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        cart_path = tmp_path / "malformed.cart"
        g2p._dt._cart.export(
            str(cart_path), metadata={"letter_preprocessing": snapshot}
        )

        with pytest.raises(ValueError, match="letter_preprocessing"):
            G2PRunner(str(cart_path))

    def test_runner_and_bundle_accept_legacy_absent_snapshot(self, tmp_path):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        cart_path = tmp_path / "legacy.cart"
        g2p._dt._cart.export(str(cart_path), metadata={"cased": False})

        runner = G2PRunner(str(cart_path))
        assert runner.portable_preprocessing is None
        bundle_path = tmp_path / "legacy.py"
        bundle_g2p(str(cart_path), str(bundle_path))
        subprocess.run(
            [sys.executable, "-S", str(bundle_path), "cat"],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_snapshot_fields_override_conflicting_outer_metadata(self, tmp_path):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        metadata = g2p._dt.vectorizer.export_config()
        metadata["cased"] = True
        metadata["join_char"] = "+"
        cart_path = tmp_path / "conflicting.cart"
        g2p._dt._cart.export(str(cart_path), metadata=metadata)

        runner = G2PRunner(str(cart_path))
        assert runner.cased is False
        assert runner.join_char == "₊"
        assert runner.uncook(["k₊s"]) == ["k", "s"]

        bundle_path = tmp_path / "conflicting.py"
        bundle_g2p(str(cart_path), str(bundle_path))
        probe = (
            "import json,runpy; "
            f"ns=runpy.run_path({str(bundle_path)!r}); "
            "g=ns['G2PPredictor'].from_embedded(); "
            "print(json.dumps([g.cased,g.join_char,g.uncook(['k₊s'])]))"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == [False, "₊", ["k", "s"]]

    def test_snapshot_liaison_pad_matches_full_vector_windows(self, tmp_path):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        vectorizer = g2p._dt.vectorizer
        vectorizer.config["join"]["letters"] = []
        vectorizer.lett_join_re = None
        vectorizer.norm_transliterator = None
        vectorizer.g2p_transliterator = None
        vectorizer.liaison_pad = "#"
        cart_path = tmp_path / "liaison.cart"
        g2p.save(str(cart_path))

        expected = vectorizer.vectorize_word("cat")
        runner = G2PRunner(str(cart_path))
        assert runner.vectorize_word("cat") == expected
        assert [row[runner.center_position] for row in expected] == list("cat#")

        bundle_path = tmp_path / "liaison.py"
        bundle_g2p(str(cart_path), str(bundle_path))
        probe = (
            "import json,runpy; "
            f"ns=runpy.run_path({str(bundle_path)!r}); "
            "g=ns['G2PPredictor'].from_embedded(); "
            "print(json.dumps(g.vectorize_word('cat')))"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == expected

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

    def test_bundle_rejects_present_null_preprocessing(self, tmp_path, monkeypatch):
        cart_path = tmp_path / "model.cart"
        cart_path.write_bytes(b"")
        monkeypatch.setattr(
            "phonebox.bundler.read_cart_metadata",
            lambda _path: {"letter_preprocessing": None},
        )

        with pytest.raises(ValueError, match="letter_preprocessing must be an object"):
            bundle_g2p(str(cart_path), str(tmp_path / "bundle.py"))

    def test_non_cart_bundle_preserves_training_metadata(self, tmp_path):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        g2p._dt.max_iterations = 17
        g2p._dt.max_combinations = 1234
        g2p._dt.min_change_ratio = 0.125
        g2p._dt.trainer = "metadata-sentinel"
        model_path = tmp_path / "metadata.g2p.gz"
        g2p.save(str(model_path))
        expected_hash = g2p._dt.dict_hash

        bundle_path = tmp_path / "metadata.py"
        bundle_g2p(str(model_path), str(bundle_path))
        probe = (
            "import json,runpy; "
            f"ns=runpy.run_path({str(bundle_path)!r}); "
            "m=ns['G2PPredictor'].from_embedded().metadata; "
            "print(json.dumps({'training_config':m['training_config'],"
            "'dict_hash':m['dict_hash']}))"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata = json.loads(result.stdout)
        training = metadata["training_config"]
        assert training["max_iterations"] == 17
        assert training["max_combinations"] == 1234
        assert training["min_change_ratio"] == 0.125
        assert training["trainer"] == "metadata-sentinel"
        assert metadata["dict_hash"] == expected_hash

    def test_non_cart_bundle_preserves_flat_legacy_metadata(self, tmp_path):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        nested_path = tmp_path / "nested.g2p.gz"
        g2p.save(str(nested_path))
        with gzip.open(nested_path, "rt", encoding="utf-8") as source:
            lines = source.readlines()
        header = json.loads(lines[0])
        metadata = header.pop("metadata")
        metadata.pop("letter_preprocessing")
        metadata.update(
            {
                "width": 5,
                "cased": True,
                "join_char": "+",
                "join": {"letters": ["c h"]},
                "exceptions": {"chat": ["LEGACY"]},
            }
        )
        header.update(metadata)
        legacy_path = tmp_path / "legacy.g2p.gz"
        with gzip.open(legacy_path, "wt", encoding="utf-8") as target:
            target.write(json.dumps(header) + "\n")
            target.writelines(lines[1:])

        bundle_path = tmp_path / "legacy.py"
        bundle_g2p(str(legacy_path), str(bundle_path))
        probe = (
            "import json,runpy; "
            f"ns=runpy.run_path({str(bundle_path)!r}); "
            "g=ns['G2PPredictor'].from_embedded(); "
            "print(json.dumps([g.width,g.cased,g.join_char,"
            "g.metadata['join']['letters'],g.exceptions]))"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == [
            5,
            True,
            "+",
            ["c h"],
            {"chat": ["LEGACY"]},
        ]

    def test_failed_non_cart_export_removes_temporary_cart(self, tmp_path, monkeypatch):
        g2p, _ = _train_with_letter_join(tmp_path, ["c h"])
        model_path = tmp_path / "model.g2p.gz"
        g2p.save(str(model_path))
        temporary_cart = tmp_path / "temporary.cart"

        class TemporaryFile:
            name = str(temporary_cart)

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                temporary_cart.touch()
                return self

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(
            "phonebox.bundler.tempfile.NamedTemporaryFile", TemporaryFile
        )
        monkeypatch.setattr(
            "cartlet.DecisionTree.export",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("export failed")
            ),
        )

        with pytest.raises(RuntimeError, match="export failed"):
            bundle_g2p(str(model_path), str(tmp_path / "bundle.py"))
        assert not temporary_cart.exists()

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


@pytest.mark.parametrize(
    ("locale", "dictionary_text", "variants", "expected"),
    [
        (
            "es_MX",
            "n N\nñ NY\nu U\nü W\n",
            ("ñ", "N\u0303", "ü", "U\u0308"),
            (("NY",), ("NY",), ("W",), ("W",)),
        ),
        (
            "it_IT",
            "e E\nè EH\no O\nò OH\n",
            ("è", "e\u0300", "ò", "o\u0300", "é"),
            (("EH",), ("EH",), ("OH",), ("OH",), ("E",)),
        ),
    ],
)
def test_locale_normalization_matches_library_runner_and_stdlib_bundle(
    tmp_path, locale, dictionary_text, variants, expected
):
    dictionary = tmp_path / "letters.dict"
    dictionary.write_text(dictionary_text, encoding="utf-8")
    g2p = G2P.train(
        dictionary,
        locale=locale,
        phoneset="ipa",
        use_dict_fallback=False,
        verbose=False,
    )
    cart_path = tmp_path / "letters.cart"
    g2p._dt.export(str(cart_path), include_exceptions=False)
    model_path = tmp_path / "letters.g2p.gz"
    g2p._dt.export(str(model_path), include_exceptions=False)

    runner = G2PRunner(str(cart_path))
    bundle_path = tmp_path / "letters.py"
    bundle_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox.cli.main",
            "bundle",
            str(model_path),
            "-o",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
    )
    assert bundle_result.returncode == 0, bundle_result.stderr

    for word, phones in zip(variants, expected, strict=True):
        assert tuple(g2p.pronounce(word)) == phones
        assert tuple(runner.pronounce(word)) == phones
        result = subprocess.run(
            [sys.executable, "-S", str(bundle_path), word],
            capture_output=True,
            text=True,
            check=True,
        )
        assert tuple(result.stdout.strip().partition("\t")[2].split()) == phones


def test_bundle_refuses_nonportable_icu_without_writing_output(tmp_path):
    dictionary = tmp_path / "letters.dict"
    dictionary.write_text("a A\n", encoding="utf-8")
    g2p = G2P.train(
        dictionary,
        locale="en_US",
        phoneset="ipa",
        norm_xlit=True,
        use_dict_fallback=False,
        verbose=False,
    )
    cart_path = tmp_path / "letters.cart"
    g2p._dt.export(str(cart_path), include_exceptions=False)
    bundle_path = tmp_path / "letters.py"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox.cli.main",
            "bundle",
            str(cart_path),
            "-o",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Latin" in result.stderr
    assert not bundle_path.exists()
