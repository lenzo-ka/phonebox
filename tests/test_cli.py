#!/usr/bin/env python
"""
Tests for CLI commands and subcommands.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from phonebox.cli.commands.bundle import handle_bundle
from phonebox.cli.commands.pronounce import handle_pronounce
from phonebox.cli.main import main


class TestCLIHelp:
    """Test help system at all levels."""

    def test_main_help(self):
        """Test g2p --help."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "model" in result.stdout
        assert "dict" in result.stdout
        assert "pronounce" in result.stdout

    def test_help_command(self):
        """Test g2p help."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "model" in result.stdout

    def test_help_subcommand(self):
        """Test g2p help model."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "help", "model"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout
        assert "build" in result.stdout

    def test_subcommand_help(self):
        """Test g2p model help."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model", "help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout

    def test_subcommand_help_nested(self):
        """Test g2p model help build."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model", "help", "build"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "dict" in result.stdout.lower()
        assert "output" in result.stdout.lower()

    def test_model_no_subcommand(self):
        """Test g2p model (no subcommand shows help)."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "benchmark" in result.stdout

    def test_dict_no_subcommand(self):
        """Test g2p dict (no subcommand shows help)."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "dict"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "fetch" in result.stdout

    def test_normalize_help(self):
        """Test phonebox normalize --help."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "normalize", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "locale-specific grapheme transformations" in result.stdout
        assert "pronounce_text" in result.stdout

    def test_main_accepts_argv_without_mutating_sys_argv(self, capsys):
        original = list(sys.argv)
        assert main(["normalize", "¡cafe\u0301!"]) == 0
        assert capsys.readouterr().out == "café\n"
        assert sys.argv == original

    def test_package_module_entry_point(self):
        result = subprocess.run(
            [sys.executable, "-m", "phonebox", "normalize", "¡hola!"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == "hola\n"

    def test_help_alias_handles_nested_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "phonebox", "help", "model", "build"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--output" in result.stdout

    @pytest.mark.parametrize(
        "arguments", [["normalize", "help"], ["normalize", "--", "help"]]
    )
    def test_literal_help_operand_is_not_rewritten(self, arguments, capsys):
        assert main(arguments) == 0
        assert capsys.readouterr().out == "help\n"


class TestPronounceCommand:
    """Test g2p pronounce command."""

    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("models/en_US_nostress.g2p.gz")
        if not path.exists():
            pytest.skip("Model not found")
        return str(path)

    def test_pronounce_single_word(self, model_path):
        """Test g2p pronounce <word>."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "pronounce",
                "hello",
                "--model",
                model_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert "HH" in result.stdout or "AH" in result.stdout

    def test_pronounce_multiple_words(self, model_path):
        """Test g2p pronounce <word1> <word2>."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "pronounce",
                "hello",
                "world",
                "--model",
                model_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_pronounce_with_confidence(self, model_path):
        """Test g2p pronounce --confidence."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "pronounce",
                "hello",
                "--model",
                model_path,
                "--with-confidence",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout
        # Should have confidence score (tab-separated format: word\tphones\tscore)
        parts = result.stdout.strip().split("\t")
        assert len(parts) >= 3  # word, phones, score
        # Score should be a float
        try:
            float(parts[-1])
        except ValueError:
            raise AssertionError(
                f"Expected confidence score, got: {parts[-1]}"
            ) from None

    def test_pronounce_nbest(self, model_path):
        """Test g2p pronounce --nbest."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "pronounce",
                "hello",
                "--model",
                model_path,
                "--nbest",
                "3",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should have numbered alternatives
        lines = result.stdout.strip().split("\n")
        assert len(lines) >= 1

    def test_pronounce_batch_mode(self, model_path, tmp_path):
        """Test g2p pronounce with stdin input."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("hello\nworld\npython\n")

        with open(input_file) as f:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phonebox.cli.main",
                    "pronounce",
                    "--model",
                    model_path,
                ],
                stdin=f,
                capture_output=True,
                text=True,
            )

        assert result.returncode == 0
        assert "hello" in result.stdout
        assert "world" in result.stdout
        assert "python" in result.stdout

    def test_pronounce_missing_model(self):
        """Test g2p pronounce without --model flag."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "pronounce", "hello"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_invalid_confidence_options_return_usage_error(self, capsys):
        args = SimpleNamespace(
            confidence_detailed=True,
            with_confidence=False,
            nbest=None,
            confidence_method="average",
        )
        assert handle_pronounce(args) == 2
        assert "requires --with-confidence" in capsys.readouterr().err

    def test_prediction_error_returns_failure_status(self, monkeypatch, capsys):
        class BrokenG2P:
            def __init__(self, **_kwargs):
                pass

            def pronounce(self, _word):
                raise RuntimeError("broken model")

        monkeypatch.setattr("phonebox.converter.G2P", BrokenG2P)
        args = SimpleNamespace(
            model="tree.g2p.gz",
            words=["hello"],
            nbest=None,
            with_confidence=False,
            confidence_detailed=False,
            confidence_method="average",
            locale=None,
            phoneset=None,
        )
        assert handle_pronounce(args) == 1
        assert "broken model" in capsys.readouterr().err


def test_bundle_rejects_multigram_sidecar_before_loading(tmp_path, capsys):
    model = tmp_path / "model.g2p.gz"
    model.write_bytes(b"not needed")
    model.with_suffix(".gz.units.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "g2p.py"

    assert handle_bundle(SimpleNamespace(model=str(model), output=str(output))) == 1
    assert "MultigramG2P" in capsys.readouterr().err
    assert not output.exists()


def test_bundle_library_rejects_exported_multigram(tmp_path):
    from phonebox.bundler import bundle_g2p
    from phonebox.core.multigram_g2p import MultigramG2P

    model_path = tmp_path / "model.g2p"
    model = MultigramG2P(
        max_letter_span=1,
        max_phone_span=1,
        min_phone_span=1,
        em_max_iterations=2,
    )
    model.train_from_pairs([(["a"], ["A"])])
    model.export(model_path)
    output = tmp_path / "predict.py"
    with pytest.raises(ValueError, match="decision-tree models only.*MultigramG2P"):
        bundle_g2p(str(model_path), str(output))
    assert not output.exists()


class TestBenchmarkCommand:
    """Test phonebox benchmark command."""

    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("models/en_US_nostress.g2p.gz")
        if not path.exists():
            pytest.skip("Model not found")
        return str(path)

    def test_benchmark_basic(self, model_path):
        """Test phonebox model benchmark."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "model",
                "benchmark",
                model_path,
                "--iterations",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Benchmarking" in result.stdout
        assert "ms per load" in result.stdout
        assert "ms per word" in result.stdout

    def test_benchmark_default_iterations(self, model_path):
        """Test benchmark with default iteration count."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "model",
                "benchmark",
                model_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Iterations: 100" in result.stdout


class TestCommandGroupStructure:
    """Test overall command structure."""

    def test_all_command_groups_exist(self):
        """Test that all command groups are registered."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        expected_groups = [
            "model",
            "dict",
            "align",
            "pronounce",
            "normalize",
            "bundle",
            "recipe",
            "vectorize",
        ]
        for group in expected_groups:
            assert group in result.stdout, f"Command '{group}' not found"

    def test_model_subcommands_exist(self):
        """Test that model subcommands are registered."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        expected = ["build", "train", "benchmark"]
        for cmd in expected:
            assert cmd in result.stdout, f"Model subcommand '{cmd}' not found"

    def test_dict_subcommands_exist(self):
        """Test that dict subcommands are registered."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "dict", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        expected = ["fetch", "export-vectors"]
        for cmd in expected:
            assert cmd in result.stdout, f"Dict subcommand '{cmd}' not found"

    def test_utility_commands_exist(self):
        """Test that utility commands are registered at top level."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Top-level commands
        expected = ["normalize", "bundle", "recipe", "pronounce"]
        for cmd in expected:
            assert cmd in result.stdout, f"Utility command '{cmd}' not found"

    def test_model_benchmark_exists(self):
        """Test that model benchmark subcommand exists."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout

    def test_version_flag(self):
        """Test g2p --version."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "phonebox" in result.stdout.lower()


class TestCLIErrors:
    """Test error handling in CLI."""

    def test_no_command(self):
        """Test g2p with no command."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_invalid_command(self):
        """Test g2p with invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "invalid"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_invalid_subcommand(self):
        """Test g2p model invalid."""
        result = subprocess.run(
            [sys.executable, "-m", "phonebox.cli.main", "model", "invalid"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()


class TestModelBuildPruning:
    """Pruning surface area for phonebox model build."""

    def test_prune_flag_is_advertised(self):
        """--prune should be visible in model build --help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "model",
                "build",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--prune" in result.stdout
        assert "--validation-split" in result.stdout
        assert "--test-split" in result.stdout

    def test_model_build_with_prune(self, tmp_path):
        """phonebox model build --prune should produce a working model."""
        dict_file = tmp_path / "prune.dict"
        # Toy dict with enough rows so 20% validation isn't empty.
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
                ]
            )
        )

        model_path = tmp_path / "model.g2p.gz"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "model",
                "build",
                "en_US",
                str(dict_file),
                "-o",
                str(model_path),
                "--prune",
                "--validation-split",
                "0.2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert model_path.exists()


def test_train_multigram_rejects_uncooked_rewrite_source(tmp_path):
    lexicon = tmp_path / "tiny.dict"
    lexicon.write_text("x K\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox.cli.main",
            "train-multigram",
            "--locale",
            "en_US",
            "--lexicon",
            str(lexicon),
            "--output",
            str(tmp_path / "model.g2p"),
            "--spelling-rewrite",
            "X=q",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "use the cooked character 'x'" in result.stderr


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("models/en_US_nostress.g2p.gz")
        if not path.exists():
            pytest.skip("Model not found")
        return path

    def test_pronounce_workflow(self, model_path):
        """Test pronouncing with a model."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phonebox.cli.main",
                "pronounce",
                "hello",
                "--model",
                str(model_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip()
