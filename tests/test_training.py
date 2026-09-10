"""Public dictionary-training workflow contracts."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from phonebox import G2P, train_g2p, train_g2p_from_config
from phonebox.cli.main import main
from phonebox.config_loader import DEFAULT_CONFIG, load_config
from phonebox.constants import (
    DEFAULT_MAX_COMBINATIONS,
    DEFAULT_STORE_DISTRIBUTIONS,
    DEFAULT_TRAIN_PARALLEL_ALIGN,
    DEFAULT_TRAIN_PHONESET,
    DEFAULT_TRAIN_PRUNE,
    DEFAULT_TRAIN_REMOVE_STRESS,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAINER,
)
from phonebox.dictionary import Dictionary

WORDS = [
    "cat K AE1 T",
    "bat B AE1 T",
    "hat HH AE1 T",
    "mat M AE1 T",
    "rat R AE1 T",
    "sat S AE1 T",
    "fat F AE1 T",
    "pat P AE1 T",
    "dog D AO1 G",
    "fog F AO1 G",
    "log L AO1 G",
    "bog B AO1 G",
    "cog K AO1 G",
    "hog HH AO1 G",
    "jog JH AO1 G",
    "frog F R AO1 G",
    "pig P IH1 G",
    "big B IH1 G",
    "dig D IH1 G",
    "wig W IH1 G",
]


@pytest.fixture
def lexicon(tmp_path):
    path = tmp_path / "tiny.dict"
    path.write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    return path


def test_public_workflow_exports_model_and_alignment_checkpoint(lexicon, tmp_path):
    output = tmp_path / "api.g2p.gz"
    result = train_g2p(lexicon, locale="en-US", output=output)

    assert result.output_path == output
    assert result.alignments_path == tmp_path / "api_alignments.txt"
    assert output.is_file()
    assert result.alignments_path.is_file()
    assert result.model.vectorizer.phoneset_name == "ipa"
    assert result.model.vectorizer.remove_stress is False
    assert result.model.store_distributions is True
    assert G2P(model=output, use_dict_fallback=False).phoneset == "ipa"


@pytest.mark.parametrize("remove_stress", [False, True])
def test_g2p_train_stress_mode_survives_roundtrip(lexicon, tmp_path, remove_stress):
    output = tmp_path / f"stress-{remove_stress}.g2p.gz"
    trained = G2P.train(
        lexicon,
        locale="en",
        phoneset="cmu",
        remove_stress=remove_stress,
        output=output,
    )
    loaded = G2P(model=output)

    assert trained._dt.vectorizer.remove_stress is remove_stress
    assert loaded._dt.vectorizer.remove_stress is remove_stress
    expected = ["K", "AE", "T"] if remove_stress else ["K", "AE1", "T"]
    assert trained("cat") == expected
    assert loaded("cat") == expected


def test_cli_matches_public_training_defaults(lexicon, tmp_path):
    output = tmp_path / "cli.g2p.gz"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox",
            "train",
            "--locale",
            "en",
            "--lexicon",
            str(lexicon),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = G2P(model=output)
    assert loaded.phoneset == "ipa"
    assert loaded.remove_stress is False
    assert loaded._dt.store_distributions is True
    assert (tmp_path / "cli_alignments.txt").is_file()


def test_config_and_cli_overrides_use_shared_workflow(lexicon, tmp_path):
    output = tmp_path / "configured.g2p.gz"
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "locale": "en",
                "dictionary": str(lexicon),
                "output": str(output),
                "phoneset": "cmu",
                "prune": False,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "phonebox",
            "train",
            "--config",
            str(config_path),
            "--remove-stress",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = G2P(model=output)
    assert loaded.phoneset == "cmu"
    assert loaded.remove_stress is True
    assert loaded("cat") == ["K", "AE", "T"]


def test_config_defaults_match_primary_training_defaults(lexicon):
    expected = {
        "phoneset": DEFAULT_TRAIN_PHONESET,
        "remove_stress": DEFAULT_TRAIN_REMOVE_STRESS,
        "trainer": DEFAULT_TRAINER,
        "parallel_align": DEFAULT_TRAIN_PARALLEL_ALIGN,
        "prune": DEFAULT_TRAIN_PRUNE,
        "validation_split": DEFAULT_TRAIN_VALIDATION_SPLIT,
        "test_split": DEFAULT_TRAIN_TEST_SPLIT,
        "store_distributions": DEFAULT_STORE_DISTRIBUTIONS,
        "max_combinations": DEFAULT_MAX_COMBINATIONS,
    }
    assert {key: DEFAULT_CONFIG[key] for key in expected} == expected
    for function in (train_g2p, G2P.train):
        signature = inspect.signature(function)
        assert {key: signature.parameters[key].default for key in expected} == expected
    dictionary_signature = inspect.signature(Dictionary.train_g2p_model)
    for key in (
        "phoneset",
        "remove_stress",
        "prune",
        "validation_split",
        "test_split",
    ):
        assert dictionary_signature.parameters[key].default == expected[key]

    result = train_g2p_from_config(
        {"dictionary": str(lexicon), "locale": "en", "prune": False}
    )
    assert result.model.vectorizer.phoneset_name == "ipa"


@pytest.mark.parametrize("collision", ["input-output", "input-alignments", "both"])
def test_training_rejects_artifact_path_collisions_before_writing(
    lexicon, tmp_path, collision
):
    original = lexicon.read_bytes()
    output = lexicon if collision == "input-output" else tmp_path / "model.g2p.gz"
    alignments = (
        lexicon
        if collision == "input-alignments"
        else (output if collision == "both" else None)
    )
    with pytest.raises(ValueError, match="different files"):
        train_g2p(
            lexicon,
            locale="en",
            output=output,
            alignments_out=alignments,
        )
    assert lexicon.read_bytes() == original


def test_training_rejects_derived_checkpoint_and_hardlink_aliases(lexicon, tmp_path):
    derived_input = tmp_path / "model_alignments.txt"
    derived_input.write_bytes(lexicon.read_bytes())
    with pytest.raises(ValueError, match="dictionary and alignments"):
        train_g2p(derived_input, locale="en", output=tmp_path / "model.g2p.gz")

    hardlink = tmp_path / "dictionary-link"
    os.link(lexicon, hardlink)
    with pytest.raises(ValueError, match="dictionary and output"):
        train_g2p(lexicon, locale="en", output=hardlink)


def test_dictionary_config_handles_alignment_checkpoint_once(lexicon, tmp_path):
    checkpoint = tmp_path / "configured-alignments.txt"
    config = tmp_path / "training.json"
    config.write_text(
        json.dumps(
            {
                "locale": "en",
                "alignments_out": str(checkpoint),
                "prune": False,
            }
        ),
        encoding="utf-8",
    )
    model = Dictionary(lexicon).train_g2p_model(config=str(config))
    assert checkpoint.is_file()
    assert model.vectorizer.locale == "en"


@pytest.mark.parametrize("contents", ["null", "[]", '"text"'])
def test_config_file_requires_mapping(tmp_path, contents):
    config = tmp_path / "training.json"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping/object"):
        load_config(str(config))


def test_pocketsphinx_facade_uses_trained_cmu_identity(lexicon, tmp_path, monkeypatch):
    monkeypatch.setattr(
        Dictionary,
        "fetch",
        classmethod(lambda cls, *args, **kwargs: Dictionary(lexicon)),
    )
    g2p = G2P.from_pocketsphinx(tmp_path)
    assert g2p.phoneset == "cmu"
    assert g2p._dt.vectorizer.phoneset_name == "cmu"
    assert g2p.remove_stress is True


def test_recipe_delegates_transformed_dictionary_to_training(tmp_path, monkeypatch):
    import phonebox.training
    from phonebox.cli.commands.recipe import _build_g2p

    source = tmp_path / "source.dict"
    source.write_text("cat K AE2 T # note\ncat(2) K AE0 T\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def capture(dictionary, **options):
        seen["dictionary"] = Path(dictionary).read_text(encoding="utf-8")
        seen["options"] = options
        return SimpleNamespace(model=object())

    monkeypatch.setattr(phonebox.training, "train_g2p", capture)
    args = SimpleNamespace(
        source=str(source),
        preset="tts",
        output=str(tmp_path / "model.g2p.gz"),
        keep_secondary=False,
        mark_unstressed=False,
        data_dir=str(tmp_path),
        verbose=False,
        prune=False,
        validation_split=0.05,
    )
    assert _build_g2p(args) == 0
    assert seen["dictionary"] == "cat K AE T\n"
    assert seen["options"]["phoneset"] == "cmu"
    assert seen["options"]["parallel_align"] is False


@pytest.mark.parametrize(
    ("source_name", "output_name"),
    [("source.dict", "source.dict"), ("runner.alignments.txt", "runner.py")],
)
def test_recipe_rejects_original_input_artifact_aliases(
    tmp_path, source_name, output_name
):
    from phonebox.cli.commands.recipe import _build_g2p

    source = tmp_path / source_name
    source.write_text("cat K AE1 T\n", encoding="utf-8")
    original = source.read_bytes()
    args = SimpleNamespace(
        source=str(source),
        preset="tts",
        output=str(tmp_path / output_name),
        keep_secondary=False,
        mark_unstressed=False,
        data_dir=str(tmp_path),
        verbose=False,
        prune=True,
        validation_split=0.05,
    )
    assert _build_g2p(args) == 2
    assert source.read_bytes() == original


def test_recipe_uses_shared_pruning_default_with_explicit_opt_out():
    from phonebox.cli.commands.recipe import setup_recipe_commands

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_recipe_commands(subparsers)
    enabled = parser.parse_args(["recipe", "input.dict", "tts", "-o", "model"])
    disabled = parser.parse_args(
        ["recipe", "input.dict", "tts", "-o", "model", "--no-prune"]
    )
    assert enabled.prune is DEFAULT_TRAIN_PRUNE is True
    assert disabled.prune is False


@pytest.mark.parametrize("flag", ["--alignments", "--vectors"])
def test_prepared_training_preserves_aliased_input(tmp_path, flag, capsys):
    source = tmp_path / "prepared.txt"
    source.write_text("prepared input must remain intact\n", encoding="utf-8")
    original = source.read_bytes()
    assert main(["model", "train", "en_US", flag, str(source), "-o", str(source)]) == 2
    assert source.read_bytes() == original
    assert "must be different files" in capsys.readouterr().err


def test_malformed_yaml_is_a_clean_cli_error(tmp_path, capsys):
    pytest.importorskip("yaml")
    config = tmp_path / "broken.yaml"
    config.write_text("dictionary: [unterminated\n", encoding="utf-8")
    assert main(["train", "--config", str(config)]) == 2
    error = capsys.readouterr().err
    assert "Invalid YAML config" in error
    assert "Traceback" not in error
