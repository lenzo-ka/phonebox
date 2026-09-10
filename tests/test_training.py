"""Public dictionary-training workflow contracts."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

from phonebox import G2P, train_g2p, train_g2p_from_config
from phonebox.config_loader import DEFAULT_CONFIG
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
