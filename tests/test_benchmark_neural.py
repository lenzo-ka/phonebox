"""Optional neural boundary and real external-toolchain compatibility witnesses."""

import json
import os
import sys
from pathlib import Path

import pytest

from phonebox.eval.benchmark_data import _assemble
from phonebox.eval.benchmark_neural import NeuralSettings, run_neural_training


def tiny_dataset():
    mapping = {"a": "AA1", "b": "t͡ʃ", "c": "ɛ̃", "d": "ɡ"}
    train_words = [a + b for a in "abc" for b in "abcd"] + ["abc"]
    dev_words = ["da", "db", "dc", "dd", "aba", "aca", "Ša"]
    return _assemble(
        "tiny",
        {
            "train": [(word, [mapping[c] for c in word]) for word in train_words],
            "dev": [
                (word, [mapping.get(c, "AA1") for c in word]) for word in dev_words
            ],
            "test": [("cab", ["ɛ̃", "AA1", "t͡ʃ"])],
        },
        {"phoneset": "ipa"},
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"threads": 3},
        {"layers": 0},
        {"warmup_steps": -1},
        {"dropout": float("nan")},
        {"learning_rate": float("inf")},
        {"d_model": 15},
        {"seed": True},
    ],
)
def test_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        NeuralSettings(**settings)


def test_no_torch_import_in_public_adapter():
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import phonebox.eval.benchmark_neural; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_interpreter_receipt_is_preflight(tmp_path):
    interpreter = tmp_path / "unreceipted-python"
    interpreter.symlink_to(sys.executable)
    with pytest.raises(ValueError):
        run_neural_training(
            tiny_dataset(), tmp_path / "run", python_executable=interpreter
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(
    not os.environ.get("PHONEBOX_NEURAL_TEST_PYTHON"),
    reason="optional pinned neural toolchain",
)
def test_actual_training_partial_batch_and_full_dev(tmp_path):
    report = run_neural_training(
        tiny_dataset(),
        tmp_path / "run",
        python_executable=os.environ["PHONEBOX_NEURAL_TEST_PYTHON"],
        settings=NeuralSettings(
            d_model=16,
            d_fft=32,
            layers=1,
            heads=2,
            dropout=0,
            batch_size=2,
            warmup_steps=2,
            max_epochs=2,
        ),
    )
    assert report["training"]["retained_entries"] == 13
    assert report["training"]["entries_visited"] == 26
    assert report["training"]["optimizer_updates"] == 14
    assert report["training"]["dropped_entries"] == 0
    assert report["training"]["nonfinite_batches"] == 0
    assert report["training"]["encoded_dev_loss_entries"] == 6
    assert all(row["metrics"]["n_test"] == 7 for row in report["history"])
    assert all(row["unsupported_inputs"] == 1 for row in report["history"])
    assert report["metrics"]["n_test"] == 1
    assert report["model_bytes"] > 0
    assert str(tmp_path) not in json.dumps(report)
    import torch

    checkpoint = torch.load(
        tmp_path / "run/neural-model.pt", map_location="cpu", weights_only=False
    )
    assert "phoneme_dict" not in checkpoint and "optimizer" not in checkpoint
    assert checkpoint["config"]["preprocessing"]["phoneme_symbols"] == sorted(
        {"AA1", "t͡ʃ", "ɛ̃", "ɡ"}
    )


@pytest.mark.skipif(
    not os.environ.get("PHONEBOX_NEURAL_TEST_PYTHON"),
    reason="optional pinned neural toolchain",
)
def test_actual_profile_never_writes_test_references(tmp_path):
    report = run_neural_training(
        tiny_dataset(),
        tmp_path / "profile",
        python_executable=os.environ["PHONEBOX_NEURAL_TEST_PYTHON"],
        settings=NeuralSettings(
            d_model=16,
            d_fft=32,
            layers=1,
            heads=2,
            dropout=0,
            batch_size=32,
            warmup_steps=2,
            max_epochs=2,
        ),
        profile_only=True,
    )
    assert report["profile_only"] is True
    assert "metrics" not in report
    assert report["training"]["entries_visited"] == 13
    assert report["training"]["optimizer_updates"] == 1
    assert not (tmp_path / "profile/neural-test.json").exists()
    assert report["history"][0]["metrics"]["n_test"] == 7


def test_installed_patch_matches_reviewable_patch(tmp_path):
    from phonebox.eval.benchmark_neural import PATCH_SHA256, write_neural_patch

    output = tmp_path / "external.patch"
    assert write_neural_patch(output) == PATCH_SHA256
    assert (
        output.read_bytes()
        == (
            Path(__file__).parents[1] / "docs/patches/deepphonemizer-0.0.19.patch"
        ).read_bytes()
    )


def test_cli_patch_and_missing_neural_python(tmp_path, capsys):
    from phonebox.cli.main import main
    from phonebox.eval.benchmark_neural import PATCH_SHA256

    output = tmp_path / "neural.patch"
    assert main(["compare", "benchmark-neural-patch", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["patch_sha256"] == PATCH_SHA256
    destination = tmp_path / "result.json"
    destination.write_text("preserve")
    assert (
        main(
            [
                "compare",
                "benchmark",
                "--dataset",
                "italian",
                "--system",
                "deepphonemizer",
                "--neural-python",
                str(tmp_path / "missing"),
                "--work-dir",
                str(tmp_path / "work"),
                "--output",
                str(destination),
            ]
        )
        == 2
    )
    assert destination.read_text() == "preserve"
    assert "Traceback" not in capsys.readouterr().err


def test_installed_source_mismatch_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from phonebox.eval.benchmark_neural_worker import _installed_source

    module = tmp_path / "dp/__init__.py"
    module.parent.mkdir()
    module.write_text("# different author source")
    monkeypatch.setitem(sys.modules, "dp", SimpleNamespace(__file__=str(module)))
    with pytest.raises(ValueError, match="source does not match"):
        _installed_source()


def test_dev_selection_earliest_tie_and_warmup():
    from phonebox.eval.benchmark_neural import _DevSelection

    selection = _DevSelection(warmup_steps=100, patience=2)
    assert selection.observe(10, 30, 1) == (True, False)
    assert selection.observe(10, 30, 2) == (False, False)
    assert selection.nonimproving == 0
    assert selection.observe(10, 20, 100) == (True, False)
    assert selection.observe(11, 0, 101) == (False, False)
    assert selection.observe(10, 20, 102) == (False, True)
    assert selection.best == (10, 20)
    assert selection.observe(9, 50, 103) == (True, False)


@pytest.mark.skipif(
    not os.environ.get("PHONEBOX_NEURAL_TEST_MPS"),
    reason="optional host MPS compatibility",
)
def test_actual_mps_training_and_dev_inference(tmp_path):
    report = run_neural_training(
        tiny_dataset(),
        tmp_path / "mps-profile",
        python_executable=os.environ["PHONEBOX_NEURAL_TEST_PYTHON"],
        device="mps",
        settings=NeuralSettings(
            d_model=16,
            d_fft=32,
            layers=1,
            heads=2,
            dropout=0,
            batch_size=32,
            warmup_steps=2,
            max_epochs=2,
            threads=1,
        ),
        profile_only=True,
    )
    assert report["history"][0]["actual_training_device"] == "mps:0"
    assert report["history"][0]["metrics"]["n_test"] == 7
    assert report["training"]["entries_visited"] == 13
    assert report["settings"]["deterministic_algorithms_requested"] is False
    assert report["history"][0]["mps_driver_allocated_bytes"] > 0


@pytest.mark.parametrize(
    "change",
    [
        {"source_revision": "wrong"},
        {"build": None},
        {"build": {"patch_sha256": "wrong"}},
        {"build": {"executable_sha256": "wrong"}},
    ],
)
def test_invalid_receipt_prevents_job_creation(tmp_path, change):
    from phonebox.eval.benchmark_neural import PATCH_SHA256, SOURCE_REVISION
    from phonebox.eval.cmudict_compare import sha256_file

    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    receipt = {
        "version": "0.0.19",
        "source_revision": SOURCE_REVISION,
        "build": {
            "patch_sha256": PATCH_SHA256,
            "executable_sha256": sha256_file(python),
        },
        **change,
    }
    Path(str(python) + ".provenance.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        run_neural_training(tiny_dataset(), tmp_path / "run", python_executable=python)
    assert not (tmp_path / "run").exists()


def test_cli_interpreter_alias_preserves_binary(tmp_path, capsys):
    from phonebox.cli.main import main

    python = tmp_path / "python"
    python.write_text("preserve executable")
    result = main(
        [
            "compare",
            "benchmark",
            "--dataset",
            "italian",
            "--system",
            "deepphonemizer",
            "--neural-python",
            str(python),
            "--work-dir",
            str(tmp_path / "run"),
            "--output",
            str(python),
        ]
    )
    assert result == 2
    assert python.read_text() == "preserve executable"
    assert not (tmp_path / "run").exists()
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.skipif(
    not os.environ.get("PHONEBOX_NEURAL_TEST_PYTHON"),
    reason="optional pinned neural toolchain",
)
def test_actual_shared_benchmark_neural_result(tmp_path):
    from phonebox.eval.benchmark import run_benchmark

    report = run_benchmark(
        tiny_dataset(),
        "deepphonemizer",
        tmp_path / "shared",
        neural_python=os.environ["PHONEBOX_NEURAL_TEST_PYTHON"],
        neural_settings=NeuralSettings(
            d_model=16,
            d_fft=32,
            layers=1,
            heads=2,
            dropout=0,
            batch_size=32,
            warmup_steps=2,
            max_epochs=1,
        ),
    )
    assert report["system"] == "deepphonemizer"
    assert report["metrics"]["n_test"] == 1
    assert report["settings"]["threads"] == 2
    assert report["settings"]["dictionary_lookup"] is False
    assert report["training"]["dictionary_entries"] == 0
    assert report["training"]["supplied_entries"] == 13
    assert report["training"]["optimizer_updates"] == 1
    assert report["timings"]["training_seconds"] > 0
    assert report["timings"]["prediction_seconds"] > 0
    assert report["provenance"]["neural_toolchain"]["interpreter_receipt"][
        "receipt_binary_binding_verified"
    ]


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        3,
        {"unknown": 1},
        {"layers": "four"},
        {"dropout": None},
        {"batch_size": True},
        {"learning_rate": float("nan")},
    ],
)
def test_settings_json_contract(value):
    with pytest.raises(ValueError):
        NeuralSettings.from_dict(value)
    assert NeuralSettings.from_dict({"threads": 1}).threads == 1


def test_cli_settings_alias_and_wrong_shapes_preserve_outputs(tmp_path, capsys):
    from phonebox.cli.main import main

    settings = tmp_path / "settings.json"
    settings.write_text('{"threads":1}')
    common = [
        "compare",
        "benchmark",
        "--dataset",
        "italian",
        "--system",
        "deepphonemizer",
        "--neural-python",
        sys.executable,
        "--neural-settings",
        str(settings),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    assert main(common + ["--output", str(settings)]) == 2
    assert settings.read_text() == '{"threads":1}'
    output = tmp_path / "result.json"
    output.write_text("preserve")
    for value in ([1], {"layers": "four"}, {"unexpected": 1}):
        settings.write_text(json.dumps(value))
        assert main(common + ["--output", str(output)]) == 2
        assert output.read_text() == "preserve"
        assert not (tmp_path / "work").exists()
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.skipif(
    not os.environ.get("PHONEBOX_NEURAL_TEST_PYTHON"),
    reason="optional pinned neural toolchain",
)
def test_actual_cli_same_settings_profile_and_full_run(tmp_path, monkeypatch):
    from phonebox.cli.main import main
    from phonebox.eval import benchmark_data

    def synthetic_fetch(name, cache, *, remove_stress=False):
        return tiny_dataset()

    monkeypatch.setattr(benchmark_data, "load_dataset", synthetic_fetch)
    settings = tmp_path / "settings.json"
    declared = NeuralSettings(
        d_model=16,
        d_fft=32,
        layers=1,
        heads=2,
        dropout=0,
        batch_size=32,
        warmup_steps=2,
        max_epochs=1,
    )
    settings.write_text(json.dumps(declared.to_dict()))
    common = [
        "compare",
        "benchmark",
        "--dataset",
        "italian",
        "--system",
        "deepphonemizer",
        "--neural-python",
        os.environ["PHONEBOX_NEURAL_TEST_PYTHON"],
        "--neural-settings",
        str(settings),
    ]
    profile = tmp_path / "profile.json"
    assert (
        main(
            common
            + [
                "--neural-profile-only",
                "--work-dir",
                str(tmp_path / "profile"),
                "--output",
                str(profile),
            ]
        )
        == 0
    )
    profiled = json.loads(profile.read_text())
    assert profiled["settings"]["d_model"] == 16
    assert profiled["profile_only"] is True and "metrics" not in profiled
    complete = tmp_path / "complete.json"
    assert (
        main(
            common
            + ["--work-dir", str(tmp_path / "complete"), "--output", str(complete)]
        )
        == 0
    )
    measured = json.loads(complete.read_text())
    assert measured["settings"]["d_model"] == 16
    assert measured["settings"]["threads"] == 2
    assert measured["metrics"]["n_test"] == 1
    assert measured["training"]["optimizer_updates"] == 1
