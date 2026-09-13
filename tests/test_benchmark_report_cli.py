"""Exercise report validation and real CLI input/output error boundaries."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phonebox.eval.benchmark_report import render_benchmark_report


def result(system="cart"):
    counts = {
        split: {
            "source_entries": 2,
            "prepared_entries": 2,
            "words": 2,
            "duplicates_removed": 0,
        }
        for split in ["train", "dev", "test"]
    }
    return {
        "schema_version": 1,
        "system": system,
        "dataset": {
            "name": "italian",
            "preparation": {"remove_stress": False},
            "counts": counts,
            "prepared_sha256": {
                split: digit * 64
                for split, digit in zip(["train", "dev", "test"], "abc", strict=True)
            },
        },
        "metrics": {
            "n_test": 2,
            "wer_relaxed_pct": 50.0,
            "per_variant_pct": 25.0,
            "empty_predictions": 1,
            "prediction_errors": 1,
        },
        "timings": {
            "training_seconds": 0.2,
            "prediction_seconds": 0.01,
            "export_seconds": 0.0,
        },
        "training": {"model_bytes": 123},
    }


def cli(tmp_path, *arguments):
    root = Path(__file__).parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root))
    prologue = (
        "import phonebox; from pathlib import Path; assert Path(phonebox.__file__).parent == Path("
        + repr(str(root / "phonebox"))
        + "); from phonebox.cli.main import main; raise SystemExit(main())"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            prologue,
            "compare",
            "benchmark-report",
            *map(str, arguments),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_valid_report_keeps_missing_predictions_and_missing_systems_explicit():
    rendered = render_benchmark_report([result("phonetisaurus"), result()])
    assert "| Phonebox CART | 50.00 | 25.00 | 1 |" in rendered
    assert "| Phonetisaurus | 50.00 | 25.00 | 1 |" in rendered
    assert "Not measured in this artifact: Phonebox n:m, Sequitur." in rendered
    assert "Missing predictions remain errors." in rendered
    assert "Held-out WER (%) | Held-out PER (%)" in rendered
    assert "dictionary lookup disabled" in rendered
    assert "a" * 64 in rendered
    assert rendered.index("| Phonebox CART |") < rendered.index("| Phonetisaurus |")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "duplicate"),
        ("digest", "different prepared splits"),
        ("counts", "different dataset populations"),
        ("population", "evaluation population"),
    ],
)
def test_incompatible_results_refused(mutation, message):
    rows = [result(), result("multigram")]
    if mutation == "duplicate":
        rows[1]["system"] = "cart"
    elif mutation == "digest":
        rows[1]["dataset"]["prepared_sha256"]["train"] = "d" * 64
    elif mutation == "counts":
        rows[1]["dataset"]["counts"]["train"] = {
            "source_entries": 3,
            "prepared_entries": 3,
            "words": 3,
            "duplicates_removed": 0,
        }
    else:
        rows[1]["metrics"]["n_test"] = 1
    with pytest.raises(ValueError, match=message):
        render_benchmark_report(rows)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("metrics", "n_test"), True),
        (("metrics", "n_test"), 0),
        (("metrics", "empty_predictions"), 0.5),
        (("metrics", "empty_predictions"), 3),
        (("metrics", "prediction_errors"), -1),
        (("metrics", "prediction_errors"), 3),
        (("metrics", "wer_relaxed_pct"), 101),
        (("metrics", "wer_relaxed_pct"), float("nan")),
        (("metrics", "per_variant_pct"), -1),
        (("dataset", "counts", "test", "words"), True),
        (("dataset", "counts", "train", "duplicates_removed"), 1),
        (("dataset", "prepared_sha256", "test"), ""),
        (("dataset", "preparation", "remove_stress"), []),
        (("training", "model_bytes"), 1.5),
        (("timings", "training_seconds"), float("inf")),
        (("timings", "export_seconds"), -1),
        (("timings", "export_seconds"), float("nan")),
    ],
)
def test_malformed_aggregate_counts_and_numbers_refused(path, value):
    row = result()
    target = row
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        render_benchmark_report([row])


def test_per_can_exceed_100_without_becoming_invalid():
    row = result()
    row["metrics"]["per_variant_pct"] = 150
    assert "| 150.00 |" in render_benchmark_report([row])


def test_training_total_includes_native_export_once_and_external_export_once():
    native, external = result(), result("sequitur")
    native["timings"].update(training_seconds=1.0, export_seconds=0.25)
    external["timings"].update(training_seconds=1.25, export_seconds=0.0)
    rendered = render_benchmark_report([native, external])
    assert "Train + export (s)" in rendered
    assert "| Phonebox CART | 50.00 | 25.00 | 1 | 1.25 |" in rendered
    assert "| Sequitur | 50.00 | 25.00 | 1 | 1.25 |" in rendered


def test_training_total_cannot_overflow_finite_components():
    row = result()
    row["timings"].update(training_seconds=1e308, export_seconds=1e308)
    with pytest.raises(ValueError, match="training plus export time"):
        render_benchmark_report([row])


@pytest.mark.parametrize("bad", [None, [], {}, {"schema_version": 1}, "wrong shape"])
def test_wrong_structure_normalized_to_value_error(bad):
    with pytest.raises(ValueError):
        render_benchmark_report([bad])


def test_actual_cli_valid_output_matches_library(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result()))
    output = tmp_path / "report.md"
    completed = cli(tmp_path, path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == render_benchmark_report([result()])


@pytest.mark.parametrize(
    "failure",
    [
        "malformed-json",
        "bad-counter",
        "duplicate",
        "missing-file",
        "output-directory",
        "alias",
        "symlink",
        "hardlink",
    ],
)
def test_actual_cli_expected_errors_preserve_existing_files(tmp_path, failure):
    path = tmp_path / "result.json"
    original = json.dumps(result())
    path.write_text(original)
    output = tmp_path / "report.md"
    output.write_text("KEEP")
    arguments = [path]
    if failure == "malformed-json":
        path.write_text("{")
    elif failure == "bad-counter":
        bad = copy.deepcopy(result())
        bad["metrics"]["empty_predictions"] = 3
        path.write_text(json.dumps(bad))
    elif failure == "duplicate":
        arguments.append(path)
    elif failure == "missing-file":
        arguments = [tmp_path / "missing.json"]
    elif failure == "output-directory":
        output.unlink()
        output.mkdir()
    elif failure == "alias":
        output = path
    elif failure == "symlink":
        output.unlink()
        output.symlink_to(path)
    elif failure == "hardlink":
        output.unlink()
        os.link(path, output)
    before = path.read_bytes()
    completed = cli(tmp_path, *arguments, "--output", output)
    assert completed.returncode == 2, completed.stderr
    assert "Traceback" not in completed.stderr
    assert path.read_bytes() == before
    if failure in {"malformed-json", "bad-counter", "duplicate", "missing-file"}:
        assert output.read_text() == "KEEP"
    elif failure == "output-directory":
        assert output.is_dir() and not list(output.iterdir())
