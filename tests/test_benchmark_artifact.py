"""The tracked measured snapshot is the complete four-system matrix, regenerated exactly."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from phonebox.eval.benchmark import _validate_receipt
from phonebox.eval.benchmark_report import render_benchmark_report

ROOT = Path(__file__).resolve().parents[1]
ROWS_DIR = ROOT / "docs" / "benchmarks"
REPORT = ROOT / "docs" / "G2P_BENCHMARKS.md"
CLASSICAL_SYSTEMS = {"cart", "multigram", "sequitur", "phonetisaurus"}
# The neural comparison is complete for one condition; the other three are owed.
NEURAL_CONDITIONS = {("cmudict", False)}
NATIVE_REVISION = "de8fbc3f966c191eee59ff48d3f0347c65ee55ef"
EXTERNAL_REVISION = "a7bfbeb7ae9e84bdd47f5ba8f1c260430148c3ef"
# The structural guard is the benchmark receipt validator applied to the whole
# row: absolute POSIX or Windows paths anywhere, and a fixed set of identity
# keys. It is a shape check; an identity under an unlisted key, or a hostname
# written into prose, is not detectable by shape. This denylist names the
# identities of the machines that produced the tracked rows, so those specific
# leaks fail even under an unlisted key. It is enumerated, not a property.
KNOWN_IDENTITIES = re.compile(r"lenzo|shrub|/Users/|/private/|/home/|/tmp/", re.I)


def load_rows() -> list[dict[str, Any]]:
    paths = sorted(ROWS_DIR.glob("*.json"))
    assert len(paths) == 16 + len(NEURAL_CONDITIONS)
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not KNOWN_IDENTITIES.search(text), f"{path.name} names a local identity"
        row = json.loads(text)
        _validate_receipt(row)
        rows.append(row)
    return rows


def test_row_guard_rejects_absolute_paths_and_identity_keys():
    for construction in (
        {"provenance": {"note": "/opt/cache/run"}},
        {"settings": {"work": "D:\\data\\bench"}},
        {"timings": {"cwd": "relative/still-an-identity-key"}},
        {"training": {"hostname": "x"}},
    ):
        with pytest.raises(ValueError):
            _validate_receipt(construction)


def test_snapshot_is_the_complete_four_system_matrix():
    conditions: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in load_rows():
        dataset = row["dataset"]
        key = (dataset["name"], dataset["preparation"]["remove_stress"])
        conditions.setdefault(key, []).append(row)
    assert set(conditions) == {
        ("cmudict", False),
        ("cmudict", True),
        ("french", False),
        ("italian", False),
    }
    for key, rows in conditions.items():
        expected = CLASSICAL_SYSTEMS | (
            {"deepphonemizer"} if key in NEURAL_CONDITIONS else set()
        )
        assert {row["system"] for row in rows} == expected
        reference = rows[0]["dataset"]
        assert reference["counts"]["train"]["words"] > 50_000
        assert reference["counts"]["test"]["words"] > 5_000
        for row in rows:
            assert row["dataset"]["prepared_sha256"] == reference["prepared_sha256"]
            assert row["dataset"]["counts"] == reference["counts"]
            assert row["dataset"]["sources"] == reference["sources"]
            assert row["metrics"]["n_test"] == reference["counts"]["test"]["words"]


def test_rows_are_model_only_with_full_accounting_and_pinned_provenance():
    for row in load_rows():
        settings, training, provenance = (
            row["settings"],
            row["training"],
            row["provenance"],
        )
        assert settings["dictionary_lookup"] is False
        assert settings["evaluation_scope"].startswith("held-out spellings; model only")
        assert row["metrics"]["prediction_errors"] == 0
        assert training["model_bytes"] > 0
        assert provenance["dirty"] is False
        if row["system"] == "deepphonemizer":
            # From-scratch author-protocol training on the native source;
            # selection used the full shared development split only.
            assert provenance["revision"] == NATIVE_REVISION
            assert training["retained_entries"] > 50_000
            assert training["dev_entries"] > 5_000
            assert training["dictionary_entries"] == 0
            assert training["nonfinite_batches"] == 0
            assert 0 < training["selected_epoch"] <= training["completed_epochs"]
            continue
        assert training["supplied_entries"] > 50_000
        if row["system"] in {"cart", "multigram"}:
            assert settings["use_dict_fallback"] is False
            assert provenance["revision"] == NATIVE_REVISION
            assert (
                training["retained_entries"] + training["skipped_entries"]
                == training["supplied_entries"]
            )
            assert training["convergence"]["converged"] is True
        else:
            assert provenance["revision"] == EXTERNAL_REVISION
            assert training["retained_entries"] is None
            assert "retention_note" in training


def test_report_is_the_exact_rendering_and_defers_the_neural_comparison():
    rendered = render_benchmark_report(load_rows())
    assert rendered == REPORT.read_text(encoding="utf-8")
    assert rendered.count(
        "Not measured in this artifact: DeepPhonemizer autoregressive."
    ) == 4 - len(NEURAL_CONDITIONS)
    assert rendered.count("| DeepPhonemizer autoregressive |") == len(NEURAL_CONDITIONS)
