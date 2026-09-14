"""Historical bytes retain their identity when the current renderer changes."""

import hashlib
import json
from pathlib import Path


def test_archived_cmudict_snapshot_integrity():
    archive = Path(__file__).resolve().parents[1] / "docs/archive/cmudict-489097f"
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["source_revision"] == "489097f15c2596b1ae8f958ec8d81336697316ed"
    assert set(manifest["files"]) == {
        "CMUDICT_COMPARISON.md",
        "cmudict-comparison.json",
    }
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == digest
    result = json.loads((archive / "cmudict-comparison.json").read_text())
    assert result["phonebox"]["revision"] == manifest["source_revision"]
    assert result["phonebox"]["dirty"] is False
    assert result["cmudict"]["commit"] == manifest["cmudict_commit"]
    assert result["cmudict"]["sha256"] == manifest["cmudict_sha256"]
    assert (
        result["parameters"]["em_iterations"]
        == manifest["multigram_em_iterations"]
        == 10
    )
    assert {condition["stress"] for condition in result["conditions"]} == {
        "preserved",
        "removed",
    }
    assert all(
        {model["model"] for model in condition["models"]}
        == {"G2PDecisionTree", "MultigramG2P"}
        for condition in result["conditions"]
    )
