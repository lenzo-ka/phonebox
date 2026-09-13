"""Bind real temporary Git source and binaries through the public API and CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phonebox.eval.benchmark import _tool_identity
from phonebox.eval.benchmark_provenance import write_tool_receipt


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    tracked = source / "implementation.py"
    tracked.write_text("print('source')\n")
    subprocess.run(["git", "-C", str(source), "add", "implementation.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    binary = tmp_path / "tool"
    binary.write_bytes(b"observed binary")
    return binary, source, revision


def produce(inputs, **kwargs):
    binary, source, revision = inputs
    return write_tool_receipt(
        binary,
        source,
        declared_version="source-1",
        expected_revision=revision,
        **kwargs,
    )


def test_actual_source_and_binary_binding_roundtrip(inputs, tmp_path):
    binary, source, revision = inputs
    result = produce(
        inputs,
        build={
            "compiler": "compiler-1",
            "dependency": {
                "version": "2",
                "upstream": "https://example.invalid/source",
            },
        },
    )
    receipt = Path(str(binary) + ".provenance.json")
    assert result == json.loads(receipt.read_text())
    assert result["source_revision"] == revision
    assert result["build"]["source_dirty"] is False
    assert (
        result["build"]["source_head_tree"]
        == subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
    )
    assert str(tmp_path) not in json.dumps(result, allow_nan=False)
    identity = _tool_identity(binary, receipt)
    assert identity["receipt_binary_binding_verified"] is True
    assert identity["sha256"] == result["build"]["executable_sha256"]
    binary.write_bytes(b"replacement binary")
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="hash differs"):
        _tool_identity(binary, receipt)
    assert receipt.read_bytes() == before


def test_tracked_diff_is_captured_and_untracked_source_rejected(inputs):
    binary, source, _ = inputs
    clean = produce(inputs)
    (source / "implementation.py").write_text("print('changed')\n")
    changed = produce(inputs)
    assert changed["build"]["source_dirty"] is True
    assert (
        changed["build"]["source_diff_sha256"] != clean["build"]["source_diff_sha256"]
    )
    assert changed["build"]["source_head_tree"] == clean["build"]["source_head_tree"]
    receipt = Path(str(binary) + ".provenance.json")
    before = receipt.read_bytes()
    (source / "untracked.py").write_text("outside Git diff")
    with pytest.raises(ValueError, match="untracked"):
        produce(inputs)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize(
    "build",
    [
        {"compiler": float("nan")},
        {"compiler": "/private/tmp/build"},
        {"nested": {"command": "compile"}},
        {"nested": ("PREFIX=/private/tmp/build",)},
        {"executable_sha256": "override"},
        [],
        {"nonjson": {1, 2}},
    ],
)
def test_invalid_build_preserves_existing_output(inputs, tmp_path, build):
    target = tmp_path / "receipt.json"
    target.write_text("SENTINEL")
    with pytest.raises(ValueError):
        produce(inputs, build=build, output=target)
    assert target.read_text() == "SENTINEL"


def test_wrong_revision_and_source_preserve_output(inputs, tmp_path):
    binary, _, _ = inputs
    target = tmp_path / "receipt.json"
    target.write_text("SENTINEL")
    with pytest.raises(ValueError, match="expected_revision"):
        write_tool_receipt(
            binary,
            inputs[1],
            declared_version="1",
            expected_revision="wrong",
            output=target,
        )
    with pytest.raises(ValueError, match="Git repository"):
        write_tool_receipt(
            binary,
            tmp_path,
            declared_version="1",
            expected_revision=inputs[2],
            output=target,
        )
    assert target.read_text() == "SENTINEL"


@pytest.mark.parametrize("which", ["binary", "tracked", "hardlink"])
def test_input_output_aliases_rejected(inputs, tmp_path, which):
    binary, source, _ = inputs
    target = binary if which == "binary" else source / "implementation.py"
    if which == "hardlink":
        target = tmp_path / "alias"
        os.link(binary, target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="output must differ"):
        produce(inputs, output=target)
    assert target.read_bytes() == before


def test_unbound_receipt_is_explicit_and_present_null_binding_rejected(inputs):
    binary, _, _ = inputs
    receipt = Path(str(binary) + ".provenance.json")
    assert _tool_identity(binary, receipt)["receipt_binary_binding_verified"] is False
    receipt.write_text(json.dumps({"version": "legacy", "source_revision": "pin"}))
    assert _tool_identity(binary, receipt)["receipt_binary_binding_verified"] is False
    receipt.write_text(json.dumps({"build": {"executable_sha256": None}}))
    with pytest.raises(ValueError, match="hash differs"):
        _tool_identity(binary, receipt)


def cli(inputs, tmp_path, *args):
    root = Path(__file__).parents[1]
    guard = (
        "import phonebox; from pathlib import Path; assert Path(phonebox.__file__).parent == Path("
        + repr(str(root / "phonebox"))
        + "); from phonebox.cli.main import main; raise SystemExit(main())"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            guard,
            "compare",
            "benchmark-receipt",
            str(inputs[0]),
            "--source-dir",
            str(inputs[1]),
            "--version",
            "source-1",
            "--expected-revision",
            inputs[2],
            *map(str, args),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
    )


def test_actual_cli_produces_receipt_and_invalid_inputs_preserve_it(inputs, tmp_path):
    target = tmp_path / "receipt.json"
    result = cli(
        inputs,
        tmp_path,
        "--build-metadata",
        '{"compiler":"fixture"}',
        "--output",
        target,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(target.read_text())
    assert _tool_identity(inputs[0], target)["receipt_binary_binding_verified"] is True
    before = target.read_bytes()
    for metadata in [
        "[]",
        '{"compiler":NaN}',
        '{"compiler":"PREFIX=/private/tmp/build"}',
    ]:
        bad = cli(inputs, tmp_path, "--build-metadata", metadata, "--output", target)
        assert bad.returncode == 2
        assert "Traceback" not in bad.stderr
        assert target.read_bytes() == before
