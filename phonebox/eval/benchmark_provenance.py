"""Public, path-free source and binary receipts for optional benchmark tools."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from cartlet.io.utils import atomic_output_path

from phonebox.eval.benchmark import _validate_receipt
from phonebox.eval.cmudict_compare import sha256_file
from phonebox.utils.io import paths_refer_to_same_file


def _git(source: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(source), *arguments], capture_output=True, check=False
    )
    if completed.returncode:
        raise ValueError("Cannot inspect the tool source Git repository")
    return completed.stdout


def write_tool_receipt(
    executable: str | Path,
    source_dir: str | Path,
    *,
    declared_version: str,
    expected_revision: str,
    build: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Write and return a verified-source, observed-binary provenance receipt.

    ``declared_version`` is caller-declared source version, not inferred from
    executable output. The actual Git HEAD must equal ``expected_revision``.
    The receipt records HEAD's tree and a hash of tracked working-tree changes;
    nonignored untracked source files are rejected because Git's diff omits them.
    Additional build facts must be finite JSON values without deployment paths
    or identities. Generated binding fields cannot be overridden.

    Default output is beside the resolved executable, with ``.provenance.json``
    appended. Validation precedes atomic replacement. The returned record omits
    local input/output paths and binds its metadata to the observed binary hash.
    """
    if not isinstance(declared_version, str) or not declared_version.strip():
        raise ValueError("declared_version must be a nonempty string")
    if not isinstance(expected_revision, str) or not expected_revision:
        raise ValueError("expected_revision must be the full source Git revision")
    binary = Path(executable).resolve(strict=True)
    source = Path(source_dir).resolve(strict=True)
    if not binary.is_file() or not source.is_dir():
        raise ValueError("Receipt inputs require a binary file and source directory")
    source = Path(_git(source, "rev-parse", "--show-toplevel").decode().strip())
    revision = _git(source, "rev-parse", "HEAD").decode().strip()
    if revision != expected_revision:
        raise ValueError("Tool source Git HEAD differs from expected_revision")
    if _git(source, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError(
            "Commit or ignore untracked source files before writing a receipt"
        )
    tree = _git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    diff = _git(source, "diff", "--binary", "HEAD", "--")
    facts = {
        "executable_sha256": sha256_file(binary),
        "version_origin": "caller-declared source version",
        "source_head_tree": tree,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_dirty": bool(diff),
    }
    if build is not None:
        if not isinstance(build, Mapping):
            raise ValueError("build metadata must be an object")
        if facts.keys() & build.keys():
            raise ValueError(
                "build metadata cannot override observed source/binary fields"
            )
        facts.update(deepcopy(dict(build)))
    result = {
        "version": declared_version,
        "source_revision": revision,
        "build": facts,
    }
    try:
        payload = json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "build metadata must contain JSON-serializable values"
        ) from exc
    result = json.loads(payload)
    _validate_receipt(result)
    destination = (
        Path(output) if output is not None else Path(str(binary) + ".provenance.json")
    )
    if paths_refer_to_same_file(binary, destination):
        raise ValueError("Receipt output must differ from its executable")
    tracked = _git(source, "ls-files", "-z").decode().split("\0")
    if any(
        name and paths_refer_to_same_file(source / name, destination)
        for name in tracked
    ):
        raise ValueError("Receipt output must differ from tracked source files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(str(destination)) as temporary:
        Path(temporary).write_text(payload, encoding="utf-8")
    return result
