"""Versions, runtime resources, and notices survive the source distribution."""

from __future__ import annotations

import ast
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


def _build(source: Path, output: Path, kind: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            kind,
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )


def test_wheel_and_sdist_versions_resources_and_notices(tmp_path):
    root = Path(__file__).parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    module = ast.parse((root / "phonebox/__init__.py").read_text())
    runtime_version = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    assert runtime_version == version
    notices = ["LICENSE", "LICENSE-UNICODE", "THIRD_PARTY_NOTICES.md"]
    resources = [root / "phonebox/py.typed", root / "phonebox/config/exemplars.json"]
    resources.extend(
        path
        for path in (root / "phonebox/config/locales").rglob("*")
        if path.suffix in {".json", ".xlit"}
    )
    resources.extend((root / "phonebox/configs").glob("*.yaml"))
    assert all(path.is_file() for path in resources)

    _build(root, tmp_path, "--sdist")
    sdist = next(tmp_path.glob("*.tar.gz"))
    release_docs = [root / "CHANGELOG.md"]
    release_docs.extend(
        path for path in (root / "docs").rglob("*") if path.suffix in {".md", ".json"}
    )
    extracted = tmp_path / "extracted"
    with tarfile.open(sdist, "r:gz") as archive:
        prefix = f"phonebox-{version}/"
        for name in notices:
            member = archive.extractfile(prefix + name)
            assert member is not None
            assert member.read() == (root / name).read_bytes()
        for path in resources + release_docs:
            member = archive.extractfile(prefix + path.relative_to(root).as_posix())
            assert member is not None
            assert member.read() == path.read_bytes()
        member = archive.extractfile(prefix + "PKG-INFO")
        assert member is not None
        metadata = Parser().parsestr(member.read().decode("utf-8"))
        assert metadata["Version"] == version
        assert metadata["License-Expression"] == "BSD-2-Clause AND Unicode-3.0"
        archive.extractall(extracted, filter="data")

    # Exercise the archive as the wheel's source, rather than building from
    # the checkout and assuming the source distribution is usable.
    _build(extracted / f"phonebox-{version}", tmp_path / "wheel", "--wheel")
    wheel = next((tmp_path / "wheel").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        assert metadata["Version"] == version
        assert metadata["License-Expression"] == "BSD-2-Clause AND Unicode-3.0"
        dist_info = metadata_name.removesuffix("METADATA")
        for name in notices:
            assert (
                archive.read(dist_info + "licenses/" + name)
                == (root / name).read_bytes()
            )
        for path in resources:
            assert archive.read(path.relative_to(root).as_posix()) == path.read_bytes()
