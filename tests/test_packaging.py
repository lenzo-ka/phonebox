"""Distribution artifacts include every applicable license notice."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_wheel_and_sdist_include_unicode_notices(tmp_path):
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        assert any(name.endswith("/licenses/LICENSE-UNICODE") for name in names)
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
        assert "License-Expression: BSD-2-Clause AND Unicode-3.0" in metadata

    sdist = next(tmp_path.glob("*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/LICENSE-UNICODE") for name in names)
        assert any(name.endswith("/THIRD_PARTY_NOTICES.md") for name in names)
