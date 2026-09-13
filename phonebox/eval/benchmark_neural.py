"""Optional isolated DeepPhonemizer training, without a Torch runtime dependency.

The model and optimizer are Axel Springer's MIT-licensed implementation:
https://github.com/axelspringer/DeepPhonemizer/tree/5dce7e27556aef4426f5623baf6351d266a30a73.
See docs/BENCHMARK_TOOLCHAINS.md for the credited, reproducible training patch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phonebox.eval.benchmark import _tool_identity, _validate_dataset, _validate_receipt

if TYPE_CHECKING:
    from phonebox.eval.benchmark_data import PreparedDataset

SOURCE_REVISION = "5dce7e27556aef4426f5623baf6351d266a30a73"
PATCH_SHA256 = "459b1a4595b8ff8950a71f392e5f9610a80463e478ddb03c763e0b7eb63e8b81"


@dataclass(frozen=True)
class NeuralSettings:
    """Author autoregressive architecture with predeclared dev convergence.

    Modified settings describe a separate experiment; defaults are the accepted
    substantial-data protocol, never the reduced compatibility smoke model.
    """

    d_model: int = 512
    d_fft: int = 1024
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.0001
    warmup_steps: int = 10000
    max_epochs: int = 500
    plateau_patience: int = 10
    stop_patience: int = 30
    seed: int = 1729
    threads: int = 2

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "d_fft",
            "layers",
            "heads",
            "batch_size",
            "warmup_steps",
            "max_epochs",
            "plateau_patience",
            "stop_patience",
            "threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.threads > 2 or self.d_model % self.heads or self.d_model % 2:
            raise ValueError(
                "threads must be <=2; d_model must be even and divisible by heads"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            type(self.dropout) not in (int, float)
            or type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "dropout must be in [0,1); learning_rate must be finite and positive"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeuralSettings:
        """Validate a JSON settings object; omitted fields retain defaults."""
        if not isinstance(value, Mapping):
            raise ValueError("neural settings must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        if set(value) - allowed:
            raise ValueError("unknown neural settings fields")
        try:
            return cls(**dict(value))
        except (TypeError, OverflowError) as error:
            raise ValueError(
                "invalid neural settings field type or magnitude"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable declared settings."""
        return asdict(self)


def run_neural_training(
    dataset: PreparedDataset,
    work_dir: str | Path,
    *,
    python_executable: str | Path,
    device: str = "cpu",
    settings: NeuralSettings | None = None,
    profile_only: bool = False,
) -> dict[str, Any]:
    """Train in a pinned optional environment and return measured accounting.

    ``profile_only=True`` completes exactly one train epoch plus full shared dev
    evaluation. The training subprocess receives no test references and performs
    no test decoding; parent dataset validation still checks all splits. This
    profile is not a benchmark result. The work directory must be empty; subprocess diagnostics stay there.
    The external installation is checked against every pinned Python source file
    plus the recorded patch. No package is installed by this function.
    """
    _validate_dataset(dataset)
    if device not in ("cpu", "mps"):
        raise ValueError("neural device must be cpu or mps")
    if type(profile_only) is not bool:
        raise ValueError("profile_only must be boolean")
    if settings is not None and not isinstance(settings, NeuralSettings):
        raise ValueError("settings must be validated NeuralSettings")
    declared = settings or NeuralSettings()
    python = Path(python_executable).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(
            "neural Python must be an executable in the isolated toolchain"
        )
    identity = _tool_identity(python.resolve(), Path(str(python) + ".provenance.json"))
    if (
        identity["source_revision"] != SOURCE_REVISION
        or identity.get("build", {}).get("patch_sha256") != PATCH_SHA256
    ):
        raise ValueError(
            "Neural interpreter receipt requires the pinned source and training patch"
        )
    if not identity["receipt_binary_binding_verified"]:
        raise ValueError("Neural receipt must bind the actual interpreter hash")
    from phonebox.eval.cmudict_compare import _code_fingerprint, _git_revision

    root = Path(__file__).resolve().parents[2]
    revision, dirty = _git_revision(root)
    fingerprint = _code_fingerprint(root)
    metadata = deepcopy(dataset.metadata)
    directory = Path(work_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("neural work directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    job = {
        "name": dataset.name,
        "train": dataset.train,
        "dev": dataset.dev,
        "settings": declared.to_dict(),
        "device": device,
        "profile_only": profile_only,
    }
    (directory / "neural-job.json").write_text(
        json.dumps(job, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    if not profile_only:
        (directory / "neural-test.json").write_text(
            json.dumps(dataset.test, ensure_ascii=False), encoding="utf-8"
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = str(declared.threads)
    with (directory / "neural.log").open("wb") as log:
        completed = subprocess.run(
            [
                str(python),
                "-m",
                "phonebox.eval.benchmark_neural_worker",
                str(directory),
            ],
            cwd=directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise ValueError(
            "Neural toolchain failed; inspect neural.log in the work directory"
        )
    result = json.loads((directory / "neural-result.json").read_text(encoding="utf-8"))
    _validate_dataset(dataset)
    if dataset.metadata != metadata or fingerprint != _code_fingerprint(root):
        raise ValueError("Neural dataset or source changed during training")
    result["dataset"] = {**metadata, "name": dataset.name}
    result["provenance"]["phonebox"] = {
        "revision": revision,
        "dirty": dirty,
        "code_sha256": fingerprint,
    }
    result["provenance"]["interpreter_receipt"] = identity
    _validate_receipt(result)
    return result


def write_neural_patch(output: str | Path) -> str:
    """Atomically export the credited external patch; return its SHA-256.

    Apply with git apply to the exact pinned DeepPhonemizer source revision.
    This exports build evidence only and installs or modifies no toolchain.
    """
    from cartlet.io.utils import atomic_output_path

    from phonebox.eval.benchmark_neural_sources import PATCH_TEXT

    data = PATCH_TEXT.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != PATCH_SHA256:
        raise ValueError("Bundled neural patch hash is invalid")
    with atomic_output_path(str(output)) as temporary:
        Path(temporary).write_bytes(data)
    return PATCH_SHA256


@dataclass
class _DevSelection:
    """One owner for epoch selection and post-warmup patience."""

    warmup_steps: int
    patience: int
    best: tuple[float, float] = (math.inf, math.inf)
    nonimproving: int = 0

    def observe(self, per: float, wer: float, updates: int) -> tuple[bool, bool]:
        key = (per, wer)
        improved = key < self.best
        if improved:
            self.best = key
            self.nonimproving = 0
        elif updates >= self.warmup_steps:
            self.nonimproving += 1
        return improved, self.nonimproving >= self.patience
