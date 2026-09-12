"""Primary dictionary-to-decision-tree training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from .config_loader import DEFAULT_CONFIG, load_config, merge_configs
from .constants import (
    DEFAULT_MAX_COMBINATIONS,
    DEFAULT_STORE_DISTRIBUTIONS,
    DEFAULT_TRAIN_PARALLEL_ALIGN,
    DEFAULT_TRAIN_PHONESET,
    DEFAULT_TRAIN_PRUNE,
    DEFAULT_TRAIN_REMOVE_STRESS,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAINER,
    DICT_ENCODING,
)
from .core.g2p_model import G2PDecisionTree
from .locale_resolution import canonical_locale
from .utils.io import paths_refer_to_same_file


@dataclass(frozen=True)
class TrainingResult:
    """A trained model, its metrics, and any artifacts written by the workflow."""

    model: G2PDecisionTree
    metrics: dict[str, Any]
    output_path: Path | None
    alignments_path: Path | None


def default_alignments_path(output: Path) -> Path:
    """Derive the adjacent alignment checkpoint name for a model output."""
    stem = output.stem.removesuffix(".g2p")
    return output.with_name(f"{stem}_alignments.txt")


def validate_training_paths(
    dictionary: Path, output: Path | None, alignments: Path | None
) -> None:
    """Reject artifact aliases before training can read or write any path."""
    named = [("dictionary", dictionary)]
    if output is not None:
        named.append(("output", output))
    if alignments is not None:
        named.append(("alignments", alignments))
    for index, (left_name, left) in enumerate(named):
        for right_name, right in named[index + 1 :]:
            if paths_refer_to_same_file(left, right):
                raise ValueError(
                    f"{left_name} and {right_name} must be different files"
                )


def train_g2p(
    dictionary: str | Path,
    *,
    locale: str,
    phoneset: str = DEFAULT_TRAIN_PHONESET,
    output: str | Path | None = None,
    alignments_out: str | Path | None = None,
    remove_stress: bool = DEFAULT_TRAIN_REMOVE_STRESS,
    prune: bool = DEFAULT_TRAIN_PRUNE,
    validation_split: float = DEFAULT_TRAIN_VALIDATION_SPLIT,
    test_split: float = DEFAULT_TRAIN_TEST_SPLIT,
    trainer: str = DEFAULT_TRAINER,
    parallel_align: bool = DEFAULT_TRAIN_PARALLEL_ALIGN,
    max_combinations: int | None = DEFAULT_MAX_COMBINATIONS,
    width: int | None = None,
    store_distributions: bool = DEFAULT_STORE_DISTRIBUTIONS,
    verbose: bool = False,
    **model_options: Any,
) -> TrainingResult:
    """Train one decision-tree G2P from a pronunciation dictionary.

    This is the shared workflow behind :meth:`phonebox.G2P.train` and
    ``phonebox train``. By default it uses IPA-tagged phones, the native tree
    trainer, serial alignment, a 5% pruning split, stored leaf distributions,
    and preserved stress. Pass ``phoneset="cmu"`` for CMU stress/join rules.

    An alignment checkpoint is derived from ``output`` when one is supplied.
    Pass an explicit ``alignments_out`` to write a checkpoint without exporting
    a model. When both are omitted, no files are written.
    """
    dictionary_path = Path(dictionary)
    if not dictionary_path.is_file():
        raise FileNotFoundError(
            f"pronunciation dictionary not found: {dictionary_path}"
        )
    output_path = Path(output) if output is not None else None
    alignments_path = (
        Path(alignments_out)
        if alignments_out is not None
        else (default_alignments_path(output_path) if output_path else None)
    )
    validate_training_paths(dictionary_path, output_path, alignments_path)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if alignments_path:
        alignments_path.parent.mkdir(parents=True, exist_ok=True)

    model = G2PDecisionTree(
        locale=canonical_locale(locale),
        phoneset_name=phoneset,
        remove_stress=remove_stress,
        trainer=trainer,
        parallel_align=parallel_align,
        max_combinations=max_combinations,
        width=width,
        store_distributions=store_distributions,
        verbose=verbose,
        **model_options,
    )
    metrics = model.train_from_dict(
        str(dictionary_path),
        encoding=DICT_ENCODING,
        validation_split=validation_split if prune else 0.0,
        test_split=test_split,
        prune=prune,
        alignments_path=str(alignments_path) if alignments_path else None,
    )
    if output_path:
        model.export(str(output_path))
    return TrainingResult(model, metrics, output_path, alignments_path)


def train_g2p_from_config(
    config: str | Path | dict[str, Any],
) -> TrainingResult:
    """Train through the primary workflow from a configuration mapping or file."""
    supplied = load_config(str(config)) if isinstance(config, (str, Path)) else config
    if "phoneset_name" in supplied:
        raise ValueError(
            "Unsupported training config option phoneset_name; use phoneset"
        )
    accepted = {
        name
        for function in (train_g2p, G2PDecisionTree)
        for name, parameter in signature(function).parameters.items()
        if parameter.kind not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
    }
    unknown = set(supplied) - accepted
    if unknown:
        raise ValueError(
            "Unknown training config options: " + ", ".join(sorted(map(str, unknown)))
        )
    options = merge_configs(DEFAULT_CONFIG, supplied)
    dictionary = options.pop("dictionary", None)
    if not dictionary:
        raise ValueError("Config must specify 'dictionary' path")
    return train_g2p(dictionary, **options)


__all__ = [
    "TrainingResult",
    "default_alignments_path",
    "train_g2p",
    "train_g2p_from_config",
    "validate_training_paths",
]
