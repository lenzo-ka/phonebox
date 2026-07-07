#!/usr/bin/env python
"""
Configuration file loader for phonebox models.

Supports YAML and TOML config files for reproducible builds.

Example config.yaml:
    locale: en_US
    phoneset: cmu
    remove_stress: true
    remove_accents: true
    trainer: sklearn
    max_iterations: 100

Usage:
    from phonebox.config_loader import load_config

    config = load_config('config.yaml')
    model = train_from_config(config)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cartlet import PROB_HIGH_CONFIDENCE

from .constants import (
    DEFAULT_CASED,
    DEFAULT_LOCALE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_PHONESET,
    DEFAULT_STORE_DISTRIBUTIONS,
    FILE_ENCODING,
)


def load_config(path: str) -> dict[str, Any]:
    """
    Load configuration from file.

    Supports:
    - YAML (.yaml, .yml)
    - TOML (.toml)
    - JSON (.json)
    - Built-in presets (phonebox:preset_name)

    Args:
        path: Path to config file OR "phonebox:preset_name"

    Returns:
        Config dict
    """
    # Check for built-in preset (phonebox:name)
    if path.startswith("phonebox:"):
        preset_name = path.split(":", 1)[1]
        from .configs import get_builtin_config

        return get_builtin_config(preset_name)

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path_obj.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path_obj)
    if suffix == ".toml":
        return _load_toml(path_obj)
    if suffix == ".json":
        return _load_json(path_obj)
    raise ValueError(f"Unsupported config format: {suffix}")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for YAML configs: pip install pyyaml") from e

    with open(path, encoding=FILE_ENCODING) as f:
        return yaml.safe_load(f)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML config (tomllib is stdlib on the supported Python >= 3.11)."""
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON config."""
    with open(path, encoding=FILE_ENCODING) as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str, fmt: str | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Config dict
        path: Output path
        fmt: Format override ('yaml', 'toml', 'json', or None for auto)
    """
    path_obj = Path(path)

    if fmt is None:
        fmt = path_obj.suffix.lower().lstrip(".")

    if fmt in {"yaml", "yml"}:
        _save_yaml(config, path_obj)
        return
    if fmt == "toml":
        _save_toml(config, path_obj)
        return
    if fmt == "json":
        _save_json(config, path_obj)
        return
    raise ValueError(f"Unsupported format: {fmt}")


def _save_yaml(config: dict, path: Path) -> None:
    """Save as YAML."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(path, "w", encoding=FILE_ENCODING) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _save_toml(config: dict, path: Path) -> None:
    """Save as TOML."""
    try:
        import tomli_w
    except ImportError as e:
        raise ImportError("tomli-w required: pip install tomli-w") from e

    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def _save_json(config: dict, path: Path) -> None:
    """Save as JSON."""
    with open(path, "w", encoding=FILE_ENCODING) as f:
        json.dump(config, f, indent=2)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """
    Merge multiple configs (later overrides earlier).

    Args:
        *configs: Config dicts to merge

    Returns:
        Merged config
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


# Default config template
DEFAULT_CONFIG = {
    "locale": DEFAULT_LOCALE,
    "phoneset": DEFAULT_PHONESET,
    "remove_stress": True,
    "remove_accents": None,  # Auto-detect based on phoneset
    "filter_non_letters": False,  # Remove non-letter chars (except -'.)
    "cased": DEFAULT_CASED,
    # Native trainer is the safe default. sklearn builds a huge sparse
    # matrix and ran out of RAM on the 428k-entry French IPA lexicon
    # (~14 GB and counting before we killed it). Native scales linearly
    # in memory with the alignment table and finishes the same job in a
    # fraction of the wall-clock. Override via --trainer sklearn on the
    # CLI if you really want it.
    "trainer": "native",
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    # None here means "unset" — the trainer falls back to
    # constants.DEFAULT_MAX_COMBINATIONS (10000) downstream. max_combinations=0
    # (unbounded) used to be the default but blew up to 200+ GB RSS on French;
    # the 10000 cap keeps >99% of entries while staying memory-bounded.
    "max_combinations": None,
    "min_samples_split": DEFAULT_MIN_SAMPLES_SPLIT,
    "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
    "store_distributions": DEFAULT_STORE_DISTRIBUTIONS,
    "min_confidence": PROB_HIGH_CONFIDENCE,
    "criterion": "entropy",
    "parallel_align": False,
    "validation_split": 0.05,  # 5% for pruning
    "test_split": 0.05,  # 5% for evaluation (train gets 90%)
    "prune": False,  # Enable pruning with validation data
}
