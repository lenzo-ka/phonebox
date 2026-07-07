#!/usr/bin/env python
"""Build models from configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_loader import DEFAULT_CONFIG, load_config, merge_configs
from .dictionary import Dictionary


def train_from_config(config: str | Path | dict[str, Any]) -> Any:
    """
    Train model from configuration dict or file.

    Args:
        config: Configuration dict OR path to config file (YAML/TOML/JSON)

    Returns:
        Trained model
    """
    # If config is a path, load it
    if isinstance(config, (str, Path)):
        config = load_config(str(config))

    # Merge with defaults
    full_config = merge_configs(DEFAULT_CONFIG, config)

    # Load dictionary
    dict_path = full_config.get("dictionary")
    if not dict_path:
        raise ValueError("Config must specify 'dictionary' path")

    dict_obj = Dictionary(path=dict_path, locale=full_config["locale"])
    dict_obj.load()

    # Train model. Every key below is guaranteed by DEFAULT_CONFIG (the sole
    # source of default values); only genuinely optional keys use .get().
    model = dict_obj.train_g2p_model(
        locale=full_config["locale"],
        phoneset=full_config["phoneset"],
        remove_stress=full_config["remove_stress"],
        remove_accents=full_config["remove_accents"],
        filter_non_letters=full_config["filter_non_letters"],
        cased=full_config["cased"],
        output=full_config.get("output"),
        trainer=full_config["trainer"],
        max_iterations=full_config["max_iterations"],
        max_combinations=full_config["max_combinations"],
        min_samples_split=full_config["min_samples_split"],
        min_samples_leaf=full_config["min_samples_leaf"],
        store_distributions=full_config["store_distributions"],
        min_confidence=full_config["min_confidence"],
        criterion=full_config["criterion"],
        parallel_align=full_config["parallel_align"],
        verbose=full_config.get("verbose", True),
        prune=full_config["prune"],
        validation_split=full_config["validation_split"],
        test_split=full_config["test_split"],
    )

    return model
