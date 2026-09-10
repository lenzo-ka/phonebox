#!/usr/bin/env python
"""Build models from configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .training import train_g2p_from_config


def train_from_config(config: str | Path | dict[str, Any]) -> Any:
    """
    Train model from configuration dict or file.

    Args:
        config: Configuration dict OR path to config file (YAML/TOML/JSON)

    Returns:
        Trained model
    """
    return train_g2p_from_config(config).model
