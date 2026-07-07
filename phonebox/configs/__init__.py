"""
Built-in configuration presets for common use cases.

Available presets:
- pocketsphinx: CMU Arpabet, PocketSphinx compatible (default)
- xsampa: X-SAMPA phoneset, multilingual
- fast: Quick training for experimentation
- accurate: Thorough training (more EM iterations); slower, higher-quality

Usage:
    from phonebox.configs import get_builtin_config

    config = get_builtin_config('pocketsphinx')
    model = train_from_config(config)

Or via CLI:
    phonebox model build --config phonebox:pocketsphinx \
        --dict mydict.txt --output model.jsonl.gz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config_loader import load_config

BUILTIN_CONFIGS = {
    "defaults": "defaults.yaml",
    "pocketsphinx": "pocketsphinx.yaml",
    "xsampa": "xsampa.yaml",
    "fast": "fast.yaml",
    "accurate": "accurate.yaml",
}


def get_builtin_config(name: str) -> dict[str, Any]:
    """
    Load a built-in configuration preset.

    Args:
        name: Preset name (pocketsphinx, xsampa, fast, accurate)

    Returns:
        Config dict
    """
    if name not in BUILTIN_CONFIGS:
        available = ", ".join(BUILTIN_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    config_file = BUILTIN_CONFIGS[name]
    config_path = Path(__file__).parent / config_file

    return load_config(str(config_path))
