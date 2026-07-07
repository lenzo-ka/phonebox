"""Reproducible lexicon train/test split (matches compare_g2p)."""

from __future__ import annotations

import random
from collections.abc import Sequence

from phonebox.constants import (
    DEFAULT_MAX_TEST_ENTRIES,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
)


def split_lexicon(
    pairs: Sequence[tuple[str, list[str]]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Return (test_raw, train_raw) after the same shuffle/slice as compare_g2p."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_test = min(max_test, max(1, int(len(shuffled) * test_fraction)))
    return shuffled[:n_test], shuffled[n_test:]
