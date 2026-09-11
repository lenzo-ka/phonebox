"""Reproducible lexicon train/test split (matches compare_g2p)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

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


def split_lexicon_by_key(
    pairs: Sequence[tuple[str, list[str]]],
    *,
    key: Callable[[str], str],
    seed: int = DEFAULT_SPLIT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Split complete key groups, preventing normalized-word leakage."""
    groups: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for pair in pairs:
        groups[key(pair[0])].append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = min(max_test, max(1, int(len(keys) * test_fraction)))
    test_keys = set(keys[:n_test])
    test = [
        pair
        for group_key in keys
        if group_key in test_keys
        for pair in groups[group_key]
    ]
    train = [
        pair
        for group_key in keys
        if group_key not in test_keys
        for pair in groups[group_key]
    ]
    return test, train
