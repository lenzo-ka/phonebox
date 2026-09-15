"""Reproducible lexicon train/test splits that keep spelling groups together."""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence

from phonebox.constants import (
    DEFAULT_MAX_TEST_ENTRIES,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
)


def spelling_group(word: str) -> str:
    """NFC-casefold key: every pronunciation variant and case alias of a spelling."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", word).casefold())


def split_lexicon(
    pairs: Sequence[tuple[str, list[str]]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    max_test: int = DEFAULT_MAX_TEST_ENTRIES,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Return (test_raw, train_raw) with whole spelling groups on one side.

    Every pronunciation variant and case alias of a spelling lands in the same
    split, so a test spelling is never seen in training. ``max_test`` and
    ``test_fraction`` count spelling groups, not pairs. Earlier releases shuffled
    individual pairs, which let variants of one spelling straddle the split;
    numbers produced that way are not reproduced by this function.
    """
    return split_lexicon_by_key(
        pairs,
        key=spelling_group,
        seed=seed,
        test_fraction=test_fraction,
        max_test=max_test,
    )


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
