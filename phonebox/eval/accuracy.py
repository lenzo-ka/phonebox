"""Train/test accuracy workflows for pronunciation dictionaries."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from phonebox.constants import DEFAULT_TRAINER, DICT_ENCODING
from phonebox.core.g2p_model import G2PDecisionTree
from phonebox.lexicon import parse_dict_line


@dataclass(frozen=True)
class AccuracyResult:
    """Counts and percentage accessors from one held-out evaluation."""

    training_entries: int
    test_entries: int
    correct_words: int
    correct_phones: int
    total_phones: int

    @property
    def word_accuracy(self) -> float:
        """Percentage of test words predicted exactly."""
        return 100 * self.correct_words / self.test_entries

    @property
    def phone_accuracy(self) -> float:
        """Position-wise phone accuracy using the historical metric."""
        return 100 * self.correct_phones / self.total_phones


def load_pronunciation_entries(path) -> list[tuple[str, list[str]]]:
    """Load whitespace-separated word/pronunciation entries, skipping alternates."""
    entries = []
    with open(path, encoding=DICT_ENCODING) as infile:
        for raw in infile:
            parsed = parse_dict_line(raw)
            if parsed is not None and raw.split()[0] == parsed[0]:
                entries.append(parsed)
    return entries


def evaluate_accuracy(
    entries: Iterable[tuple[str, list[str]]],
    *,
    locale: str = "en_US",
    phoneset: str = "cmu",
    train_fraction: float = 0.95,
    seed: int = 42,
    width: int | None = None,
    parallel_align: bool = False,
    trainer: str = DEFAULT_TRAINER,
) -> AccuracyResult:
    """Train natively by default on a deterministic split and report accuracy.

    The optional sklearn trainer requires ``phonebox[sklearn]``.
    """
    pairs = list(entries)
    random.Random(seed).shuffle(pairs)
    split = int(len(pairs) * train_fraction)
    train, test = pairs[:split], pairs[split:]
    if not train or not test:
        raise ValueError("accuracy evaluation requires non-empty train and test splits")
    model = G2PDecisionTree(
        locale=locale,
        phoneset_name=phoneset,
        remove_stress=True,
        verbose=False,
        trainer=trainer,
        parallel_align=parallel_align,
        width=width,
    )
    model.load_prondict(f"{word}\t{' '.join(phones)}" for word, phones in train)
    model.align()
    model.train()
    correct_words = correct_phones = total_phones = 0
    for word, expected in test:
        expected = model.vectorizer.cook_phones(expected)
        predicted = model.pronounce(word)
        correct_words += predicted == expected
        correct_phones += sum(a == b for a, b in zip(predicted, expected, strict=False))
        total_phones += max(len(predicted), len(expected))
    return AccuracyResult(
        len(train), len(test), correct_words, correct_phones, total_phones
    )


__all__ = ["AccuracyResult", "evaluate_accuracy", "load_pronunciation_entries"]
