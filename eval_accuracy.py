#!/usr/bin/env python
"""Evaluate G2P accuracy on CMUdict with train/test split."""

import random

from phonebox.constants import DICT_ENCODING
from phonebox.core.g2p_model import G2PDecisionTree


def main():
    # Load CMUdict
    dict_path = "data/cmudict/cmudict.dict"
    entries = []
    with open(dict_path, encoding=DICT_ENCODING) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    word, phones = parts
                    # Skip alternates like WORD(2)
                    if "(" not in word:
                        entries.append((word, phones.split()))

    print(f"Loaded {len(entries):,} entries (excluding alternates)")

    # Split 95/5
    random.seed(42)
    random.shuffle(entries)
    split_idx = int(len(entries) * 0.95)
    train_entries = entries[:split_idx]
    test_entries = entries[split_idx:]

    print(f"Train: {len(train_entries):,}, Test: {len(test_entries):,}")

    # Train
    print("\nTraining (this takes a few minutes)...")
    dt = G2PDecisionTree(
        locale="en_US",
        phoneset_name="cmu",
        remove_stress=True,
        verbose=False,
        trainer="sklearn",
        parallel_align=False,  # Disable parallel to avoid multiprocessing issues
    )

    # Load training data
    train_lines = [f"{word}\t{' '.join(phones)}" for word, phones in train_entries]
    dt.load_prondict(iter(train_lines))
    dt.align()
    dt.train()

    print("Training complete. Evaluating...")

    # Evaluate
    correct_words = 0
    correct_phones = 0
    total_phones = 0

    for word, expected in test_entries:
        # Remove stress from expected
        expected_nostress = [p.rstrip("012") for p in expected]

        predicted = dt.pronounce(word)

        if predicted == expected_nostress:
            correct_words += 1

        # Phone-level accuracy (aligned comparison)
        min_len = min(len(predicted), len(expected_nostress))
        for i in range(min_len):
            if predicted[i] == expected_nostress[i]:
                correct_phones += 1
        total_phones += max(len(predicted), len(expected_nostress))

    word_acc = correct_words / len(test_entries) * 100
    phone_acc = correct_phones / total_phones * 100

    print(f"\nResults on {len(test_entries):,} test words:")
    print(f"  Word accuracy:   {word_acc:.1f}%")
    print(f"  Phone accuracy:  {phone_acc:.1f}%")


if __name__ == "__main__":
    main()
