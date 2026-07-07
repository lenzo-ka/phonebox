#!/usr/bin/env python
"""
Example: Score pronunciation likelihood given a word.

Demonstrates how to compute the probability that a model would generate
a specific pronunciation for a given word.
"""

from phonebox.converter import G2P


def main():
    # Load model with distributions
    g2p = G2P(model="models/en_US_nostress.g2p.gz")

    print("Pronunciation Likelihood Scoring")
    print("=" * 60)
    print()

    # Example 1: Score different pronunciations of "READ"
    word = "READ"
    pronunciations = [
        ["R", "IY", "D"],  # present tense (common)
        ["R", "EH", "D"],  # past tense (also common)
        ["R", "AH", "D"],  # unlikely pronunciation
    ]

    print(f"Word: {word}")
    print("-" * 60)

    for phones in pronunciations:
        score = g2p._dt.score_pronunciation(word, phones)
        phones_str = " ".join(phones)
        print(f"  {phones_str:20s} → score: {score:.6f}")
    print()

    # Example 2: Compare model's prediction with given pronunciation
    word = "HELLO"
    print(f"Word: {word}")
    print("-" * 60)

    # Get model's prediction
    predicted = g2p.pronounce(word)
    predicted_str = " ".join(predicted)
    print(f"  Model predicts: {predicted_str}")

    # Score the prediction
    score = g2p._dt.score_pronunciation(word, predicted)
    print(f"  Self-score:     {score:.6f}")
    print()

    # Score an alternative pronunciation
    alternative = ["HH", "EH", "L", "OW"]
    alt_str = " ".join(alternative)
    alt_score = g2p._dt.score_pronunciation(word, alternative)
    print(f"  Alternative:    {alt_str}")
    print(f"  Alt score:      {alt_score:.6f}")
    print()

    # Example 3: Multiple words comparison
    print("Comparing pronunciations across words:")
    print("-" * 60)

    test_cases = [
        ("CAT", ["K", "AE", "T"]),
        ("DOG", ["D", "AO", "G"]),
        ("BIRD", ["B", "ER", "D"]),
        ("XYLOPHONE", ["Z", "AY", "L", "AH", "F", "OW", "N"]),
    ]

    for word, phones in test_cases:
        score = g2p._dt.score_pronunciation(word, phones)
        phones_str = " ".join(phones)
        print(f"  {word:12s} → {phones_str:30s} = {score:.6f}")

    print()
    print("Note: Scores close to 1.0 indicate high confidence.")
    print("      Scores near 0.0 indicate unlikely pronunciations.")


if __name__ == "__main__":
    main()
