#!/usr/bin/env python
"""
Basic usage example for the G2P package.
"""

from phonebox import DecisionTree


def main():
    # Example 1: Using a pre-trained model
    print("Example 1: Using a pre-trained model")
    print("-" * 40)

    # Note: You'll need to have a trained model file
    # You can build one using: python build.py en_US
    try:
        # Load a model
        g2p = DecisionTree(model="build/en_US/g2p.json.gz")

        # Test some words
        test_words = ["hello", "world", "python", "grapheme", "phoneme"]

        for word in test_words:
            pronunciation = g2p.pronounce(word)
            print(f"{word:15} -> {' '.join(pronunciation)}")

    except FileNotFoundError:
        print("No pre-trained model found. Run 'python build.py en_US' first.")

    print()

    # Example 2: Training a simple model
    print("Example 2: Training a model from scratch")
    print("-" * 40)

    # Create a small dictionary for demonstration
    mini_dict = """
cat\tK AE T
dog\tD AO G
bird\tB ER D
fish\tF IH SH
hello\tHH AH L OW
world\tW ER L D
    """.strip()

    # Save mini dictionary
    with open("mini_dict.txt", "w") as f:
        f.write(mini_dict)

    # Train a model
    dt = DecisionTree(
        locale="en_US", phoneset_name="cmu", remove_stress=True, verbose=False
    )

    print("Loading dictionary...")
    with open("mini_dict.txt") as f:
        dt.load_prondict(f)

    print("Aligning...")
    dt.align()

    print("Training...")
    dt.train()

    # Test the model
    print("\nTesting trained model:")
    test_words = ["cat", "dog", "hat", "fog"]
    for word in test_words:
        pronunciation = dt.pronounce(word)
        print(f"{word:10} -> {' '.join(pronunciation)}")

    # Clean up
    import os

    os.remove("mini_dict.txt")

    print("\nDone!")


if __name__ == "__main__":
    main()
