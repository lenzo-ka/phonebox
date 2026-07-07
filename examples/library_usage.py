#!/usr/bin/env python
"""
Examples of using phonebox as a Python library.
"""

from phonebox import G2P, Dictionary


def example_simple_usage():
    """Simplest way to use phonebox."""
    print("=== Simple Usage ===")

    # Load pre-trained model
    g2p = G2P(model="models/en_US_nostress.g2p.gz")

    # Pronounce a word
    pronunciation = g2p.pronounce("hello")
    print(f"hello: {pronunciation}")

    # Or use as callable
    print(f"world: {g2p('world')}")

    # Batch pronunciation
    words = ["python", "grapheme", "phoneme"]
    for word, phones in g2p.pronounce_batch(words):
        print(f"{word}: {' '.join(phones)}")


def example_pocketsphinx_setup():
    """One-command PocketSphinx setup."""
    print("\n=== PocketSphinx Setup ===")

    # This fetches CMUdict, processes it, and trains a model
    g2p = G2P.from_pocketsphinx()

    # Use it
    print(f"test: {g2p('test')}")


def example_dictionary_class():
    """Using the Dictionary class."""
    print("\n=== Dictionary Class ===")

    # Fetch a dictionary
    dict = Dictionary.fetch("cmudict")
    print(f"Dictionary: {dict}")
    print(f"Size: {len(dict)} entries")

    # Process it
    processed = dict.process(remove_stress=True, output="processed.dict")
    print(f"Processed: {processed}")

    # Train a model
    _ = processed.train_g2p_model(locale="en_US", output="my_model.g2p.gz")
    print("Model trained")


def example_train_from_custom_dict():
    """Train from your own dictionary."""
    print("\n=== Train from Custom Dictionary ===")

    # Create a small test dictionary
    test_dict = """
hello HH AH L OW
world W ER L D
python P AY TH AA N
    """.strip()

    with open("test_dict.txt", "w") as f:
        f.write(test_dict)

    # Train directly
    g2p = G2P.train(dictionary="test_dict.txt", locale="en_US", remove_stress=False)

    # Use it
    print(f"hello: {g2p('hello')}")

    # Clean up
    import os

    os.remove("test_dict.txt")


def example_method_chaining():
    """Fluent API with method chaining."""
    print("\n=== Method Chaining ===")

    # Fetch, process, and train in one line
    _ = (
        Dictionary.fetch("cmudict")
        .process(remove_stress=True)
        .train_g2p_model("en_US", output="model.g2p.gz")
    )

    print("Model trained via chaining")


def example_advanced_usage():
    """Advanced features."""
    print("\n=== Advanced Usage ===")

    # Direct DecisionTree access for full control
    from phonebox import DecisionTree

    dt = DecisionTree(
        locale="en_US",
        phoneset_name="cmu",
        remove_stress=True,
        max_iterations=50,
        min_samples_split=5,
    )

    # Load and train
    with open("data/cmudict/cmudict_nostress.dict") as f:
        dt.load_prondict(f)
    dt.align()
    dt.train()

    # Use it
    print(f"pronunciation: {dt.pronounce('pronunciation')}")


if __name__ == "__main__":
    # Run only the simple example by default
    # Uncomment others as needed

    example_simple_usage()

    # example_pocketsphinx_setup()  # Requires internet
    # example_dictionary_class()     # Requires internet
    # example_train_from_custom_dict()
    # example_method_chaining()      # Requires internet
    # example_advanced_usage()       # Requires data/
