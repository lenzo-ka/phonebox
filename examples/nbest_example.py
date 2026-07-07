#!/usr/bin/env python
"""
Example demonstrating N-best and confidence scoring features.

This script shows how to:
1. Train a model with distributions enabled
2. Get confidence scores for pronunciations
3. Generate n-best pronunciation alternatives
4. Use confidence scores for quality control
"""

import sys
from pathlib import Path

from phonebox import G2P
from phonebox.core.decision_tree import DecisionTree


def train_model_with_distributions():
    """Train a simple model with distributions enabled."""
    print("=" * 60)
    print("Training model with distributions enabled...")
    print("=" * 60)

    # Create simple dictionary with some ambiguous words
    sample_dict = """
hello	HH EH L OW
world	W ER L D
read	R IY D
read	R EH D
lead	L IY D
lead	L EH D
test	T EH S T
python	P AY TH AA N
either	IY DH ER
either	AY DH ER
    """.strip()

    # Write to temp file
    dict_path = Path("temp_dict.txt")
    with open(dict_path, "w") as f:
        f.write(sample_dict)

    # Train with distributions
    dt = DecisionTree(
        locale="en_US",
        phoneset_name="cmu",
        store_distributions=True,
        min_dist_entropy=0.01,  # Low threshold to capture ambiguity
        verbose=True,
    )

    with open(dict_path) as f:
        dt.load_prondict(f)

    dt.align()
    dt.train()

    # Export model
    model_path = Path("temp_model.g2p.gz")
    dt.export(str(model_path))

    print(f"\nDone: Model saved to: {model_path}")
    print("  Model version: 2 (supports n-best)")

    # Cleanup
    dict_path.unlink()

    return model_path


def demo_confidence_scoring(g2p):
    """Demonstrate confidence scoring."""
    print("\n" + "=" * 60)
    print("DEMO 1: Confidence Scoring")
    print("=" * 60)

    test_words = ["hello", "world", "read", "test"]

    print("\nPer-phoneme confidence scores:\n")

    for word in test_words:
        phones, confidences = g2p.pronounce_with_confidence(word)

        print(f"{word:10}", end="  ")
        for phone, conf in zip(phones, confidences):
            # Color code by confidence
            if conf >= 0.9:
                marker = "+"  # High confidence
            elif conf >= 0.7:
                marker = "~"  # Medium confidence
            else:
                marker = "?"  # Low confidence

            print(f"{phone}({conf:.2f}){marker}", end=" ")
        print()

    print("\n  + = High confidence (>=0.9)")
    print("  ~ = Medium confidence (≥0.7)")
    print("  ? = Low confidence (<0.7)")


def demo_nbest_generation(g2p):
    """Demonstrate n-best pronunciation alternatives."""
    print("\n" + "=" * 60)
    print("DEMO 2: N-Best Pronunciation Alternatives")
    print("=" * 60)

    # Words with multiple pronunciations
    ambiguous_words = ["read", "lead", "either"]

    for word in ambiguous_words:
        print(f"\n'{word}':")
        nbest = g2p.pronounce_nbest(word, n=5)

        for i, (phones, score) in enumerate(nbest, 1):
            pron = " ".join(phones)
            bar = "█" * int(score * 30)  # Visual bar
            print(f"  {i}. {pron:20} {score:5.1%} {bar}")


def demo_quality_control(g2p):
    """Demonstrate using confidence for quality control."""
    print("\n" + "=" * 60)
    print("DEMO 3: Quality Control with Confidence Scores")
    print("=" * 60)

    test_words = ["hello", "world", "read", "python", "test"]

    print("\nQuality assessment:\n")

    low_confidence_words = []

    for word in test_words:
        phones, confidences = g2p.pronounce_with_confidence(word)

        if not confidences:
            continue

        min_conf = min(confidences)
        avg_conf = sum(confidences) / len(confidences)

        # Classify quality
        if min_conf >= 0.8:
            quality = "HIGH  "
            emoji = "[match]"
        elif min_conf >= 0.6:
            quality = "MEDIUM"
            emoji = "[warn]"
        else:
            quality = "LOW   "
            emoji = "[diff]"
            low_confidence_words.append((word, min_conf))

        print(f"{emoji} {quality}  {word:10}  min={min_conf:.2f}  avg={avg_conf:.2f}")

    if low_confidence_words:
        print("\nWARNING: Words flagged for manual review:")
        for word, conf in low_confidence_words:
            print(f"   - {word} (confidence: {conf:.2f})")


def demo_contextual_pronunciation(g2p):
    """Demonstrate context-aware pronunciation selection."""
    print("\n" + "=" * 60)
    print("DEMO 4: Context-Aware Pronunciation")
    print("=" * 60)

    sentences = [
        "I read the book yesterday",  # Past tense
        "I read books every day",  # Present tense
        "You can either come or stay",  # First pronunciation
        "I don't like it either",  # Second pronunciation
    ]

    print("\nSelecting pronunciations based on context:\n")

    for sentence in sentences:
        print(f'\nSentence: "{sentence}"')

        # Find ambiguous words
        words = sentence.lower().split()

        for word in words:
            if word in ["read", "either", "lead"]:
                nbest = g2p.pronounce_nbest(word, n=3)

                if len(nbest) > 1:
                    print(f"  '{word}' alternatives:")
                    for phones, score in nbest:
                        pron = " ".join(phones)
                        print(f"    - {pron:20} ({score:.1%})")


def demo_batch_processing(g2p):
    """Demonstrate batch processing with statistics."""
    print("\n" + "=" * 60)
    print("DEMO 5: Batch Processing with Statistics")
    print("=" * 60)

    words = ["hello", "world", "read", "python", "test", "either", "lead"]

    print(f"\nProcessing {len(words)} words...\n")

    total_phones = 0
    total_conf = 0.0
    conf_buckets = {"high": 0, "medium": 0, "low": 0}

    for word in words:
        phones, confidences = g2p.pronounce_with_confidence(word)

        if confidences:
            total_phones += len(confidences)
            total_conf += sum(confidences)

            min_conf = min(confidences)
            if min_conf >= 0.8:
                conf_buckets["high"] += 1
            elif min_conf >= 0.6:
                conf_buckets["medium"] += 1
            else:
                conf_buckets["low"] += 1

    avg_conf = total_conf / total_phones if total_phones > 0 else 0

    print("Statistics:")
    print(f"  Total phonemes: {total_phones}")
    print(f"  Average confidence: {avg_conf:.2%}")
    print("\nQuality distribution:")
    print(
        f"  High confidence:   {conf_buckets['high']:2} words ({conf_buckets['high'] / len(words):.0%})"
    )
    print(
        f"  Medium confidence: {conf_buckets['medium']:2} words ({conf_buckets['medium'] / len(words):.0%})"
    )
    print(
        f"  Low confidence:    {conf_buckets['low']:2} words ({conf_buckets['low'] / len(words):.0%})"
    )


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print(" G2P N-Best and Confidence Scoring Demo")
    print("=" * 60)

    # Train model
    model_path = train_model_with_distributions()

    # Load model
    print("\nLoading model...")
    g2p = G2P(model=str(model_path))

    # Run demonstrations
    demo_confidence_scoring(g2p)
    demo_nbest_generation(g2p)
    demo_quality_control(g2p)
    demo_contextual_pronunciation(g2p)
    demo_batch_processing(g2p)

    # Cleanup
    model_path.unlink()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Train with --store-distributions to enable n-best")
    print("  2. Use confidence scores to flag uncertain predictions")
    print("  3. N-best provides alternatives for ambiguous words")
    print("  4. Oracle accuracy (any candidate correct) upper-bounds 1-best")
    print("\nFor more examples, see docs/NBEST_USAGE.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
