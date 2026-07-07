#!/usr/bin/env python
"""
Score all pronunciations in a dictionary using a trained model.

Shows which dictionary entries the model has low confidence in,
which can indicate:
- Irregular pronunciations
- Rare words
- Potential errors in the dictionary
- Words that need to be in the exceptions list
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from phonebox.constants import DICT_ENCODING
from phonebox.converter import G2P
from phonebox.utils.io import is_dict_comment

PROGRESS_INTERVAL = 10000

DEFAULT_BOTTOM_COUNT = 100

# Probability thresholds for score-distribution histogram buckets.
# score_pronunciation returns 0.0-1.0; higher is better.
BUCKET_VERY_HIGH_MIN = 0.99
BUCKET_HIGH_MIN = 0.90
BUCKET_MEDIUM_MIN = 0.50
BUCKET_LOW_MIN = 0.10


def strip_variant_number(word: str) -> tuple[str, int]:
    """
    Strip variant markers like (1), (2) from words and return the number.

    Examples:
        HELLO(1) → (HELLO, 1)
        READ(2) → (READ, 2)
        HELLO → (HELLO, 1)
    """
    match = re.search(r"\((\d+)\)$", word)
    if not match:
        return word, 1
    return word[: match.start()], int(match.group(1))


def score_dictionary(
    model_path: str, dict_path: str, limit: int | None = None
) -> list[tuple[float, str, list[str], list[str], int]]:
    """
    Score all pronunciations in a dictionary.

    Returns a list of (score, word, phones, predicted, orig_rank) tuples,
    one per dictionary entry (with variants preserved).
    """
    print(f"Loading model: {model_path}")
    g2p = G2P(model=model_path)

    if not g2p.has_distributions:
        print("WARNING: Model does not have distributions!")
        print("         Scores will all be 1.0")
        print("         Retrain with --store-distributions")
        return []

    print(f"Loading dictionary: {dict_path}")
    results: list[tuple[float, str, list[str], list[str], int]] = []

    with open(dict_path, encoding=DICT_ENCODING) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if is_dict_comment(line):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            word, phones = parts[0], parts[1:]

            base_word, orig_rank = strip_variant_number(word)
            score = g2p.score_pronunciation(base_word, phones)
            predicted = g2p.pronounce(base_word)
            results.append((score, word, phones, predicted, orig_rank))

            if (i + 1) % PROGRESS_INTERVAL == 0:
                print(f"  Processed {i + 1:,} entries...")

    print(f"Scored {len(results):,} pronunciations")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score all pronunciations in a dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score CMUdict with default model
  python score_dictionary.py

  # Show only bottom 50 (lowest confidence)
  python score_dictionary.py --bottom 50

  # Reorder pronunciations by likelihood (best first, no instance number)
  python score_dictionary.py --reorder -o reordered_dict.tsv

  # Show only words where pronunciation ranking changed
  python score_dictionary.py --rank-changes-only -o rank_changes.tsv

  # Score custom dictionary
  python score_dictionary.py --dict my_dict.txt --model my_model.g2p.gz
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        default="models/en_US_nostress.g2p.gz",
        help="Path to g2p model (default: models/en_US_nostress.g2p.gz)",
    )
    parser.add_argument(
        "--dict",
        "-d",
        default="data/cmudict/cmudict_nostress.dict",
        help="Path to dictionary (default: data/cmudict/cmudict_nostress.dict)",
    )
    parser.add_argument(
        "--bottom",
        "-b",
        type=int,
        default=DEFAULT_BOTTOM_COUNT,
        help=f"Show bottom N entries (default: {DEFAULT_BOTTOM_COUNT})",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        help="Show top N entries (highest confidence)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Show only entries below this score",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of dictionary entries to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write results to file",
    )
    parser.add_argument(
        "--reorder",
        "-r",
        action="store_true",
        help="Reorder pronunciations by score (highest first, no instance number for top)",
    )
    parser.add_argument(
        "--rank-changes-only",
        action="store_true",
        help="Only output words where ranks changed (implies --reorder)",
    )

    args = parser.parse_args()
    reorder = args.reorder or args.rank_changes_only

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print(f"Run: phonebox recipe cmudict pocketsphinx -o {args.model}")
        return 1

    if not Path(args.dict).exists():
        print(f"Error: Dictionary not found: {args.dict}")
        return 1

    results = score_dictionary(args.model, args.dict, args.limit)
    if not results:
        return 1

    # Sort by score ascending: lowest probability (worst pronunciations) first
    results.sort(key=lambda x: x[0])

    if args.threshold:
        results = [
            (s, w, p, pred, orank)
            for s, w, p, pred, orank in results
            if s < args.threshold
        ]

    reordered: list[tuple[float, str, list[str], list[str], int, int]] = []
    if reorder:
        word_groups: dict[str, list[tuple[float, str, list[str], list[str], int]]] = (
            defaultdict(list)
        )
        for score, word, phones, predicted, orig_rank in results:
            base_word, _ = strip_variant_number(word)
            word_groups[base_word].append((score, word, phones, predicted, orig_rank))

        for base_word, variants in word_groups.items():
            variants.sort(key=lambda x: x[0], reverse=True)
            for i, (score, _, phones, predicted, orig_rank) in enumerate(variants):
                new_rank = i + 1
                new_word = base_word if i == 0 else f"{base_word}({i + 1})"
                reordered.append(
                    (score, new_word, phones, predicted, orig_rank, new_rank)
                )

        if args.rank_changes_only:
            grouped: dict[
                str, list[tuple[float, str, list[str], list[str], int, int]]
            ] = defaultdict(list)
            for entry in reordered:
                base_word, _ = strip_variant_number(entry[1])
                grouped[base_word].append(entry)

            reordered = [
                entry
                for variants_6 in grouped.values()
                if any(
                    orig_rank != new_rank
                    for _, _, _, _, orig_rank, new_rank in variants_6
                )
                for entry in variants_6
            ]

        reordered.sort(key=lambda x: x[1])

    if reorder:
        scores = [score for score, _, _, _, _, _ in reordered]
    else:
        scores = [score for score, _, _, _, _ in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"Total entries:     {len(scores):,}", file=sys.stderr)
    print(f"Average score:     {avg_score:.6f}", file=sys.stderr)
    print(f"Lowest score:      {min(scores):.6f}", file=sys.stderr)
    print(f"Highest score:     {max(scores):.6f}", file=sys.stderr)
    print("", file=sys.stderr)

    very_high = sum(1 for s in scores if s >= BUCKET_VERY_HIGH_MIN)
    high = sum(1 for s in scores if BUCKET_HIGH_MIN <= s < BUCKET_VERY_HIGH_MIN)
    medium = sum(1 for s in scores if BUCKET_MEDIUM_MIN <= s < BUCKET_HIGH_MIN)
    low = sum(1 for s in scores if BUCKET_LOW_MIN <= s < BUCKET_MEDIUM_MIN)
    very_low = sum(1 for s in scores if s < BUCKET_LOW_MIN)
    total = len(scores)

    print("Score distribution (probability):", file=sys.stderr)
    print(
        f"  Very High (>= {BUCKET_VERY_HIGH_MIN:.2f}):  "
        f"{very_high:6,} ({100 * very_high / total:5.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  High      (>= {BUCKET_HIGH_MIN:.2f}):  "
        f"{high:6,} ({100 * high / total:5.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  Medium    (>= {BUCKET_MEDIUM_MIN:.2f}):  "
        f"{medium:6,} ({100 * medium / total:5.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  Low       (>= {BUCKET_LOW_MIN:.2f}):  {low:6,} ({100 * low / total:5.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  Very Low  (<  {BUCKET_LOW_MIN:.2f}):  "
        f"{very_low:6,} ({100 * very_low / total:5.1f}%)",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    if reorder:
        if args.bottom:
            display_6 = reordered[: args.bottom]
        elif args.top:
            display_6 = reordered[-args.top :]
        else:
            display_6 = []

        header = "word\tdict_pronunciation\tscore\tmodel_prediction\tmatch\torig_rank\tnew_rank"
        print(header)
        for score, word, phones, predicted, orig_rank, new_rank in display_6:
            match = "=" if phones == predicted else "!"
            print(
                f"{word}\t{' '.join(phones)}\t{score:.6f}\t{' '.join(predicted)}\t"
                f"{match}\t{orig_rank}\t{new_rank}"
            )

        if args.output:
            with open(args.output, "w") as f:
                f.write(header + "\n")
                for score, word, phones, predicted, orig_rank, new_rank in reordered:
                    match = "=" if phones == predicted else "!"
                    f.write(
                        f"{word}\t{' '.join(phones)}\t{score:.6f}\t{' '.join(predicted)}\t"
                        f"{match}\t{orig_rank}\t{new_rank}\n"
                    )
            print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        if args.bottom:
            display_5 = results[: args.bottom]
        elif args.top:
            display_5 = results[-args.top :]
        else:
            display_5 = []

        header = "word\tdict_pronunciation\tscore\tmodel_prediction\tmatch\torig_rank"
        print(header)
        for score, word, phones, predicted, orig_rank in display_5:
            match = "=" if phones == predicted else "!"
            print(
                f"{word}\t{' '.join(phones)}\t{score:.6f}\t{' '.join(predicted)}\t"
                f"{match}\t{orig_rank}"
            )

        if args.output:
            with open(args.output, "w") as f:
                f.write(header + "\n")
                for score, word, phones, predicted, orig_rank in results:
                    match = "=" if phones == predicted else "!"
                    f.write(
                        f"{word}\t{' '.join(phones)}\t{score:.6f}\t{' '.join(predicted)}\t"
                        f"{match}\t{orig_rank}\n"
                    )
            print(f"Results written to: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    exit(main())
