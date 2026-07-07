#!/usr/bin/env python
"""Find suspicious dictionary entries using G2P scores.

Usage:
    python scripts/find_suspicious.py etc/cmudict-scored.jsonl --zeros
    python scripts/find_suspicious.py etc/cmudict-scored.jsonl --low 0.01
    python scripts/find_suspicious.py etc/cmudict-scored.jsonl --gaps
    python scripts/find_suspicious.py etc/cmudict-scored.jsonl --triage
"""

import argparse
import json
import re
import sys


def load_entries(path):
    """Load scored JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_zeros(entries):
    """Find entries with any 0-score pronunciations."""
    for entry in entries:
        word = entry["word"]
        zeros = [p for p, s in entry["prons"].items() if float(s) == 0]
        if zeros:
            yield word, entry["prons"], zeros


def find_low_scores(entries, threshold, min_len=5):
    """Find words where ALL pronunciations score below threshold."""
    for entry in entries:
        word = entry["word"]
        if len(word) < min_len:
            continue
        # Skip punctuation/numbers
        if not re.match(r"^[a-z]+$", word, re.I):
            continue
        scores = [float(v) for v in entry["prons"].values()]
        max_score = max(scores)
        if max_score < threshold:
            yield max_score, word, entry["prons"]


def find_score_gaps(entries, high_thresh=0.1, low_thresh=0.01):
    """Find entries with large score gaps between variants."""
    for entry in entries:
        word = entry["word"]
        prons = entry["prons"]
        if len(prons) < 2:
            continue
        scores = [float(s) for s in prons.values()]
        max_s, min_s = max(scores), min(scores)
        if max_s > high_thresh and min_s < low_thresh:
            yield max_s / min_s if min_s > 0 else float("inf"), word, prons


def classify_variant(word, pron, score, best_score):
    """Classify a single pronunciation variant."""
    phones = pron.split()
    letters_only = re.sub(r"[^a-z]", "", word.lower())

    # Zero score = impossible mapping
    if score == 0:
        return "ABBREV", "spelled out (0 score)"

    # Short word with way more phones = spelled out
    if len(letters_only) <= 4 and len(phones) > len(letters_only) * 1.5:
        return "ABBREV", "spelled out"

    # Title abbreviations
    if word.lower() in {
        "mr",
        "mr.",
        "dr",
        "dr.",
        "jr",
        "jr.",
        "sr",
        "sr.",
        "st",
        "st.",
    }:
        return "ABBREV", "title abbreviation"

    # Function words
    if word.lower() in {"of", "the", "to", "is", "was", "were", "have", "has", "had"}:
        return "FUNCTION", "function word"

    # Chinese romanization
    if re.search(r"^xia|^zh[aeiou]|iao$|^qia|^qiu", word, re.I):
        return "FOREIGN", "Chinese"

    # Irish/Scottish
    if re.search(r"^mc[a-z]|^mac[a-z]", word, re.I):
        return "FOREIGN", "Irish/Scottish"

    # Default: needs review (including short words)
    if best_score > 0 and best_score > score:
        return "REVIEW", f"score={score:.3f} (best={best_score:.3f})"
    return "REVIEW", f"score={score:.3f}"


def triage_entries(entries, threshold=0.05):
    """Triage all pronunciation variants into categories."""
    results: dict[str, list] = {
        "REVIEW": [],
        "FOREIGN": [],
        "ABBREV": [],
        "FUNCTION": [],
        "OK": [],
    }

    for entry in entries:
        word = entry["word"]
        prons = entry["prons"]

        # Skip very short words
        letters_only = re.sub(r"[^a-z]", "", word.lower())
        if len(letters_only) < 2:
            continue

        scores = [(p, float(s)) for p, s in prons.items()]
        best_score = max(s for _, s in scores)

        # Classify each variant individually
        for pron, score in scores:
            if score >= threshold:
                results["OK"].append((score, word, {pron: str(score)}, "ok"))
            else:
                category, reason = classify_variant(word, pron, score, best_score)
                results[category].append((score, word, {pron: str(score)}, reason))

    # Sort each category by score (key=first element to avoid dict comparison)
    for cat in results:
        if cat == "OK":
            results[cat].sort(
                key=lambda x: (-x[0], x[1])
            )  # descending score, then word
        else:
            results[cat].sort(key=lambda x: (x[0], x[1]))  # ascending score, then word

    return results


def main():
    parser = argparse.ArgumentParser(description="Find suspicious dictionary entries")
    parser.add_argument("input", help="Scored JSONL file")
    parser.add_argument("--zeros", action="store_true", help="Find 0-score entries")
    parser.add_argument("--low", type=float, help="Find entries below threshold")
    parser.add_argument("--gaps", action="store_true", help="Find large score gaps")
    parser.add_argument("--triage", action="store_true", help="Triage into categories")
    parser.add_argument("-o", "--output", help="Output directory for triage files")
    parser.add_argument(
        "-n",
        type=int,
        default=50,
        help="Max results per category for stdout (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Score threshold for triage (default: 0.05)",
    )
    args = parser.parse_args()

    if args.zeros:
        print("=== Entries with 0-score pronunciations ===\n")
        for count, (word, prons, zeros) in enumerate(
            find_zeros(load_entries(args.input))
        ):
            if count >= args.n:
                break
            print(f"{word}:")
            for p, s in prons.items():
                marker = " ← 0!" if p in zeros else ""
                print(f"  {s:>12}  {p}{marker}")
            print()

    elif args.low:
        print(f"=== Words with max score < {args.low} ===\n")
        results = list(find_low_scores(load_entries(args.input), args.low))
        results.sort()
        for score, word, prons in results[: args.n]:
            print(f"{word} (max={score:.4f}):")
            for p, s in prons.items():
                print(f"  {s:>12}  {p}")
            print()

    elif args.gaps:
        print("=== Entries with large score gaps ===\n")
        results = list(find_score_gaps(load_entries(args.input)))
        results.sort(reverse=True)
        for ratio, word, prons in results[: args.n]:
            print(f"{word} (ratio={ratio:.0f}x):")
            for p, s in prons.items():
                print(f"  {s:>12}  {p}")
            print()

    elif args.triage:
        # Distinct local name from the `results: list[...]` flavour used
        # in the args.low / args.gaps branches above — keeps the
        # dict-typed flow self-contained so mypy can infer correctly.
        print(f"Triaging entries with max score < {args.threshold}...", file=sys.stderr)
        triage_results = triage_entries(load_entries(args.input), args.threshold)

        category_info = {
            "REVIEW": ("REVIEW - Suspicious variants, need human check", "review.txt"),
            "FOREIGN": ("FOREIGN - Foreign origin, probably valid", "foreign.txt"),
            "ABBREV": ("ABBREV - Abbreviations/acronyms, valid", "abbrev.txt"),
            "FUNCTION": ("FUNCTION - Common function words, valid", "function.txt"),
            "OK": ("OK - Normal variants", "ok.txt"),
        }

        total = sum(len(v) for v in triage_results.values())

        # Output to directory if specified
        if args.output:
            import os

            os.makedirs(args.output, exist_ok=True)

            for cat in ["REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK"]:
                entries_list = triage_results[cat]
                if not entries_list:
                    continue

                header, filename = category_info[cat]
                # Use .tsv extension
                filepath = os.path.join(args.output, filename.replace(".txt", ".tsv"))

                with open(filepath, "w") as f:
                    # TSV header
                    f.write("word\tpron\tscore\treason\n")

                    for score, word, prons, reason in entries_list:
                        # The pron-frequency value (`_freq`) is unused
                        # here — we only emit the model's overall score.
                        for p, _freq in prons.items():
                            f.write(f"{word}\t{p}\t{score:.6f}\t{reason}\n")

                print(f"  {cat:12} {len(entries_list):5} → {filepath}", file=sys.stderr)

            print(f"\nWrote {total} entries to {args.output}/", file=sys.stderr)

        else:
            # Print to stdout
            print(f"\nFound {total} suspicious entries:\n")

            for cat in ["REVIEW", "FOREIGN", "ABBREV", "FUNCTION"]:
                entries_list = triage_results[cat]
                if not entries_list:
                    continue

                header, _ = category_info[cat]
                print(f"{'=' * 60}")
                print(f"{header} ({len(entries_list)} total)")
                print(f"{'=' * 60}\n")

                for score, word, prons, reason in entries_list[: args.n]:
                    print(f"{word} [{reason}] (score={score:.4f}):")
                    for p, s in prons.items():
                        print(f"  {s:>12}  {p}")
                    print()

                if len(entries_list) > args.n:
                    print(f"  ... and {len(entries_list) - args.n} more\n")

            print(f"\n{'=' * 60}")
            print("SUMMARY")
            print(f"{'=' * 60}")
            for cat in ["REVIEW", "FOREIGN", "ABBREV", "FUNCTION", "OK"]:
                count = len(triage_results[cat])
                if count > 0:
                    print(f"  {cat:12} {count:5}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
