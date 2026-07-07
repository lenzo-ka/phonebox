#!/usr/bin/env python
"""
Score pronunciations in cmudict-freq.jsonl using G2P model.

Input:  {"word": "the", "freq": 123, "prons": ["DH AH0", "DH AH1"]}
Output: {"word": "the", "freq": 123, "prons": {"DH AH0": 0.85, "DH AH1": 0.15}}

Scores are from the G2P model's probability distributions.
"""

import argparse
import json
import re
import sys
from typing import TextIO

from phonebox.converter import G2P


def strip_stress(pron: str) -> list[str]:
    """Remove stress markers (0, 1, 2) from phonemes, return as list."""
    return re.sub(r"[012]", "", pron).split()


def score_pronunciations(
    g2p: G2P, word: str, prons: list[str], method: str = "geometric"
) -> dict[str, float]:
    """
    Score each pronunciation using the G2P model's distributions.

    Strips stress for scoring but keeps original prons in output.
    Returns dict of pron -> score, sorted by score descending.
    """
    scored = {}
    for pron in prons:
        # Strip stress for scoring (model doesn't predict stress)
        phones = strip_stress(pron)
        # Score this pronunciation
        score = g2p.score_pronunciation(word, phones, method=method)
        scored[pron] = score

    # Sort by score descending
    return dict(sorted(scored.items(), key=lambda x: -x[1]))


def main():
    parser = argparse.ArgumentParser(description="Score pronunciations using G2P model")
    parser.add_argument("input", help="Input JSONL file (- for stdin)")
    parser.add_argument("-m", "--model", required=True, help="G2P model file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--method",
        default="geometric",
        choices=["geometric", "product", "arithmetic", "min", "harmonic"],
        help="Score combining method (default: geometric)",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model {args.model}...", file=sys.stderr)
    g2p = G2P(model=args.model, use_dict_fallback=False)

    if not g2p.has_distributions:
        print("WARNING: Model does not have distributions!", file=sys.stderr)
        print("         Scores will be less meaningful.", file=sys.stderr)

    # Open input. Use explicit TextIO union annotation so mypy can
    # accept both branches (sys.stdin is a TextIO-ish; opened file is
    # a TextIOWrapper). ruff would prefer a `with` context manager, but
    # we need infile/outfile to be live across the try/finally below
    # and shared between the stdin/stdout fallback paths — context
    # managers don't compose cleanly with the optional-args pattern.
    infile: TextIO = sys.stdin if args.input == "-" else open(args.input)  # noqa: SIM115
    outfile: TextIO = (
        open(args.output, "w") if args.output else sys.stdout  # noqa: SIM115
    )

    try:
        count = 0
        for line in infile:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            word = entry["word"]
            prons = entry["prons"]

            # Score pronunciations
            scored_prons = score_pronunciations(g2p, word, prons, method=args.method)

            # Update entry
            entry["prons"] = scored_prons

            # Output (format floats as decimals, not scientific notation)
            def format_value(v):
                if isinstance(v, float):
                    return f"{v:.10f}".rstrip("0").rstrip(".")
                return v

            formatted_prons = {k: format_value(v) for k, v in entry["prons"].items()}
            entry["prons"] = formatted_prons
            print(json.dumps(entry, ensure_ascii=False), file=outfile)

            count += 1
            if count % 10000 == 0:
                print(f"Processed {count:,} entries...", file=sys.stderr)

        print(f"Done: {count:,} entries processed", file=sys.stderr)

    finally:
        if args.input != "-":
            infile.close()
        if args.output:
            outfile.close()


if __name__ == "__main__":
    main()
