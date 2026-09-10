"""CLI adapters for pronunciation scoring and suspicious-entry analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TextIO

from ...converter import G2P
from ...pronunciation_analysis import (
    CATEGORIES,
    Category,
    ScoreMethod,
    find_low_scores,
    find_score_gaps,
    find_zeros,
    score_entries,
    triage_entries,
)

_METHODS = ("geometric", "product", "arithmetic", "min", "harmonic")


def _jsonl(stream: TextIO) -> Iterator[dict[str, Any]]:
    for line in stream:
        if line.strip():
            yield json.loads(line)


def _score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input JSONL file (- for stdin)")
    parser.add_argument("-m", "--model", required=True, help="G2P model file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--method",
        choices=_METHODS,
        default="geometric",
        help="Score combining method (default: geometric)",
    )
    parser.set_defaults(func=handle_score_prons)


def setup_score_prons_command(subparsers) -> None:
    parser = subparsers.add_parser("score-prons", help="Score pronunciation candidates")
    _score_arguments(parser)


def handle_score_prons(args: argparse.Namespace) -> int:
    print(f"Loading model {args.model}...", file=sys.stderr)
    g2p = G2P(model=args.model, use_dict_fallback=False)
    if not g2p.has_distributions:
        print("WARNING: Model does not have distributions!", file=sys.stderr)
        print("         Scores will be less meaningful.", file=sys.stderr)
    source = (
        nullcontext(sys.stdin)
        if args.input == "-"
        else open(args.input, encoding="utf-8")  # noqa: SIM115
    )
    destination = (
        nullcontext(sys.stdout)
        if not args.output
        else open(args.output, "w", encoding="utf-8")  # noqa: SIM115
    )
    with source as infile, destination as outfile:
        count = 0
        method: ScoreMethod = args.method
        for entry in score_entries(g2p, _jsonl(infile), method):
            entry["prons"] = {
                pron: f"{score:.10f}".rstrip("0").rstrip(".")
                for pron, score in entry["prons"].items()
            }
            print(json.dumps(entry, ensure_ascii=False), file=outfile)
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count:,} entries...", file=sys.stderr)
    print(f"Done: {count:,} entries processed", file=sys.stderr)
    return 0


def _suspicious_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Scored JSONL file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--zeros", action="store_true", help="Find 0-score entries")
    mode.add_argument("--low", type=float, help="Find entries below threshold")
    mode.add_argument("--gaps", action="store_true", help="Find large score gaps")
    mode.add_argument("--triage", action="store_true", help="Triage into categories")
    parser.add_argument("-o", "--output", help="Output directory for triage files")
    parser.add_argument(
        "-n", type=int, default=50, help="Max stdout results (default: 50)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Triage threshold"
    )
    parser.set_defaults(func=handle_find_suspicious)


def setup_find_suspicious_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "find-suspicious", help="Find suspicious scored dictionary entries"
    )
    _suspicious_arguments(parser)


def _load(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        yield from _jsonl(stream)


def _print_prons(prons, marked=()) -> None:
    for pron, score in prons.items():
        marker = " ← 0!" if pron in marked else ""
        print(f"  {score!s:>12}  {pron}{marker}")
    print()


def handle_find_suspicious(args: argparse.Namespace) -> int:
    if args.zeros:
        print("=== Entries with 0-score pronunciations ===\n")
        for count, zero_result in enumerate(find_zeros(_load(args.input))):
            if count >= args.n:
                break
            print(f"{zero_result.word}:")
            _print_prons(zero_result.pronunciations, zero_result.zero_pronunciations)
    elif args.low is not None:
        print(f"=== Words with max score < {args.low} ===\n")
        low_results = sorted(
            find_low_scores(_load(args.input), args.low),
            key=lambda x: (x.score, x.word),
        )
        for low_result in low_results[: args.n]:
            print(f"{low_result.word} (max={low_result.score:.4f}):")
            _print_prons(low_result.pronunciations)
    elif args.gaps:
        print("=== Entries with large score gaps ===\n")
        gap_results = sorted(
            find_score_gaps(_load(args.input)),
            key=lambda x: (x.ratio, x.word),
            reverse=True,
        )
        for gap_result in gap_results[: args.n]:
            print(f"{gap_result.word} (ratio={gap_result.ratio:.0f}x):")
            _print_prons(gap_result.pronunciations)
    elif args.triage:
        _handle_triage(args)
    return 0


def _handle_triage(args: argparse.Namespace) -> None:
    print(f"Triaging entries with max score < {args.threshold}...", file=sys.stderr)
    results = triage_entries(_load(args.input), args.threshold)
    total = sum(len(values) for values in results.values())
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            if not results[category]:
                continue
            path = output / f"{category.lower()}.tsv"
            with path.open("w", encoding="utf-8") as stream:
                stream.write("word\tpron\tscore\treason\n")
                for result in results[category]:
                    stream.write(
                        f"{result.word}\t{result.pronunciation}\t{result.score:.6f}\t{result.reason}\n"
                    )
            print(
                f"  {category:12} {len(results[category]):5} → {path}", file=sys.stderr
            )
        print(f"\nWrote {total} entries to {output}/", file=sys.stderr)
        return
    print(f"\nFound {total} suspicious entries:\n")
    headings: Mapping[Category, str] = {
        "REVIEW": "REVIEW - Suspicious variants, need human check",
        "FOREIGN": "FOREIGN - Foreign origin, probably valid",
        "ABBREV": "ABBREV - Abbreviations/acronyms, valid",
        "FUNCTION": "FUNCTION - Common function words, valid",
    }
    for category, heading in headings.items():
        values = results[category]
        if values:
            print("=" * 60, f"{heading} ({len(values)} total)", "=" * 60, sep="\n")
            for result in values[: args.n]:
                print(f"{result.word} [{result.reason}] (score={result.score:.4f}):")
                print(f"  {result.score!s:>12}  {result.pronunciation}\n")
            if len(values) > args.n:
                print(f"  ... and {len(values) - args.n} more\n")
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for category in CATEGORIES:
        if results[category]:
            print(f"  {category:12} {len(results[category]):5}")


def score_prons_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score pronunciations using G2P model")
    _score_arguments(parser)
    return handle_score_prons(parser.parse_args(argv))


def find_suspicious_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find suspicious dictionary entries")
    _suspicious_arguments(parser)
    return handle_find_suspicious(parser.parse_args(argv))


__all__ = [
    "find_suspicious_main",
    "handle_find_suspicious",
    "handle_score_prons",
    "score_prons_main",
    "setup_find_suspicious_command",
    "setup_score_prons_command",
]
