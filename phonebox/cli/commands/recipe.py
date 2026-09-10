"""
Recipe command for common G2P workflows.

Usage:
    phonebox recipe cmudict tts -o g2p.py
    phonebox recipe cmudict pocketsphinx -o g2p.py
    phonebox recipe my_dict.txt tts -o g2p.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from ...constants import DEFAULT_TRAIN_PRUNE, DEFAULT_TRAIN_VALIDATION_SPLIT


def setup_recipe_commands(subparsers):
    """Setup recipe command."""
    parser = subparsers.add_parser(
        "recipe",
        help="Build G2P from dictionary (fetch, train, bundle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""One-command G2P recipe: fetch dictionary, train model, bundle executable.

Presets:
  pocketsphinx  No stress markers (for speech recognition)
  tts           Primary stress only (for text-to-speech)""",
        epilog="""
Examples:
  # Build PocketSphinx G2P from CMUdict
  phonebox recipe cmudict pocketsphinx -o g2p.py

  # Build TTS G2P from CMUdict
  phonebox recipe cmudict tts -o g2p.py

  # TTS with secondary stress
  phonebox recipe cmudict tts -o g2p.py --keep-secondary

  # Use local dictionary
  phonebox recipe my_dict.txt tts -o g2p.py
""",
    )

    parser.add_argument(
        "source",
        help="Dictionary source: 'cmudict' or path to local file",
    )
    parser.add_argument(
        "preset",
        choices=["pocketsphinx", "tts"],
        help="pocketsphinx (no stress) or tts (primary stress)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file (.py for standalone runner, .g2p.gz for model only)",
    )
    parser.add_argument(
        "--keep-secondary",
        action="store_true",
        help="TTS: keep secondary stress (2) as well as primary (1)",
    )
    parser.add_argument(
        "--mark-unstressed",
        action="store_true",
        help="TTS: add 0 to vowels with no stress marker",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for fetched dictionary files (default: data)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAIN_PRUNE,
        help="Post-prune the tree using a validation split (default: enabled)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Fraction held out for pruning when enabled (default: 0.05)",
    )

    parser.set_defaults(func=handle_recipe)


def handle_recipe(args):
    """Handle recipe command."""
    return _build_g2p(args)


def _get_stress_mode(args):
    """Determine stress processing mode from args."""
    if args.preset == "pocketsphinx":
        return "remove"  # Remove all stress
    # TTS preset
    if args.keep_secondary:
        return "keep_primary_secondary"  # Keep 1 and 2
    return "keep_primary"  # Keep only 1


def _stress_transform(
    mode: str, mark_unstressed: bool
) -> tuple[Callable[[list[str]], list[str]], dict[str, int]]:
    """Build the recipe's phone-only stress transform and counters."""
    vowels = {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
    keep_markers = {"1"} if mode == "keep_primary" else {"1", "2"}
    stats = {"entries": 0, "stress_removed": 0, "unstressed_added": 0}

    def transform(phones: list[str]) -> list[str]:
        cooked = []
        for phone in phones:
            if phone[-1:] not in "012":
                cooked.append(phone)
                continue
            base, stress = phone[:-1], phone[-1]
            if stress in keep_markers:
                cooked.append(phone)
                continue
            stats["stress_removed"] += 1
            if mark_unstressed and base in vowels:
                cooked.append(base + "0")
                stats["unstressed_added"] += 1
            else:
                cooked.append(base)
        stats["entries"] += 1
        return cooked

    return transform, stats


def _build_g2p(args):
    """Build G2P model from a dictionary source through shared training."""
    from ...training import (
        default_alignments_path,
        train_g2p,
        validate_training_paths,
    )

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    verbose = args.verbose
    source = args.source
    is_runner = output_path.suffix == ".py"

    if source.lower() == "cmudict":
        from ...dictionary import Dictionary

        print("Fetching CMUdict...", file=sys.stderr)
        try:
            dictionary = Dictionary.fetch("cmudict", data_dir=data_dir, verbose=verbose)
            if dictionary.path is None:
                raise ValueError("fetched dictionary has no path")
            dict_path = dictionary.path
        except Exception as error:
            print(f"Error fetching dictionary: {error}", file=sys.stderr)
            return 1
    else:
        dict_path = Path(source)
        if not dict_path.is_file():
            print(f"Error: Dictionary file not found: {dict_path}", file=sys.stderr)
            return 1
        print(f"Using dictionary: {dict_path}", file=sys.stderr)

    checkpoint = (
        output_path.with_suffix(".alignments.txt")
        if is_runner
        else default_alignments_path(output_path)
    )
    try:
        validate_training_paths(dict_path, output_path, checkpoint)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    stress_mode = _get_stress_mode(args)
    print(f"Preset: {args.preset} (stress: {stress_mode})", file=sys.stderr)
    print("Training G2P model (this may take a minute)...", file=sys.stderr)
    validation_split = args.validation_split if args.prune else 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        training_dictionary = dict_path
        remove_stress = stress_mode == "remove"
        if not remove_stress:
            from ...dictionary import Dictionary

            training_dictionary = Path(temp_dir) / "filtered.dict"
            transform, stats = _stress_transform(stress_mode, args.mark_unstressed)
            Dictionary(dict_path).process(
                output=training_dictionary,
                phone_transform=transform,
                phoneset="cmu",
            )
            if verbose:
                print(
                    f"Stress filter: {stats['entries']} entries, "
                    f"{stats['stress_removed']} markers removed, "
                    f"{stats['unstressed_added']} unstressed added",
                    file=sys.stderr,
                )
        result = train_g2p(
            training_dictionary,
            locale="en_US",
            phoneset="cmu",
            remove_stress=remove_stress,
            output=None if is_runner else output_path,
            alignments_out=checkpoint if is_runner else None,
            validation_split=validation_split,
            prune=args.prune,
            parallel_align=False,
            trainer="native",
            verbose=verbose,
        )

    if is_runner:
        _bundle_runner(result.model, str(output_path))
    else:
        print(f"Done: {output_path}", file=sys.stderr)
        print(
            f"\nUsage: phonebox pronounce hello world -m {output_path}", file=sys.stderr
        )
    return 0


def _bundle_runner(dt, output_path: str) -> None:
    """Export model to temp file, bundle as standalone .py runner, and print usage."""
    from ...bundler import bundle_g2p

    with tempfile.NamedTemporaryFile(suffix=".g2p.gz", delete=False) as tmp:
        model_path = tmp.name

    try:
        dt.export(model_path)
        print(f"Bundling to {output_path}...", file=sys.stderr)
        bundle_g2p(model_path, output_path)
    finally:
        Path(model_path).unlink(missing_ok=True)

    print(f"Done: {output_path}", file=sys.stderr)
    print(f'\nUsage: python {output_path} "Hello, world!"', file=sys.stderr)
