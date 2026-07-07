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
from pathlib import Path

from ...constants import DICT_ENCODING
from ...utils.io import is_dict_comment


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
        action="store_true",
        help="Post-prune the tree using a validation split (reduces overfitting)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.05,
        help="Fraction held out for pruning when --prune is set (default 0.05)",
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


def _filter_stress(dict_path, mode, mark_unstressed, verbose):
    """
    Filter stress markers in dictionary.

    Args:
        dict_path: Path to dictionary file
        mode: "keep_primary" or "keep_primary_secondary"
        mark_unstressed: Add 0 to vowels with no stress after filtering
        verbose: Print stats

    Yields:
        Processed dictionary lines
    """

    # CMU vowels (without stress markers)
    VOWELS = {
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
    stats = {"lines": 0, "stress_removed": 0, "unstressed_added": 0}

    with open(dict_path, encoding=DICT_ENCODING) as f:
        for line in f:
            line = line.strip()
            if is_dict_comment(line):
                continue

            # Split word and phones
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            word, phones_str = parts
            phones = phones_str.split()
            new_phones = []

            for phone in phones:
                # Check if it's a stressed vowel (ends with 0, 1, or 2)
                if phone[-1:] in "012":
                    base = phone[:-1]
                    stress = phone[-1]

                    if stress in keep_markers:
                        # Keep this stress marker
                        new_phones.append(phone)
                    else:
                        # Remove stress marker
                        stats["stress_removed"] += 1
                        if mark_unstressed and base in VOWELS:
                            # Add 0 for unstressed
                            new_phones.append(base + "0")
                            stats["unstressed_added"] += 1
                        else:
                            new_phones.append(base)
                else:
                    new_phones.append(phone)

            stats["lines"] += 1
            yield f"{word}\t{' '.join(new_phones)}\n"

    if verbose:
        print(
            f"Stress filter: {stats['lines']} lines, "
            f"{stats['stress_removed']} markers removed, "
            f"{stats['unstressed_added']} unstressed added",
            file=sys.stderr,
        )


def _build_g2p(args):
    """Build G2P model from dictionary source."""
    from ...core.g2p_model import G2PDecisionTree

    data_dir = Path(args.data_dir)
    output_path = args.output
    verbose = args.verbose
    source = args.source

    # Determine if we're making a runner or just a model
    is_runner = output_path.endswith(".py")

    # Step 1: Get dictionary (local file or fetch)
    if source.lower() == "cmudict":
        from ...dictionary import Dictionary

        print("Fetching CMUdict...", file=sys.stderr)
        try:
            dict_obj = Dictionary.fetch("cmudict", data_dir=data_dir, verbose=verbose)
            dict_path = dict_obj.path
        except Exception as e:
            print(f"Error fetching dictionary: {e}", file=sys.stderr)
            return 1
    else:
        dict_path = source
        if not Path(dict_path).exists():
            print(f"Error: Dictionary file not found: {dict_path}", file=sys.stderr)
            return 1
        print(f"Using dictionary: {dict_path}", file=sys.stderr)

    # Determine stress processing
    stress_mode = _get_stress_mode(args)
    mark_unstressed = args.mark_unstressed

    print(f"Preset: {args.preset} (stress: {stress_mode})", file=sys.stderr)

    # Step 2: Build model
    print("Training G2P model (this may take a minute)...", file=sys.stderr)

    # For pocketsphinx, use built-in remove_stress
    # For TTS modes, we preprocess the dictionary
    validation_split = args.validation_split if args.prune else 0.0
    if args.prune:
        print(
            f"Pruning enabled (validation_split={validation_split})",
            file=sys.stderr,
        )

    if stress_mode == "remove":
        dt = G2PDecisionTree(
            locale="en_US",
            phoneset_name="cmu",
            remove_stress=True,
            verbose=verbose,
            trainer="native",
        )
        dt.train_from_dict(
            str(dict_path),
            validation_split=validation_split,
            prune=args.prune,
        )
    else:
        # TTS mode: filter stress markers before training
        dt = G2PDecisionTree(
            locale="en_US",
            phoneset_name="cmu",
            remove_stress=False,
            verbose=verbose,
            trainer="native",
        )
        processed_lines = _filter_stress(
            dict_path, stress_mode, mark_unstressed, verbose
        )
        dt.load_prondict(iter(processed_lines))
        dt.align()
        dt.train(validation_split=validation_split, prune=args.prune)

    if is_runner:
        _bundle_runner(dt, output_path)
    else:
        dt.export(output_path)
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
