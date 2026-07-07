#!/usr/bin/env python
"""
Main CLI entry point for phonebox with subcommands.

Usage:
    phonebox <command> [<args>]

Commands:
    recipe          Build G2P from dictionary (fetch, train, bundle)
    train           Train 1:1 G2P with safe defaults
    train-multigram Train n:m MultigramG2P (joint Viterbi)
    compare         Compare 1:1 vs MultigramG2P (eval)
    pronounce       Get pronunciations (1:1 or n:m model)
    normalize       Normalize text for G2P (preview tokenization)
    bundle          Create standalone G2P predictor
    model           Model operations (build, train, benchmark)
    dict            Dictionary operations (fetch, export-vectors)
    align           Align letters to phonemes
    vectorize       Convert alignments to feature vectors
    check           Validate lexicon against phoneset
    suggest-joins   Discover letter/phone joins (multigram EM)
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from .commands.align import setup_align_commands
from .commands.bundle import setup_bundle_command
from .commands.check import setup_check_command
from .commands.compare import setup_compare_commands
from .commands.dict import setup_dict_commands
from .commands.model import setup_model_commands
from .commands.normalize import setup_normalize_command
from .commands.pronounce import setup_pronounce_command
from .commands.recipe import setup_recipe_commands
from .commands.suggest_joins import setup_suggest_joins_command
from .commands.train import setup_train_command
from .commands.train_multigram import setup_train_multigram_command
from .commands.vectorize import setup_vectorize_command


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="phonebox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Fast, lightweight grapheme-to-phoneme conversion.

Quick Start:
  recipe       Build complete G2P from dictionary (one command)

Using Models:
  pronounce    Get pronunciations for words
  normalize    Preview text normalization/tokenization
  bundle       Create standalone executable with embedded model

Building Models:
  train        Train 1:1 G2P from a lexicon (safe defaults)
  train-multigram  Train n:m MultigramG2P (+ .units.json sidecar)
  model        Model operations (build, train, benchmark)
  dict         Dictionary operations (fetch, export-vectors)

Quality / locale tuning:
  check        Validate lexicon against canonical phoneset
  suggest-joins  Discover config.json join candidates (multigram EM)
  compare      1:1 vs n:m eval (locale or all six IPA locales)

Low-Level:
  align        Align letters to phonemes (EM algorithm)
  vectorize    Convert alignments to feature vectors

See docs/G2P_EVAL.md for compare metrics and frozen baselines.""",
        epilog="""
Examples:
  # One-liner: build G2P from CMUdict
  phonebox recipe cmudict pocketsphinx -o g2p.py

  # Use the bundled G2P
  python g2p.py "Hello, world!"

  # Preview text normalization
  phonebox normalize "Hello, world!"

  # Get pronunciation with existing model
  phonebox pronounce hello world -m model.g2p.gz

For more help:
  phonebox <command> --help
  phonebox help <command>
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"phonebox {__version__}",
    )

    # Handle "phonebox help <command>" pattern
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        if len(sys.argv) > 2:
            # phonebox help <command> -> phonebox <command> --help
            sys.argv = [sys.argv[0], sys.argv[2], "--help"]
        else:
            # phonebox help -> phonebox --help
            sys.argv = [sys.argv[0], "--help"]

    # Handle "phonebox <command> help [<subcommand>]" pattern
    if len(sys.argv) > 2 and sys.argv[2] == "help":
        if len(sys.argv) > 3:
            # phonebox <command> help <subcommand> -> phonebox <command> <subcommand> --help
            sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3], "--help"]
        else:
            # phonebox <command> help -> phonebox <command> --help
            sys.argv = [sys.argv[0], sys.argv[1], "--help"]

    # Create subparsers for command groups
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to run",
    )

    # Setup all commands
    setup_recipe_commands(subparsers)
    setup_pronounce_command(subparsers)
    setup_normalize_command(subparsers)
    setup_bundle_command(subparsers)
    setup_model_commands(subparsers)
    setup_dict_commands(subparsers)
    setup_align_commands(subparsers)
    setup_vectorize_command(subparsers)
    setup_check_command(subparsers)
    setup_train_command(subparsers)
    setup_train_multigram_command(subparsers)
    setup_compare_commands(subparsers)
    setup_suggest_joins_command(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Execute the command
    if hasattr(args, "func"):
        result = args.func(args)
        sys.exit(result if result is not None else 0)
    elif hasattr(args, "parser"):
        # Subcommand group called with no subcommand - show its help
        args.parser.print_help()
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
