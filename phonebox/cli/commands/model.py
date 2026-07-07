#!/usr/bin/env python
"""Model operations: build, train, benchmark."""

from __future__ import annotations

import argparse
import sys

from ...constants import DEFAULT_PHONESET, FILE_ENCODING


def setup_model_commands(subparsers):
    """Setup model subcommands."""
    model_parser = subparsers.add_parser(
        "model",
        help="Model operations (build, train, benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build, train, and benchmark G2P models.",
        epilog="""
Examples:
  # Build model from dictionary
  phonebox model build en_US cmudict.txt -o model.g2p.gz

  # Train from pre-computed alignments
  phonebox model train en_US --alignments aligned.txt -o model.g2p.gz

  # Benchmark model performance
  phonebox model benchmark model.g2p.gz
""",
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    model_parser.set_defaults(parser=model_parser)

    # phonebox model build
    build_parser = model_subparsers.add_parser(
        "build",
        help="Build complete model from dictionary",
        description="Build a complete G2P model (align + train + export)",
    )
    build_parser.add_argument("locale", nargs="?", help="Language locale (e.g., en_US)")
    build_parser.add_argument("dict", nargs="?", help="Dictionary file path")
    build_parser.add_argument("-o", "--output", help="Output model file")
    build_parser.add_argument(
        "--phoneset",
        default=DEFAULT_PHONESET,
        help="Phoneset tag (e.g. cmu, xsampa, ipa); free-form, only "
        "selects stress-stripping rules and locale join keys",
    )
    build_parser.add_argument(
        "--remove-stress", action="store_true", help="Remove stress markers"
    )
    build_parser.add_argument("--cased", action="store_true", help="Case sensitive")
    build_parser.add_argument(
        "--norm-xlit", action="store_true", help="Normalize transliteration"
    )
    build_parser.add_argument("--max-iterations", type=int, help="Max EM iterations")
    build_parser.add_argument(
        "--max-combinations", type=int, help="Max combinations per alignment"
    )
    build_parser.add_argument(
        "-c", "--config", help="Config file (YAML/TOML/JSON) - CLI args override"
    )
    build_parser.add_argument(
        "--save-config",
        help="Save final merged config to file (after CLI overrides)",
    )
    build_parser.add_argument(
        "--prune",
        action="store_true",
        help="Post-prune the tree using a validation split (reduces overfitting)",
    )
    build_parser.add_argument(
        "--validation-split",
        type=float,
        help="Fraction held out for pruning (default 0.05 when --prune is set)",
    )
    build_parser.add_argument(
        "--test-split",
        type=float,
        help="Fraction held out for held-out test evaluation",
    )
    build_parser.set_defaults(func=handle_model_build)

    # phonebox model train
    train_parser = model_subparsers.add_parser(
        "train",
        help="Train decision tree model",
        description="Train a decision tree from aligned data",
    )
    train_parser.add_argument("locale", help="Language locale")
    train_parser.add_argument("-d", "--dict", help="Dictionary file")
    train_parser.add_argument("-a", "--alignments", help="Alignment file")
    train_parser.add_argument("--vectors", help="Vectorized data file")
    train_parser.add_argument("-o", "--output", help="Output model file")
    train_parser.add_argument("--remove-stress", action="store_true")
    train_parser.add_argument("--cased", action="store_true")
    train_parser.add_argument("--max-iterations", type=int)
    train_parser.add_argument("--max-combinations", type=int)
    train_parser.add_argument(
        "--phoneset",
        default=DEFAULT_PHONESET,
        help="Phoneset tag (e.g. cmu, xsampa, ipa); free-form",
    )
    train_parser.add_argument(
        "--norm-xlit", action="store_true", help="Normalize transliteration"
    )
    train_parser.add_argument(
        "--target-first",
        action="store_true",
        help="Target column is first (default: last)",
    )
    train_parser.add_argument(
        "--trainer",
        default="native",
        choices=["native", "sklearn"],
        help="Trainer backend (default: native)",
    )
    train_parser.add_argument(
        "--criterion",
        default="entropy",
        choices=["entropy", "gini"],
        help="Split criterion (default: entropy)",
    )
    train_parser.add_argument(
        "--prune",
        action="store_true",
        help="Post-prune the tree using a validation split (reduces overfitting)",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Fraction held out for pruning (set together with --prune, e.g. 0.05)",
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Fraction held out for held-out test evaluation",
    )
    train_parser.set_defaults(func=handle_model_train)

    # phonebox model benchmark
    benchmark_parser = model_subparsers.add_parser(
        "benchmark",
        help="Benchmark model performance",
        description="Benchmark model loading and inference speed",
    )
    benchmark_parser.add_argument("model", help="Model file")
    benchmark_parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )
    benchmark_parser.set_defaults(func=handle_model_benchmark)


def handle_model_build(args):
    """Handle 'phonebox model build' command."""
    from ...config_loader import DEFAULT_CONFIG, load_config, merge_configs

    # Start with defaults
    config = dict(DEFAULT_CONFIG)

    # Load from config file if provided
    if args.config:
        print(f"Loading config: {args.config}", file=sys.stderr)
        file_config = load_config(args.config)
        config = merge_configs(config, file_config)

    # Override with CLI args (CLI takes precedence)
    if args.locale:
        config["locale"] = args.locale
    if args.dict:
        config["dictionary"] = args.dict
    if args.output:
        config["output"] = args.output
    if args.phoneset:
        config["phoneset"] = args.phoneset
    if args.remove_stress:
        config["remove_stress"] = True
    if args.cased:
        config["cased"] = True
    if args.norm_xlit:
        config["norm_xlit"] = True
    if args.max_iterations:
        config["max_iterations"] = args.max_iterations
    if args.max_combinations:
        config["max_combinations"] = args.max_combinations
    if args.prune:
        config["prune"] = True
    if args.validation_split is not None:
        config["validation_split"] = args.validation_split
    if args.test_split is not None:
        config["test_split"] = args.test_split

    # Save merged config if requested (before validation)
    if args.save_config:
        from ...config_loader import save_config

        print(f"Saving merged config to: {args.save_config}", file=sys.stderr)
        save_config(config, args.save_config)
        print("Done: Config saved", file=sys.stderr)

    # Validate required fields
    if not config.get("dictionary"):
        print(
            "Error: No dictionary specified (use dict arg or config file)",
            file=sys.stderr,
        )
        return 1
    if not config.get("locale"):
        print(
            "Error: No locale specified (use locale arg or config file)",
            file=sys.stderr,
        )
        return 1

    # Build using merged config
    from ...config_builder import train_from_config

    print(
        f"Building model: {config.get('locale')} from {config.get('dictionary')}",
        file=sys.stderr,
    )
    print(f"  Trainer: {config.get('trainer', 'native')}", file=sys.stderr)

    try:
        train_from_config(config)

        if config.get("output"):
            print(f"Done: Model saved to {config['output']}", file=sys.stderr)
        else:
            print("Done: Model built (no output file specified)", file=sys.stderr)

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_model_train(args):
    """Handle 'phonebox model train' command."""
    from ...constants import DICT_ENCODING
    from ...core.g2p_model import G2PDecisionTree

    print(f"Training model for {args.locale}", file=sys.stderr)

    # Create decision tree
    dt = G2PDecisionTree(
        locale=args.locale,
        phoneset_name=args.phoneset,
        remove_stress=args.remove_stress,
        cased=args.cased,
        verbose=True,
        max_iterations=args.max_iterations,
        max_combinations=args.max_combinations,
        norm_xlit=args.norm_xlit,
        trainer=args.trainer,
        criterion=args.criterion,
    )

    # Load data
    if args.dict:
        print(f"Loading dictionary: {args.dict}", file=sys.stderr)
        with open(args.dict, encoding=DICT_ENCODING) as f:
            dt.load_prondict(f)
        dt.align()
    elif args.alignments:
        print(f"Loading alignments: {args.alignments}", file=sys.stderr)
        with open(args.alignments, encoding=FILE_ENCODING) as f:
            dt.load_alignments(f)
    elif args.vectors:
        print(f"Loading vectors: {args.vectors}", file=sys.stderr)
        # For vectors, we need to parse them differently
        from ...core.vectorizer import Vectorizer

        vectorizer = Vectorizer(
            locale=args.locale,
            phoneset_name=args.phoneset,
            remove_stress=args.remove_stress,
            cased=args.cased,
            target_position="first" if args.target_first else "last",
        )

        vectors_dict = vectorizer.load_vectors_file(args.vectors)
        X, y, counts = vectorizer.parse_vectors_to_data(vectors_dict)
        dt.load_vectors_data(X, y, counts)
    else:
        print("Error: Must specify --dict, --alignments, or --vectors", file=sys.stderr)
        return 1

    # Train
    if args.prune and args.validation_split <= 0:
        print(
            "Warning: --prune set without --validation-split; defaulting to 0.05.",
            file=sys.stderr,
        )
        validation_split = 0.05
    else:
        validation_split = args.validation_split

    print(f"Training with {args.trainer} backend...", file=sys.stderr)
    if args.prune:
        print(
            f"  Pruning enabled (validation_split={validation_split})",
            file=sys.stderr,
        )
    dt.train(
        validation_split=validation_split,
        test_split=args.test_split,
        prune=args.prune,
    )

    # Export
    if args.output:
        print(f"Exporting to {args.output}", file=sys.stderr)
        dt.export(args.output)

    suffix = "" if args.output else " (no output file specified)"
    print(f"Done: Training complete{suffix}", file=sys.stderr)
    return 0


def handle_model_benchmark(args):
    """Handle 'phonebox model benchmark' command."""
    import time

    from ...converter import G2P

    print(f"Benchmarking: {args.model}")
    print(f"Iterations: {args.iterations}")
    print()

    print("Testing model loading...")
    start = time.perf_counter()
    for _ in range(args.iterations):
        g2p = G2P(model=args.model)
    load_time = time.perf_counter() - start
    print(f"  Total: {load_time:.3f}s")
    print(f"  Average: {load_time / args.iterations * 1000:.1f}ms per load")
    print()

    print("Testing pronunciation...")
    test_words = ["hello", "world", "python", "test", "benchmark"]
    start = time.perf_counter()
    for _ in range(args.iterations):
        for word in test_words:
            _ = g2p.pronounce(word)
    pron_time = time.perf_counter() - start
    total_calls = args.iterations * len(test_words)
    print(f"  Total: {pron_time:.3f}s ({total_calls} calls)")
    print(f"  Average: {pron_time / total_calls * 1000:.3f}ms per word")

    return 0
