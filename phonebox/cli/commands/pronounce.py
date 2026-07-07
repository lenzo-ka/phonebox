#!/usr/bin/env python
"""Pronounce command: word-level grapheme-to-phoneme conversion."""

from __future__ import annotations

import sys
from pathlib import Path

from ...normalize import normalize_nfc
from ...utils.io import is_dict_comment


def _multigram_sidecar(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".units.json")


def setup_pronounce_command(subparsers):
    """Setup pronounce command."""
    pronounce_parser = subparsers.add_parser(
        "pronounce",
        help="Get pronunciations for words",
        description="Convert words to phoneme sequences",
    )
    pronounce_parser.add_argument(
        "words",
        nargs="*",
        help="Words to pronounce (if none, reads from stdin)",
    )
    pronounce_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model file path",
    )
    pronounce_parser.add_argument(
        "-n",
        "--nbest",
        type=int,
        metavar="N",
        help="Output n-best pronunciations with scores",
    )
    pronounce_parser.add_argument(
        "--with-confidence",
        action="store_true",
        help="Output confidence scores",
    )
    pronounce_parser.add_argument(
        "--confidence-detailed",
        action="store_true",
        help="Output per-phoneme confidence (requires --with-confidence)",
    )
    pronounce_parser.add_argument(
        "--confidence-method",
        choices=["average", "product"],
        default="average",
        help="Overall confidence method (requires --with-confidence)",
    )
    pronounce_parser.add_argument(
        "--locale",
        help="Locale tag for MultigramG2P letter cooking (or override model metadata)",
    )
    pronounce_parser.add_argument(
        "--phoneset",
        help="Phoneset for MultigramG2P letter cooking (or override model metadata)",
    )
    pronounce_parser.set_defaults(func=handle_pronounce)


def handle_pronounce(args):
    """Handle 'phonebox pronounce' command with user-friendly input."""
    model_path = Path(args.model)
    multigram = _multigram_sidecar(model_path).is_file()

    if multigram:
        if args.nbest or args.with_confidence:
            print(
                "Error: n-best and confidence are not supported for MultigramG2P models",
                file=sys.stderr,
            )
            return 1
        from ...core.multigram_g2p import MultigramG2P
        from ...core.vectorizer import Vectorizer

        try:
            mg = MultigramG2P.load(model_path)
        except Exception as e:
            print(f"Error loading multigram model: {e}", file=sys.stderr)
            return 1
        locale = args.locale or mg.locale
        phoneset = args.phoneset or mg.phoneset_name or "ipa"
        vec = None
        if locale:
            vec = Vectorizer(locale=locale, phoneset_name=phoneset, remove_stress=False)
        if locale is None:
            print(
                "Warning: no locale in model metadata; using per-character letters "
                "(pass --locale for joined graphemes)",
                file=sys.stderr,
            )

        def pronounce_word(w: str) -> list[str]:
            if vec is not None:
                return mg.pronounce_letters(vec.cook_letters(w, g2p=True), word=w)
            return mg.pronounce(w)

        tree_g2p = None
    else:
        from ...converter import G2P

        try:
            tree_g2p = G2P(model=args.model)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return 1

        def pronounce_word(w: str) -> list[str]:
            return tree_g2p.pronounce(w)

    # Determine input source
    if args.words:
        words = args.words
    elif not sys.stdin.isatty():
        words = []
        for line in sys.stdin:
            line = normalize_nfc(line)
            if not is_dict_comment(line):
                words.extend(line.split())
    else:
        # Interactive mode
        print("G2P mode. Enter words (Ctrl+D to exit):", file=sys.stderr)
        words = []
        try:
            while True:
                try:
                    line = normalize_nfc(input("> "))
                    if not is_dict_comment(line):
                        words.extend(line.split())
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted", file=sys.stderr)
            return 130

    for word in words:
        word = normalize_nfc(word)

        if not word:
            continue

        try:
            # N-best mode
            if args.nbest:
                if tree_g2p is None:
                    raise RuntimeError("unreachable: n-best checked for multigram")
                nbest = tree_g2p.pronounce_nbest(word, n=args.nbest)
                for phones, score in nbest:
                    print(f"{word}\t{' '.join(phones)}\t{score:.4f}")

            # Confidence mode
            elif args.with_confidence:
                if tree_g2p is None:
                    raise RuntimeError("unreachable: confidence checked for multigram")
                phones, confidences = tree_g2p.pronounce_with_confidence(word)

                if args.confidence_detailed:
                    # Per-phoneme confidence
                    conf_str = " ".join(f"{c:.2f}" for c in confidences)
                    print(f"{word}\t{' '.join(phones)}\t{conf_str}")
                else:
                    # Overall confidence
                    if args.confidence_method == "product":
                        overall_conf = 1.0
                        for c in confidences:
                            overall_conf *= c
                    else:  # average
                        overall_conf = (
                            sum(confidences) / len(confidences) if confidences else 1.0
                        )
                    print(f"{word}\t{' '.join(phones)}\t{overall_conf:.4f}")

            # Standard 1-best mode
            else:
                phones = pronounce_word(word)
                print(f"{word}\t{' '.join(phones)}")

        except Exception as e:
            print(f"{word}\t[ERROR: {e}]", file=sys.stderr)

    return 0
