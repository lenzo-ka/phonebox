#!/usr/bin/env python
"""
G2P runner using cartlet's lightweight `.cart` inference path.

For deployment, use the bundled G2P predictor instead:

    # Create standalone Python executable with the model embedded
    phonebox bundle model.g2p.gz -o g2p.py

    # Use standalone (zero dependencies)
    python g2p.py hello world

    # Or as library
    from g2p import G2PPredictor
    g2p = G2PPredictor.from_embedded()
    phones = g2p.pronounce('hello')

This module is for use within the phonebox library:
    from phonebox.runner import G2PRunner
    g2p = G2PRunner('model.cart')
    phones = g2p.pronounce('hello')

CLI:
    python -m phonebox.runner model.cart hello world
"""

from __future__ import annotations

import sys

from cartlet import Predictor

from .constants import AETHER, CONTEXT_WINDOW_SIZE, EPSILON, JOIN_CHAR
from .core.vectorizer import join_seq, make_join_re


class G2PRunner(Predictor):
    """
    Lightweight G2P inference over `.cart` models.

    Extends ``cartlet.Predictor`` with phonebox-specific behavior: applies
    the vectorizer's context window and digraph joining, emits epsilon for
    out-of-vocabulary center letters, and uncooks joined phones on output.
    """

    def __init__(self, model_path: str):
        """
        Load g2p model.

        Args:
            model_path: Path to a `.cart` model file. Other formats
                (`.g2p.gz`, `.jsonl.gz`, `.json`, `.pkl`) need to be
                converted to `.cart` first (e.g. via `phonebox bundle`
                or `G2PDecisionTree.load_model(...).export(...)`).
        """
        super().__init__(model_path)
        # cartlet's Predictor.metadata exposes the .cart trailer dict; it
        # carries phonebox's vectorizer config (width, cased, join, exceptions).
        meta = self.metadata

        self.width = meta.get("width", CONTEXT_WINDOW_SIZE)
        self.aether = meta.get("aether", AETHER)
        self.epsilon = meta.get("epsilon", EPSILON)
        self.join_char = meta.get("join_char", JOIN_CHAR)
        self.cased = meta.get("cased", False)
        self.exceptions = meta.get("exceptions", {})

        self.center_position = (self.width - 1) // 2
        self.padding = [self.aether] * self.center_position

        # Letter-joining regex (handles locales that define digraphs
        # like German "ch"/"sch", Welsh "ll"/"dd", French "eau"); when the
        # locale lists no letter joinings this stays None and vectorize_word
        # falls back to plain character splitting.
        letter_joins = meta.get("join", {}).get("letters", [])
        self.lett_join_re = make_join_re(letter_joins)

    def vectorize_word(self, word: str) -> list[list[str]]:
        """
        Convert word to context vectors.

        Applies letter cooking (digraph joining) so the runner produces the
        same context windows the model was trained on.

        Args:
            word: Word to vectorize

        Returns:
            List of feature vectors (one per cooked letter token)
        """
        if not self.cased:
            word = word.lower()

        letters = list(word)
        if self.lett_join_re is not None:
            letters = join_seq(self.lett_join_re, letters, self.join_char)

        padded = self.padding + letters + self.padding
        return [padded[i : i + self.width] for i in range(len(letters))]

    def uncook(self, phonemes: list[str]) -> list[str]:
        """
        Uncook phonemes: split joined phones and remove epsilon.

        Args:
            phonemes: Raw phonemes from model

        Returns:
            Final phoneme list
        """
        result = []
        for phoneme in phonemes:
            if phoneme == self.epsilon:
                continue
            result.extend(phoneme.split(self.join_char))
        return result

    def pronounce(self, word: str) -> list[str]:
        """
        Pronounce a word with OOV handling.

        Strategy:
        - Check exceptions first
        - OOV letters at center → epsilon (silent)
        - Context OOV → handled naturally by tree

        Args:
            word: Word to pronounce

        Returns:
            List of phonemes
        """
        # Check exceptions first
        lookup_word = word if self.cased else word.lower()
        if lookup_word in self.exceptions:
            return self.exceptions[lookup_word]

        # Model prediction with OOV checking
        vectors = self.vectorize_word(word)
        raw_phones = []

        for vec in vectors:
            center_letter = vec[self.center_position]

            if self.is_oov(self.center_position, center_letter):
                # OOV letter → epsilon (will be removed by uncook)
                raw_phones.append(self.epsilon)
            else:
                # Known letter → predict via tree
                raw_phones.append(self.predict(vec))

        return self.uncook(raw_phones)

    def pronounce_batch(self, words: list[str]) -> list[tuple[str, list[str]]]:
        """
        Pronounce multiple words.

        Args:
            words: List of words

        Returns:
            List of (word, phonemes) tuples
        """
        return [(word, self.pronounce(word)) for word in words]

    def __call__(self, word: str) -> list[str]:
        """Allow runner to be called directly: runner('word')"""
        return self.pronounce(word)

    def __repr__(self) -> str:
        locale = self.metadata.get("locale", "unknown")
        phoneset = self.metadata.get("phoneset_name", "unknown")
        return f"G2PRunner(locale={locale}, phoneset={phoneset})"


# =============================================================================
# CLI
# =============================================================================


def main():
    """Command-line interface."""
    import argparse

    from phonebox.normalize import normalize_nfc
    from phonebox.utils.io import is_dict_comment

    parser = argparse.ArgumentParser(
        description="Phonebox standalone runner",
        epilog="""
Examples:
  python -m phonebox.runner model.cart hello world
  echo "hello world" | python -m phonebox.runner model.cart
        """,
    )

    parser.add_argument("model", help="Model file (.cart)")
    parser.add_argument("words", nargs="*", help="Words to pronounce")

    args = parser.parse_args()

    # Load
    try:
        g2p = G2PRunner(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Get input
    if args.words:
        words = args.words
    elif not sys.stdin.isatty():
        words = []
        for line in sys.stdin:
            line = normalize_nfc(line)
            if not is_dict_comment(line):
                words.extend(line.split())
    else:
        # Interactive
        locale = g2p.metadata.get("locale", "unknown")
        print(f"Phonebox ({locale}). Enter words (Ctrl+D to exit):", file=sys.stderr)
        words = []
        try:
            while True:
                line = normalize_nfc(input("> "))
                if not is_dict_comment(line):
                    for word in line.split():
                        phones = g2p.pronounce(word)
                        print(f"{word}\t{' '.join(phones)}")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0

    # Process
    for word in words:
        word = normalize_nfc(word)
        if word:
            phones = g2p.pronounce(word)
            print(f"{word}\t{' '.join(phones)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
