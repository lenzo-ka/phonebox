#!/usr/bin/env python
"""
G2P Predictor - extends cartlet's Predictor with phoneme-specific logic.

This template is bundled with a model using:
    phonebox bundle model.g2p.gz -o g2p.py

The bundler:
1. Uses cartlet's bundler with library_only=True
2. Appends this G2P wrapper code (vectorization, uncooking, CLI)

Usage (bundled):
    python g2p.py "Hello, world!"     # Tokenizes and pronounces
    from g2p import G2PPredictor; g2p = G2PPredictor.from_embedded()

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

# NOTE: This file is appended to cartlet's predict.py during bundling.
# The Predictor class, load_embedded(), etc. are defined above.
#
# The bundler slices source from the G2PPredictor class header to EOF, so
# any module-level helpers MUST appear after the class definition (see the
# helper section below) to be included in the bundled output.


# =============================================================================
# G2P Predictor - extends Predictor with phoneme handling
# =============================================================================


class G2PPredictor(Predictor):  # type: ignore[name-defined]  # noqa: F821
    """
    G2P predictor with word-to-phoneme conversion.

    Extends cartlet's Predictor with:
    - Word vectorization (letter context windows, including digraph joining)
    - Phoneme uncooking (split joined phones, remove epsilon)
    - OOV handling

    Example:
        g2p = G2PPredictor.from_embedded()
        phones = g2p.pronounce("hello")  # ['HH', 'AH', 'L', 'OW']
    """

    # Defensive defaults: applied only when the .cart file ships with no
    # metadata trailer at all. Phonebox always writes one; this fallback
    # keeps the bundle predictable rather than blowing up.
    DEFAULT_WIDTH = 7
    DEFAULT_AETHER = "\u2022"  # •
    DEFAULT_EPSILON = "\u2205"  # ∅
    DEFAULT_JOIN_CHAR = "\u208a"  # ₊
    DEFAULT_CASED = False

    def __init__(self, path_or_bytes=None):
        super().__init__(path_or_bytes)
        self._apply_meta()

    def _apply_meta(self):
        """Apply g2p config from model metadata (or defaults when empty)."""
        # Predictor.metadata reads self._model["metadata"] safely.
        meta = self.metadata

        self.width = meta.get("width", self.DEFAULT_WIDTH)
        self.aether = meta.get("aether", self.DEFAULT_AETHER)
        self.epsilon = meta.get("epsilon", self.DEFAULT_EPSILON)
        self.join_char = meta.get("join_char", self.DEFAULT_JOIN_CHAR)
        self.cased = meta.get("cased", self.DEFAULT_CASED)
        self.exceptions = meta.get("exceptions", {})

        self.center_position = (self.width - 1) // 2
        self.padding = [self.aether] * self.center_position

        letter_joins = meta.get("join", {}).get("letters", [])
        self.lett_join_re = _make_join_re(letter_joins)

    @classmethod
    def from_embedded(cls):
        """Create G2PPredictor from embedded model data."""
        model_data = load_embedded()  # type: ignore[name-defined]  # noqa: F821 - defined by cartlet
        if model_data is None:
            raise ValueError("No embedded model found")
        obj = cls()
        obj._model = model_data
        obj._apply_meta()
        return obj

    def vectorize_word(self, word):
        """Convert word to context vectors (applies digraph joining)."""
        if not self.cased:
            word = word.lower()

        letters = list(word)
        if self.lett_join_re is not None:
            letters = _join_seq(self.lett_join_re, letters, self.join_char)

        padded = self.padding + letters + self.padding
        return [padded[i : i + self.width] for i in range(len(letters))]

    def uncook(self, phonemes):
        """Split joined phones and remove epsilon."""
        result = []
        for phoneme in phonemes:
            if phoneme == self.epsilon:
                continue
            result.extend(phoneme.split(self.join_char))
        return result

    def pronounce(self, word):
        """Pronounce a word, returning list of phonemes."""
        if self._model is None:
            raise ValueError("No model loaded")

        # Check exceptions
        lookup_word = word if self.cased else word.lower()
        if lookup_word in self.exceptions:
            return self.exceptions[lookup_word]

        # Vectorize and predict, dropping any position whose center letter was
        # never seen in training (otherwise the tree falls through default
        # branches and emits a random phoneme). Matches phonebox.runner.G2PRunner.
        vectors = self.vectorize_word(word)
        raw_phones = []
        for vec in vectors:
            center = vec[self.center_position]
            if self.is_oov(self.center_position, center):
                raw_phones.append(self.epsilon)
            else:
                raw_phones.append(self.predict(vec))

        return self.uncook(raw_phones)

    def pronounce_batch(self, words):
        """Pronounce multiple words."""
        return [(word, self.pronounce(word)) for word in words]

    def tokenize(self, text):
        """
        Tokenize text into words for pronunciation.

        Normalizes and extracts pronounceable tokens:
        - NFC unicode normalization
        - Strips punctuation/symbols from token edges (Unicode categories P, S, C, M, Z)
        - Keeps internal punctuation (apostrophes, hyphens)
        """
        import unicodedata

        # Unicode categories to strip from token edges
        exclude_cats = {"P", "S", "C", "M", "Z"}

        # Normalize unicode
        text = unicodedata.normalize("NFC", text)

        # Split and normalize each token
        result = []
        for token in text.split():
            # Strip leading excluded characters
            while token and unicodedata.category(token[0])[0] in exclude_cats:
                token = token[1:]
            # Strip trailing excluded characters
            while token and unicodedata.category(token[-1])[0] in exclude_cats:
                token = token[:-1]
            if token:
                result.append(token)

        return result

    def pronounce_text(self, text, raw=False):
        """
        Pronounce text, returning list of (word, phones) tuples.

        Args:
            text: Input text
            raw: If True, skip normalization (just split on whitespace)
        """
        words = text.split() if raw else self.tokenize(text)
        return [(word, self.pronounce(word)) for word in words]

    def __call__(self, word):
        """Allow direct calling: g2p('hello')"""
        return self.pronounce(word)


# =============================================================================
# Letter-joining helpers (stdlib-only)
#
# Kept module-level so they stay easy to inline into the bundled standalone
# script. The .cart metadata trailer is parsed by cartlet's loader and
# exposed as Predictor.metadata; the helpers below only handle the digraph
# joining that's specific to G2P (not part of cartlet's domain).
# =============================================================================


def _make_join_re(join_list):
    """Compile a regex matching any space-separated joining sequence, or None."""
    import re

    if not join_list:
        return None
    items = sorted(join_list, key=len, reverse=True)
    items = [re.escape(x) for x in items]
    return re.compile(r" (" + r"|".join(items) + r") ")


def _join_seq(regex, seq, join_char):
    """Collapse adjacent tokens in seq matched by regex using join_char."""
    if not regex:
        return list(seq)
    as_str = " " + " ".join(seq) + " "
    m = regex.search(as_str)
    while m:
        s = m.group(0)
        joined = s[1:-1].replace(" ", join_char)
        as_str = as_str.replace(s, " " + joined + " ")
        m = regex.search(as_str)
    return as_str[1:-1].split()


# =============================================================================
# G2P CLI (replaces cartlet's main when bundled as G2P)
# =============================================================================


def g2p_main():
    """G2P command-line interface."""
    import argparse
    import sys

    embedded_model = load_embedded()  # type: ignore[name-defined]  # noqa: F821 - defined by cartlet
    has_embedded = embedded_model is not None

    parser = argparse.ArgumentParser(
        description="G2P: Convert text to phonemes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello, world!"        # Pronounce text (tokenizes automatically)
  %(prog)s -f input.txt           # Read text from file, tokenize each line
  echo "Hello world" | %(prog)s   # Read from stdin
""",
    )

    if has_embedded:
        parser.add_argument(
            "-m", "--model", metavar="FILE", help="Model file (overrides embedded)"
        )
    else:
        parser.add_argument("-m", "--model", metavar="FILE", help="Model file (.cart)")

    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        help="Read text from file (- for stdin), tokenize each line",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Disable normalization (no punctuation stripping)",
    )
    parser.add_argument(
        "text", nargs="*", help="Text to pronounce (tokenized into words)"
    )

    args = parser.parse_args()

    # Load model
    if args.model:
        g2p = G2PPredictor(args.model)
    elif has_embedded:
        g2p = G2PPredictor.from_embedded()
    else:
        parser.error("No model specified. Use -m MODEL or use a bundled version.")

    # Process text and output pronunciations
    def process_text(text):
        for word, phones in g2p.pronounce_text(text, raw=args.raw):
            print(f"{word}\t{' '.join(phones)}")

    # Get input
    if args.file:
        f = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")  # noqa: SIM115
        try:
            for line in f:
                process_text(line)
        finally:
            if args.file != "-":
                f.close()
    elif args.text:
        # Join all positional args as one text string
        process_text(" ".join(args.text))
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            process_text(line)
    else:
        # Interactive mode
        print("G2P Predictor. Enter text (Ctrl+D to exit):", file=sys.stderr)
        try:
            while True:
                line = input("> ")
                process_text(line)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(g2p_main())
