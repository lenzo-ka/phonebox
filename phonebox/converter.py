#!/usr/bin/env python
"""
High-level G2P API for easy pronunciation lookups.
"""

from __future__ import annotations

from pathlib import Path

from .constants import (
    DEFAULT_LOCALE,
    DEFAULT_MULTIGRAM_PHONESET,
    DEFAULT_NBEST_COUNT,
    DEFAULT_PHONESET,
    DICT_ENCODING,
)
from .core.decision_tree import DecisionTree


class G2P:
    """
    Simple, high-level interface for grapheme-to-phoneme conversion.

    Zero external dependencies - uses DecisionTree for all operations.

    Examples:
        # Use pre-trained model
        g2p = G2P(model='models/en_US_nostress.g2p.gz')
        g2p.pronounce('hello')  # ['HH', 'AH', 'L', 'OW']

        # Train from dictionary
        g2p = G2P.train('mydict.txt', locale='en_US')

        # Batch pronunciation
        words = ['hello', 'world', 'python']
        for word, phones in g2p.pronounce_batch(words):
            print(f"{word}: {phones}")
    """

    def __init__(
        self,
        model: str | Path | None = None,
        locale: str = DEFAULT_LOCALE,
        phoneset: str = DEFAULT_PHONESET,
        remove_stress: bool = False,
        use_dict_fallback: bool = True,
    ) -> None:
        """
        Initialize G2P converter.

        Args:
            model: Path to trained model file (or None to train new)
            locale: Language locale
            phoneset: Phoneset name ('cmu' or 'xsampa')
            remove_stress: Whether stress was removed in training
            use_dict_fallback: Use exceptions dictionary (hybrid lookup)
        """
        self.locale = locale
        self.phoneset = phoneset
        self.remove_stress = remove_stress
        self.use_dict_fallback = use_dict_fallback

        # Create DecisionTree
        self._dt = DecisionTree(
            locale=locale,
            phoneset_name=phoneset,
            remove_stress=remove_stress,
            use_dict_fallback=use_dict_fallback,
        )

        if model:
            # Load existing model (overwrites initial vectorizer)
            self._dt.load_model(str(model))
            self.locale = self._dt.vectorizer.locale

    def pronounce(self, word: str) -> list[str]:
        """
        Get pronunciation for a single word.

        Uses hybrid lookup:
        1. Checks exceptions dictionary first (if enabled)
        2. Falls back to model prediction

        Args:
            word: Word to pronounce

        Returns:
            List of phonemes

        Examples:
            >>> g2p = G2P(model='models/en_US_nostress.g2p.gz')
            >>> g2p.pronounce('hello')
            ['HH', 'AH', 'L', 'OW']
        """
        return self._dt.pronounce(word)

    def pronounce_batch(self, words: list[str]) -> list[tuple[str, list[str]]]:
        """
        Get pronunciations for multiple words.

        Args:
            words: List of words

        Returns:
            List of (word, pronunciation) tuples

        Examples:
            >>> g2p.pronounce_batch(['hello', 'world'])
            [('hello', ['HH', 'AH', 'L', 'OW']), ('world', ['W', 'ER', 'L', 'D'])]
        """
        return [(word, self.pronounce(word)) for word in words]

    def pronounce_text(self, text: str) -> list[tuple[str, list[str]]]:
        """
        Pronounce all words in a text string.

        Args:
            text: Text containing multiple words

        Returns:
            List of (word, pronunciation) tuples
        """
        words = text.split()
        return self.pronounce_batch(words)

    def pronounce_with_confidence(self, word: str) -> tuple[list[str], list[float]]:
        """
        Get pronunciation with per-phoneme confidence scores.

        Args:
            word: Word to pronounce

        Returns:
            Tuple of (phonemes, confidences)

        Examples:
            >>> g2p = G2P(model='models/en_US_nostress.g2p.gz')
            >>> phonemes, confidences = g2p.pronounce_with_confidence("hello")
            >>> # (['HH', 'AH', 'L', 'OW'], [0.95, 0.82, 0.98, 0.87])
        """
        return self._dt.pronounce_with_confidence(word)

    def pronounce_nbest(
        self, word: str, n: int = DEFAULT_NBEST_COUNT
    ) -> list[tuple[list[str], float]]:
        """
        Get n-best pronunciations with scores.

        Args:
            word: Word to pronounce
            n: Number of alternatives to return

        Returns:
            List of (phonemes, score) tuples, sorted by score descending

        Examples:
            >>> g2p = G2P(model='models/en_US_nostress.g2p.gz')
            >>> results = g2p.pronounce_nbest("read", n=3)
            >>> # [(['R', 'IY', 'D'], 0.65), (['R', 'EH', 'D'], 0.35), ...]
        """
        return self._dt.pronounce_nbest(word, n)

    @classmethod
    def train(
        cls,
        dictionary: str | Path,
        locale: str = DEFAULT_LOCALE,
        phoneset: str = DEFAULT_PHONESET,
        remove_stress: bool = False,
        output: str | Path | None = None,
        prune: bool = False,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        **kwargs,
    ) -> G2P:
        """
        Train a new G2P model from a dictionary.

        Args:
            dictionary: Path to dictionary file
            locale: Language locale
            phoneset: Phoneset name
            remove_stress: Remove stress markers
            output: Where to save the model (optional)
            prune: Post-prune the trained tree on the validation split
            validation_split: Fraction held out for pruning (e.g. 0.05)
            test_split: Fraction held out for held-out test evaluation
            **kwargs: Additional DecisionTree parameters

        Returns:
            Trained G2P instance

        Examples:
            g2p = G2P.train('mydict.txt', locale='en_US', remove_stress=True)
            g2p = G2P.train(
                'mydict.txt', locale='en_US',
                prune=True, validation_split=0.05,
            )
        """
        dt = DecisionTree(
            locale=locale, phoneset_name=phoneset, remove_stress=remove_stress, **kwargs
        )
        dt.train_from_dict(
            str(dictionary),
            encoding=DICT_ENCODING,
            validation_split=validation_split,
            test_split=test_split,
            prune=prune,
        )

        if output:
            dt.export(str(output))

        instance = cls(locale=locale, phoneset=phoneset, remove_stress=remove_stress)
        instance._dt = dt
        return instance

    @classmethod
    def from_lang(
        cls,
        language: str,
        *,
        search_paths: list[str | Path] | None = None,
        phoneset: str = DEFAULT_MULTIGRAM_PHONESET,
    ) -> G2P:
        """Load a pre-trained model for ``language`` from a conventional location.

        Resolves the BCP47-ish ``language`` argument (e.g. ``"fr_FR"``,
        ``"fr-fr"``, ``"fr"``) to a model file by searching, in order:

        1. Each directory in ``search_paths`` (if provided);
        2. ``$PHONEBOX_MODELS`` if the env var is set (colon-separated);
        3. ``./build/g2p/<lang-hyphen>/<lang-hyphen>-<phoneset>.g2p.gz``
           (a common build-artifact layout for downstream projects);
        4. ``./models/<lang-hyphen>-<phoneset>.g2p.gz``;
        5. The packaged ``phonebox/models/`` directory.

        The filename probed is ``<lang-hyphen>-<phoneset>.g2p.gz`` —
        e.g. for ``fr_FR`` + ``ipa`` we look for ``fr-fr-ipa.g2p.gz``.

        Raises FileNotFoundError if nothing matches; the error message
        lists every path that was tried, so it's obvious where to drop
        a model if you have one elsewhere.

        Examples::

            g2p = G2P.from_lang("fr_FR")
            g2p = G2P.from_lang("es-mx")
            g2p = G2P.from_lang("en", phoneset="cmu")
        """
        import os

        from .locales import canonical_locale

        canon = canonical_locale(language)  # e.g. "fr_FR"

        # Build hyphenated forms to try. If the user gave just a language
        # code ("it", "fr"), also probe the conventional "xx-xx" filename
        # so that e.g. `from_lang("it")` finds it-it-ipa.g2p.gz.
        primary = canon.replace("_", "-").lower()  # "fr-fr" or "it"
        forms = [primary]
        if "_" not in canon and "-" not in canon:
            forms.append(f"{primary}-{primary}")  # "it-it"

        def names(form: str) -> str:
            return f"{form}-{phoneset}.g2p.gz"

        candidates: list[Path] = []
        for d in search_paths or []:
            for form in forms:
                candidates.append(Path(d) / names(form))
                candidates.append(Path(d) / form / names(form))
        env = os.environ.get("PHONEBOX_MODELS")
        if env:
            for d in env.split(os.pathsep):
                for form in forms:
                    candidates.append(Path(d) / names(form))
                    candidates.append(Path(d) / form / names(form))
        for form in forms:
            candidates.append(Path("build/g2p") / form / names(form))
            candidates.append(Path("models") / names(form))
            candidates.append(Path(__file__).parent / "models" / names(form))

        for path in candidates:
            if path.is_file():
                return cls(model=str(path))

        tried = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"No phonebox model for language {language!r} (canon={canon!r}, "
            f"phoneset={phoneset!r}). Tried:\n  {tried}\n"
            f"Set $PHONEBOX_MODELS or pass search_paths=[...] to add "
            f"locations."
        )

    @classmethod
    def from_pocketsphinx(cls, data_dir: Path = Path("data")) -> G2P:
        """
        Quick setup for PocketSphinx (fetch CMUdict, process, train).

        Args:
            data_dir: Base data directory

        Returns:
            Trained G2P instance

        Examples:
            g2p = G2P.from_pocketsphinx()
            g2p.pronounce('hello')
        """
        from .dictionary import Dictionary

        # Fetch and process CMUdict
        cmudict = Dictionary.fetch("cmudict", data_dir=data_dir)
        processed = cmudict.process(remove_stress=True)

        # Train model
        dt = processed.train_g2p_model(locale=DEFAULT_LOCALE, remove_stress=True)

        instance = cls(
            locale=DEFAULT_LOCALE, phoneset=DEFAULT_PHONESET, remove_stress=True
        )
        instance._dt = dt
        return instance

    def save(self, path: str | Path) -> None:
        """Save the trained model to a file."""
        self._dt.export(str(path))

    @property
    def has_distributions(self) -> bool:
        """Check if model has probability distributions at leaves."""
        return self._dt.store_distributions

    def score_pronunciation(
        self, word: str, phones: list[str], method: str = "geometric"
    ) -> float:
        """
        Score a specific pronunciation using the model's probability distributions.

        Args:
            word: Word to score
            phones: Target phoneme sequence
            method: How to combine per-phone scores:
                - "geometric" (default): geometric mean, length-normalized
                - "product": raw product of probabilities
                - "arithmetic": arithmetic mean
                - "min": minimum (weakest link)
                - "harmonic": harmonic mean

        Returns:
            Overall score (0.0 to 1.0), higher is better.

        Examples:
            >>> g2p = G2P(model='models/en_US_nostress.g2p.gz')
            >>> score = g2p.score_pronunciation("READ", ["R", "IY", "D"])
            >>> print(f"Score: {score:.3f}")
        """
        return self._dt.score_pronunciation(word, phones, method=method)

    def __repr__(self) -> str:
        return f"G2P(locale={self.locale}, phoneset={self.phoneset})"

    def __call__(self, word: str) -> list[str]:
        """Allow instance to be called directly: g2p('word')"""
        return self.pronounce(word)
