#!/usr/bin/env python
"""
G2P-specific wrapper around generic CART decision tree.

This module adds g2p-specific functionality on top of the generic
decision tree implementation:
- EM alignment for letter-to-phoneme mapping
- Vectorization with context windows
- Pronunciation dictionary loading
- Exception dictionary handling
- G2P-specific scoring and n-best generation
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any, cast

from cartlet import (
    CRITERION_ENTROPY,
    PROB_HIGH_CONFIDENCE,
)
from cartlet import DecisionTree as GenericDecisionTree

from phonebox.utils.logging_config import get_logger

from ..constants import (
    DEFAULT_CASED,
    DEFAULT_MIN_DIST_ENTROPY,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_NBEST_COUNT,
    DEFAULT_STORE_DISTRIBUTIONS,
    DICT_ENCODING,
    DICT_HASH_LENGTH,
    FILE_ENCODING,
    LENGTH_PENALTY_WEIGHT,
)
from ..utils.io import is_dict_comment
from .em_align import EMAlign
from .nbest import generate_nbest
from .vectorizer import Vectorizer, make_join_re

logger = get_logger(__name__)


class G2PDecisionTree:
    """
    G2P-specific decision tree that wraps the generic CART implementation.

    This class adds:
    - Pronunciation dictionary loading and alignment
    - Feature vectorization for letter contexts
    - Exception dictionary for hybrid lookup
    - G2P-specific scoring and evaluation
    """

    VERSION = "1.2"
    _G2P_EXT = ".g2p"

    @classmethod
    def _format_for(cls, path: str) -> str | None:
        """Return cartlet's `format=` override for phonebox's .g2p suffix.

        cartlet auto-detects format from the extension; .g2p is phonebox's own
        suffix that maps to cartlet's JSONL wire format. Returns None for
        cartlet-native suffixes so the loader/exporter handles them normally.
        """
        return "jsonl" if cls._G2P_EXT in path else None

    def __init__(
        self,
        model=None,
        locale=None,
        phoneset_name=None,
        remove_stress=False,
        remove_accents=None,
        cased=DEFAULT_CASED,
        verbose=False,
        max_iterations=None,
        max_combinations=None,
        min_change_ratio=None,
        min_samples_split=None,
        min_samples_leaf=None,
        norm_xlit=False,
        filter_non_letters=False,
        exceptions_dict=None,
        use_dict_fallback=True,
        store_distributions=True,
        min_dist_entropy=DEFAULT_MIN_DIST_ENTROPY,
        min_confidence=PROB_HIGH_CONFIDENCE,
        criterion=CRITERION_ENTROPY,
        parallel_align=True,
        trainer="native",
        width=None,
    ) -> None:
        """
        Initialize G2P decision tree.

        Args:
            model: Path to pre-trained model
            locale: Language locale
            phoneset_name: Phoneset tag (e.g. 'cmu', 'xsampa', 'ipa'); see
                ``DEFAULT_PHONESET`` for the role this plays.
            remove_stress: Remove stress markers
            cased: Case-sensitive
            verbose: Verbose output
            max_iterations: Max EM iterations
            max_combinations: Max alignment combinations
            min_change_ratio: Min change ratio for EM
            min_samples_split: Min samples to split node
            min_samples_leaf: Min samples in leaf
            norm_xlit: Use normalizing transliteration
            filter_non_letters: Keep only letters, hyphens, apostrophes, periods
            exceptions_dict: Path to exceptions dictionary
            use_dict_fallback: Use exceptions dictionary
            store_distributions: Store probability distributions
            min_dist_entropy: Min entropy for distributions
            min_confidence: Confidence threshold for storing distributions
                vs. just the best class (default 0.95)
            criterion: Split criterion ("entropy" or "gini")
            parallel_align: Parallel alignment
            trainer: Trainer backend ("native" or "sklearn")
            width: Context window width (default 7, must be odd)
        """
        self.verbose = verbose

        # EM alignment parameters
        self.max_iterations = max_iterations
        self.max_combinations = max_combinations
        self.min_change_ratio = min_change_ratio
        self.parallel_align = parallel_align

        # Exception dictionary for hybrid lookup
        self.exceptions: dict[str, list[str]] = {}
        self.use_dict_fallback = use_dict_fallback

        # Dictionary hash for versioning
        self.dict_hash: str | None = None

        # Vectorizer for feature extraction
        # For CMU/PocketSphinx: default to removing accents (unless explicitly set)
        # For other phonesets: default to keeping accents
        if remove_accents is None:
            remove_accents = phoneset_name == "cmu"

        vectorizer_kwargs = {
            "locale": locale,
            "phoneset_name": phoneset_name,
            "remove_stress": remove_stress,
            "remove_accents": remove_accents,
            "filter_non_letters": filter_non_letters,
            "cased": cased,
            "verbose": self.verbose,
            "norm_xlit": norm_xlit,
        }
        if width is not None:
            vectorizer_kwargs["width"] = width
        self.vectorizer = Vectorizer(**vectorizer_kwargs)

        # Store trainer for later use in train()
        self.trainer = trainer

        # Generic decision tree (feature_names excludes target column P)
        self._cart = GenericDecisionTree(
            feature_names=self.vectorizer.feature_names,
            min_samples_split=min_samples_split
            if min_samples_split is not None
            else DEFAULT_MIN_SAMPLES_SPLIT,
            min_samples_leaf=min_samples_leaf
            if min_samples_leaf is not None
            else DEFAULT_MIN_SAMPLES_LEAF,
            store_distributions=store_distributions
            if store_distributions is not None
            else DEFAULT_STORE_DISTRIBUTIONS,
            min_dist_entropy=min_dist_entropy,
            min_confidence=min_confidence,
            criterion=criterion,
            verbose=verbose,
            logger=logger,
        )

        # EM aligner (created when loading pronunciation dictionary)
        self.em: EMAlign | None = None

        # Load model if provided
        if model:
            self.load_model(model)
            if exceptions_dict and self.use_dict_fallback and not self.exceptions:
                self.load_exceptions(exceptions_dict)

    @property
    def model(self):
        """Access to underlying CART model."""
        return self._cart.model

    @model.setter
    def model(self, value):
        """Set underlying CART model."""
        self._cart.model = value

    @property
    def store_distributions(self):
        """Access to underlying CART store_distributions setting."""
        return self._cart.store_distributions

    @store_distributions.setter
    def store_distributions(self, value):
        """Set underlying CART store_distributions setting."""
        self._cart.store_distributions = value

    @property
    def _center_position(self) -> int:
        """Center position in context window, computed from vectorizer width."""
        return (self.vectorizer.width - 1) // 2

    def load_prondict(self, infile) -> None:
        """
        Load a pronunciation dictionary and compute its version identifier.

        Args:
            infile: File handle or iterable of dictionary lines
        """
        # Read all lines and compute content hash
        lines = []
        hasher = hashlib.sha256()

        for line in infile:
            lines.append(line)
            hasher.update(line.strip().encode(DICT_ENCODING))

        self.dict_hash = hasher.hexdigest()[:DICT_HASH_LENGTH]

        if self.verbose:
            logger.info("Dictionary version: %s", self.dict_hash)

        # Create EM aligner and feed lines
        self.em = EMAlign(
            self.vectorizer,
            max_iterations=self.max_iterations,
            max_combinations=self.max_combinations,
            min_change_ratio=self.min_change_ratio,
            verbose=self.verbose,
            parallel=self.parallel_align,
        )

        for line in lines:
            if not is_dict_comment(line):
                self.em.add_line(line)

    def align(self) -> None:
        """Align input letters and phones (calls EMAlign)."""
        self.em.align()
        self.load_alignments()

    def train_from_dict(
        self,
        dict_path: str,
        encoding: str = DICT_ENCODING,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        prune: bool = False,
    ) -> dict:
        """Load dictionary, align, and train in one call.

        Args:
            dict_path: Path to pronunciation dictionary file
            encoding: File encoding
            validation_split: Fraction held out for validation-set pruning
            test_split: Fraction held out for held-out test evaluation
            prune: Post-prune the trained tree using the validation split.
                Pruning helps avoid overfitting to idiosyncratic spellings.

        Returns:
            Training metrics dict
        """
        with open(dict_path, encoding=encoding) as f:
            self.load_prondict(f)
        self.align()
        return self.train(
            validation_split=validation_split,
            test_split=test_split,
            prune=prune,
        )

    def save_alignments(self, path: str) -> None:
        """Save alignments to file for reuse.

        Args:
            path: Output file path
        """
        if not self.em or not self.em.em_data:
            raise ValueError("No alignments to save. Call align() first.")

        with open(path, "w", encoding=FILE_ENCODING) as f:
            self.em.write(f)

        if self.verbose:
            logger.info("Saved alignments to %s", path)

    def load_alignments(self, infile=None) -> None:
        """Load letter/phone alignments from memory or file."""
        if self.verbose:
            start = time()
            logger.info("Converting alignments to vectors...")

        if infile is None and self.em:
            # From EM alignments in memory
            vectors_dict = Counter(
                vector.strip()
                for alignment in self.em.next_alignment()
                for vector in self.vectorizer.next_alignment_vector(alignment)
            )
        else:
            # From file
            vectors_dict = Counter(
                vector.strip() for vector in self.vectorizer.vectorize_file(infile)
            )

        # Convert to X, y, counts format for CART
        X = []
        y = []
        counts = []

        for vector, count in vectors_dict.items():
            if len(vector.split()) != len(self.vectorizer.default_cols):
                logger.warning("Skipping malformed vector: %s", vector)
                continue

            X_val, y_val = self.vectorizer.parse_vector(vector)
            X.append(X_val)
            y.append(y_val)
            counts.append(count)

        # Update feature names to match actual features (excluding y/phoneme column)
        if X:
            actual_feature_count = len(X[0])
            expected_feature_count = (
                len(self.vectorizer.default_cols) - 1
            )  # -1 for y column

            if actual_feature_count != expected_feature_count:
                logger.warning(
                    "Feature count mismatch: got %d, expected %d",
                    actual_feature_count,
                    expected_feature_count,
                )

            self._sync_feature_names()

        # Load into generic CART
        self._cart.load_data(X, y, counts)

        if self.verbose:
            logger.info("done. (%.1f seconds)", time() - start)
            logger.info("Loaded %d unique vectors", len(X))

    def _sync_feature_names(self) -> None:
        """Push vectorizer's feature names into the underlying CART."""
        self._cart.feature_names = self.vectorizer.feature_names
        self._cart.name_to_col = {
            name: idx for idx, name in enumerate(self._cart.feature_names)
        }

    def load_vectors_data(self, X, y, counts) -> None:
        """Load pre-parsed vector data directly into CART.

        Args:
            X: Feature vectors (list of lists)
            y: Target values (list)
            counts: Occurrence counts (list)
        """
        self._sync_feature_names()
        self._cart.load_data(X, y, counts)

        if self.verbose:
            logger.info("Loaded %d vectors", len(X))

    def train(
        self,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        prune: bool = False,
    ) -> dict[str, Any]:
        """
        Train the model.

        Args:
            validation_split: Fraction for validation (pruning)
            test_split: Fraction for test (evaluation)
            prune: Whether to prune tree

        Returns:
            Dict with accuracy metrics
        """
        if self.verbose:
            logger.debug(
                "Features(%d): %s",
                len(self._cart.feature_names),
                ", ".join(str(f) for f in self._cart.feature_names),
            )

            for which, counts in self.vectorizer.counts.items():
                if not counts:
                    continue
                logger.debug(":: %d %s", len(counts), which)
                for value, count in sorted(
                    counts.items(),
                    reverse=True,
                    key=lambda x: x[1],
                ):
                    logger.debug("%s\t%d", value, count)

        # Train generic CART
        metrics = self._cart.train(
            validation_split=validation_split,
            test_split=test_split,
            prune=prune,
            trainer=self.trainer,
        )

        # Build exceptions dictionary from all data for complete coverage
        if self.use_dict_fallback and self.em:
            if self.verbose:
                logger.info("Building exceptions dictionary...")
            self.exceptions = self.build_exceptions_dict()
            if self.verbose:
                logger.info("Found %d exception words", len(self.exceptions))

        return metrics

    def _lookup_exception(self, word: str) -> list[str] | None:
        """Look up word in exceptions dictionary, returning phones or None."""
        if not self.use_dict_fallback:
            return None
        lookup_word = word if self.vectorizer.cased else word.lower()
        return self.exceptions.get(lookup_word)

    def pronounce(self, word: str) -> list[str]:
        """
        Pronounce a word (produce phoneme sequence).

        Uses hybrid lookup:
        1. Check exceptions dictionary first (if enabled)
        2. Fall back to model prediction

        Args:
            word: Word to pronounce

        Returns:
            List of phonemes
        """
        exception = self._lookup_exception(word)
        if exception is not None:
            return exception

        return self.vectorizer.uncook(self.predict(word))

    def predict(self, word: str) -> list[str]:
        """
        Predict what each letter turns into.

        Handles OOV (out-of-vocabulary) letters by returning epsilon.
        Checks feature vocabulary from model header before calling tree.

        Args:
            word: Word to predict

        Returns:
            List of raw phonemes (may include joined phonemes and epsilon)
        """
        # Get valid letter vocabulary from model header (if available)
        letter_vocab = self._get_letter_vocabulary()

        predictions = []
        for vec in self.vectorizer.vectorize_word(word):
            # Check if center letter is in vocabulary
            if letter_vocab and not self._is_valid_vector(vec, letter_vocab):
                # OOV letter → epsilon (don't call tree)
                predictions.append(self.vectorizer.epsilon)
            else:
                # Known letter → use tree. cartlet's predict() is generic
                # (returns Any) but in classification mode it produces the
                # phone-label string; cast to keep the public list[str]
                # signature accurate.
                pred = cast(str, self._cart.predict(vec))
                predictions.append(pred)
        return predictions

    def _get_letter_vocabulary(self) -> set | None:
        """
        Get valid letter vocabulary from model header.

        Reads from cartlet's feature specifications which include
        the "values" list for categorical features.

        Returns:
            Set of valid letters, or None if not available
        """
        if not self._cart.model:
            return None

        # Get from cartlet's feature specs (JSONL format)
        if self._cart.feature_specs and self._center_position < len(
            self._cart.feature_specs
        ):
            spec = self._cart.feature_specs[self._center_position]
            if spec.values is not None:
                return spec.values if isinstance(spec.values, set) else set(spec.values)

        # Fall back: build from training data if available
        if self._cart.X:
            return {
                row[self._center_position]
                for row in self._cart.X
                if len(row) > self._center_position
            } | {self.vectorizer.aether}

        return None

    def _is_valid_vector(self, vec: list[str], letter_vocab: set) -> bool:
        """
        Check if vector's center letter is in vocabulary.

        Args:
            vec: Feature vector (context window)
            letter_vocab: Set of valid letters

        Returns:
            True if center letter is known
        """
        # vec = [L3, L2, L1, C, R1, R2, R3] for width=7
        if len(vec) <= self._center_position:
            return True  # Can't check, let tree handle

        center_letter = vec[self._center_position]
        return center_letter in letter_vocab

    def _predict_distributions(self, word: str) -> list[str | dict[str, float]]:
        """
        Get probability distributions for each phoneme position.

        Args:
            word: Word to predict

        Returns:
            List of distributions (one per letter)
        """
        dists: list[str | dict[str, float]] = []
        for vec in self.vectorizer.vectorize_word(word):
            # cartlet.predict(return_dist=True) returns dict[str, float]
            # for classification leaves; cast since the signature is generic.
            dist = cast(
                "str | dict[str, float]", self._cart.predict(vec, return_dist=True)
            )
            dists.append(dist)
        return dists

    def pronounce_with_confidence(self, word: str) -> tuple[list[str], list[float]]:
        """
        Get pronunciation with per-phoneme confidence scores.

        Args:
            word: Word to pronounce

        Returns:
            Tuple of (phonemes, confidences)
        """
        exception = self._lookup_exception(word)
        if exception is not None:
            return exception, [1.0] * len(exception)

        # Get distributions from model
        dists = self._predict_distributions(word)

        raw_phonemes = []
        raw_confidences = []

        for dist in dists:
            if isinstance(dist, dict):
                best_phone = next(iter(dist.keys()))
                raw_phonemes.append(best_phone)
                raw_confidences.append(dist[best_phone])
            elif isinstance(dist, str):
                raw_phonemes.append(dist)
                raw_confidences.append(1.0)

        # Uncook phonemes
        final_phonemes = []
        final_confidences = []

        for phone, conf in zip(raw_phonemes, raw_confidences):
            uncooked = self.vectorizer.uncook([phone])
            for p in uncooked:
                final_phonemes.append(p)
                final_confidences.append(conf)

        return final_phonemes, final_confidences

    def pronounce_nbest(
        self, word: str, n: int = DEFAULT_NBEST_COUNT
    ) -> list[tuple[list[str], float]]:
        """
        Get n-best pronunciations with scores.

        Args:
            word: Word to pronounce
            n: Number of alternatives

        Returns:
            List of (phonemes, score) tuples
        """
        exception = self._lookup_exception(word)
        if exception is not None:
            return [(exception, 1.0)]

        # Get distributions from model
        dists = self._predict_distributions(word)

        # Generate n-best from distributions
        nbest_raw = generate_nbest(dists, n)  # type: ignore[arg-type]

        # Uncook phonemes for each alternative
        nbest_cooked = []
        for phones, score in nbest_raw:
            uncooked = self.vectorizer.uncook(phones)
            nbest_cooked.append((uncooked, score))

        return nbest_cooked

    def score_pronunciation(
        self, word: str, phones: list[str], method: str = "geometric"
    ) -> float:
        """
        Score how likely a given pronunciation is for a word.

        Uses a greedy alignment: for each target phone, find the position
        with highest probability for that phone.

        Args:
            word: The word to score
            phones: List of phonemes to score
            method: How to combine per-phone scores:
                - "geometric" (default): geometric mean, length-normalized
                - "product": raw product of probabilities
                - "arithmetic": arithmetic mean
                - "min": minimum (weakest link)
                - "harmonic": harmonic mean

        Returns:
            Overall score (0.0 to 1.0, higher is better)
        """
        # Get distributions from model
        dists = self._predict_distributions(word)

        # Cook the phones to match model's internal representation
        cooked_phones = self.vectorizer.cook_phones(phones)

        # For each phone, find the best probability across all positions
        phone_scores = []

        for phone in cooked_phones:
            best_prob = 0.0
            for dist in dists:
                if isinstance(dist, dict):
                    prob = dist.get(phone, 0.0)
                    best_prob = max(best_prob, prob)
            phone_scores.append(best_prob)

        # If no scores or any zero, return 0
        if not phone_scores or any(s == 0 for s in phone_scores):
            return 0.0

        if method == "product":
            return math.prod(phone_scores)
        if method == "arithmetic":
            return sum(phone_scores) / len(phone_scores)
        if method == "min":
            return min(phone_scores)
        if method == "harmonic":
            return len(phone_scores) / sum(1 / s for s in phone_scores)
        # geometric (default)
        log_sum = sum(math.log(s) for s in phone_scores)
        return math.exp(log_sum / len(phone_scores))

    def build_exceptions_dict(self, infile=None) -> dict[str, list[str]]:
        """
        Build exceptions dictionary from pronunciation dictionary.

        Returns:
            Dict of {word: phonemes} for mispronounced words
        """
        data = self.em.init_data if (infile is None and self.em) else infile
        word_candidates: dict[str, list[list[str]]] = defaultdict(list)
        liaison_pad = self.vectorizer.liaison_pad

        # First pass: collect all pronunciations
        for line in data:
            letters, phones = (
                line
                if isinstance(line, list)
                else self.vectorizer.letters_and_phones(line)
            )
            # The EM aligner appends `liaison_pad` (currently '#' for fr_*)
            # as a sentinel letter for liaison context. It must NOT survive
            # into the exceptions-dict key, otherwise every key ends in '#'
            # and runtime `_lookup_exception(word)` never matches anything.
            if liaison_pad and letters and letters[-1] == liaison_pad:
                letters = letters[:-1]
            letters_ = "".join(self.vectorizer.uncook(letters))
            phones_ = " ".join(self.vectorizer.uncook(phones))
            word = letters_ if self.vectorizer.cased else letters_.lower()
            word_candidates[word].append(phones_.split())

        # Second pass: find words where model prediction differs
        exceptions: dict[str, list[str]] = {}

        # Temporarily disable dict fallback so predictions reflect the model
        # alone; restore it even if a prediction raises.
        old_use_dict = self.use_dict_fallback
        self.use_dict_fallback = False
        try:
            for word, pronunciations in word_candidates.items():
                predicted = self.pronounce(word)

                # Skip if model gets the only pronunciation right
                if len(pronunciations) == 1 and pronunciations[0] == predicted:
                    continue

                # Multiple pronunciations or model is wrong - find best match
                exceptions[word] = self._find_closest_pronunciation(
                    predicted, pronunciations
                )
        finally:
            self.use_dict_fallback = old_use_dict
        return exceptions

    @staticmethod
    def _find_closest_pronunciation(
        predicted: list[str],
        candidates: list[list[str]],
    ) -> list[str]:
        """
        Find the pronunciation from candidates closest to predicted.

        Args:
            predicted: Model prediction
            candidates: Candidate pronunciations

        Returns:
            Best matching pronunciation
        """
        if len(candidates) == 1:
            return candidates[0]

        best_score: float = -1.0
        best_pron = candidates[0]

        for candidate in candidates:
            # Count matching phonemes
            matches = sum(1 for p, c in zip(predicted, candidate) if p == c)
            length_penalty = abs(len(predicted) - len(candidate))
            score = float(matches) - (LENGTH_PENALTY_WEIGHT * length_penalty)

            if score > best_score:
                best_score = score
                best_pron = candidate

        return best_pron

    def load_exceptions(self, path: str) -> None:
        """
        Load exceptions dictionary from TSV file.

        Args:
            path: Path to exceptions file
        """
        path_obj = Path(path)
        if not path_obj.exists():
            if self.verbose:
                logger.warning("Exceptions file not found: %s", path)
            return

        count = 0
        with open(path_obj, encoding=DICT_ENCODING) as f:
            for line in f:
                line = line.strip()
                if is_dict_comment(line):
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    phonemes = parts[1].strip()

                    # Normalize word
                    if not self.vectorizer.cased:
                        word = word.lower()

                    self.exceptions[word] = phonemes.split()
                    count += 1

        if self.verbose:
            logger.info("Loaded %d exceptions from %s", count, path)

    def export(self, outpath: str, include_exceptions: bool = True) -> None:
        """
        Export model to file.

        Supports .g2p extension (phonebox format) which maps to cartlet's .jsonl.
        Also supports cartlet's native formats: .cart, .json, .jsonl

        Args:
            outpath: Output path (.g2p.gz, .jsonl.gz, .cart, etc.)
            include_exceptions: Include exceptions dict in metadata
        """
        # Build metadata with ACTUAL config used during training
        metadata = self.vectorizer.export_config()
        export_time = datetime.now(timezone.utc)

        # Capture complete training configuration
        metadata["training_config"] = {
            "cased": self.vectorizer.cased,
            "criterion": self._cart.criterion,
            "filter_non_letters": self.vectorizer.filter_non_letters,
            "locale": self.vectorizer.locale,
            "max_combinations": self.max_combinations,
            "max_iterations": self.max_iterations,
            "min_change_ratio": self.min_change_ratio,
            "min_confidence": self._cart.min_confidence,
            "min_samples_leaf": self._cart.min_samples_leaf,
            "min_samples_split": self._cart.min_samples_split,
            "phoneset_name": self.vectorizer.phoneset_name,
            "remove_accents": self.vectorizer.remove_accents,
            "remove_stress": self.vectorizer.remove_stress,
            "store_distributions": self._cart.store_distributions,
            "trainer": self.trainer,
            "width": self.vectorizer.width,
        }

        metadata["export_time"] = export_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        metadata["g2p_version"] = self.VERSION

        # Dictionary hash (version tracking)
        if self.dict_hash:
            metadata["dict_hash"] = self.dict_hash

        # Exceptions dictionary
        if include_exceptions and self.exceptions:
            metadata["exceptions"] = self.exceptions
            if self.verbose:
                logger.info("Embedding %d exceptions in model", len(self.exceptions))

        # Use cartlet's export with distributions if enabled.
        # The format= override lets cartlet write JSONL into our .g2p suffix
        # without a temp-file rename dance.
        self._cart.export(
            outpath,
            metadata=metadata,
            store_distributions=self._cart.store_distributions,
            format=self._format_for(outpath),
        )

        if self.verbose:
            logger.info("Model exported to %s", outpath)
            logger.info("  Training config saved in metadata for reproducibility")

    def load_model(self, path: str) -> None:
        """
        Load a pre-trained model from file.

        Supports .g2p extension (phonebox format) which is internally .jsonl.
        Also supports cartlet's native formats: .cart, .json, .jsonl

        Args:
            path: Path to model file (.g2p.gz, .jsonl.gz, .cart, etc.)
        """
        config = self._cart.load_model(path, format=self._format_for(path))

        # Store header for vocabulary access (OOV checking)
        self._model_header = config if config else {}

        # Extract g2p-specific config from metadata. Mirrors export_config():
        # every flag we save there has to be restored here, otherwise the
        # loaded model silently runs with the constructor defaults instead
        # of the values used during training.
        #
        # export() places the vectorizer state under the ``metadata`` trailer;
        # older callers passed flat configs, so look there as a fallback.
        if config:
            v = self.vectorizer
            metadata = config.get("metadata", {})

            def pick(key, default):
                if key in metadata:
                    return metadata[key]
                return config.get(key, default)

            # Re-run setup_locale so the transliterator (g2p.xlit, norm.xlit)
            # for the model's true locale gets loaded. Without this, a model
            # trained on (say) es_MX but loaded via G2P() (default locale
            # en_US) would silently keep the en_US xlit and mangle inputs.
            loaded_locale = pick("locale", v.locale)
            if loaded_locale != v.locale:
                v.setup_locale(loaded_locale)
            v.phoneset_name = pick("phoneset_name", v.phoneset_name)
            v.remove_stress = pick("remove_stress", v.remove_stress)
            v.remove_accents = pick("remove_accents", v.remove_accents)
            v.filter_non_letters = pick("filter_non_letters", v.filter_non_letters)
            v.cased = pick("cased", v.cased)
            v.aether = pick("aether", v.aether)
            v.epsilon = pick("epsilon", v.epsilon)
            v.join_char = pick("join_char", v.join_char)
            v.liaison_pad = pick("liaison_pad", v.liaison_pad)

            width = pick("width", None)
            if width:
                v.width = width
                v._pad = (width - 1) // 2
                v.padding = [v.aether] * v._pad
            v.counts = pick("counts", v.counts)

            # Rebuild letter / phone joining regexes from the saved join lists.
            # export_config() renames the phoneset-specific key to "phones".
            joinings = pick("join", {}) or {}
            letter_joins = joinings.get("letters", [])
            phone_joins = joinings.get("phones") or joinings.get(v.phoneset_name, [])
            v.lett_join_re = make_join_re(letter_joins) if letter_joins else None
            v.phon_join_re = make_join_re(phone_joins) if phone_joins else None
            # Keep vectorizer.config consistent so a re-export round-trips.
            v.config = {"join": {"letters": letter_joins, v.phoneset_name: phone_joins}}

            # Load embedded exceptions dictionary (saved under metadata).
            exceptions = pick("exceptions", None)
            if exceptions and self.use_dict_fallback:
                self.exceptions = exceptions
                if self.verbose:
                    logger.info(
                        "Loaded %d exceptions from embedded dictionary",
                        len(self.exceptions),
                    )

            # Load dictionary hash
            if "dict_hash" in config:
                self.dict_hash = config["dict_hash"]
                if self.verbose:
                    logger.info("Model trained from dictionary: %s", self.dict_hash)

    def __repr__(self) -> str:
        return (
            f"G2PDecisionTree(locale={self.vectorizer.locale}, "
            f"phoneset={self.vectorizer.phoneset_name})"
        )
