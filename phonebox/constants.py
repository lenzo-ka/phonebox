"""
Shared constants for g2p package.

This module defines all magic numbers and default values used throughout g2p.
Centralizing these makes the code more maintainable and self-documenting.
"""

from __future__ import annotations

# =============================================================================
# Context Window Configuration
# =============================================================================

CONTEXT_WINDOW_SIZE = 7
"""
Size of context window for feature extraction.

A 7-gram window captures:
  - 3 letters to the left
  - Current letter
  - 3 letters to the right

Example: For 'l' in "hello":
  Context: [h, e, l, l, o, •, •]
           L3 L2 L1 C  R1 R2 R3
"""


# =============================================================================
# Default Locale and Phoneset
# =============================================================================

DEFAULT_LOCALE = "en_US"
"""Default language locale."""

DEFAULT_PHONESET = "cmu"
"""
Default phoneset name for the CART (decision-tree) pipeline.

The phoneset name is an arbitrary tag that affects only:
  - which key the vectorizer looks up in the locale's ``join`` dict
  - which stress-stripping regex is used when ``remove_stress=True``
    (see ``STRESS_STRIPPERS`` in ``core.vectorizer``)
  - whether accents are auto-folded (``cmu`` does, others don't)

Well-known names with stress-stripping support: ``"cmu"`` (CMU Arpabet,
strips 0/1/2 digits) and ``"xsampa"`` (X-SAMPA, strips ``"`` and ``%``).
Other names (e.g. ``"ipa"``) are accepted; stress stripping becomes a
no-op for them.
"""

DEFAULT_MULTIGRAM_PHONESET = "ipa"
"""
Default phoneset name for the multigram / joint-decode pipeline.

The multigram tools (``pronounce``, ``train-multigram``, ``compare``) work
over IPA lexicons, whereas the CART pipeline (see ``DEFAULT_PHONESET``)
defaults to CMU Arpabet. Kept as two named constants so the split is
explicit rather than a pair of drifting string literals.
"""

DEFAULT_CASED = False
"""
Default: case-insensitive (lowercase all letters).

Case-insensitive is typical for G2P since pronunciation rarely depends on case.
Set to True for languages where case affects pronunciation.
"""


# =============================================================================
# EM Alignment Defaults
# =============================================================================

DEFAULT_MAX_ITERATIONS = 100
"""
Default maximum iterations for EM alignment.

EM typically converges in 10-30 iterations, but we allow 100
to handle difficult cases.
"""

DEFAULT_MIN_CHANGE_RATIO = 0.0
"""
Default minimum change ratio for EM convergence.

Stop when changes drop below this threshold. 0.0 means run until
no changes or max iterations reached.
"""

DEFAULT_MAX_COMBINATIONS = 10000
"""
Default maximum alignment combinations per word.

EMAlign.initialize() enumerates and stores every possible letter→phone
alignment per training entry. For a word with N letters and K phones,
there are C(N, N-K) alignments. With no cap, this is O(memory) bomb on
languages where many words have long silent-letter runs or compound
morphology (French was the canary — 428k entries × ~500 combinations
average = 200+ GB RSS).

10000 keeps memory bounded while still admitting >99% of entries in
every language we've trained so far (English, Spanish, French, German,
Italian, Portuguese). Words whose combinatorial space exceeds this are
silently skipped — they're almost always rare compound forms whose base
forms remain in the training set.

Set to 0 to disable the cap (use only if you've checked combinatorial
estimates with `EMAlign.estimate_init_combinations(lexicon)` first).
"""

INIT_DATA_MEMORY_WARN_BYTES = 8 * 1024 * 1024 * 1024
"""
Estimated `init_data` size at which EMAlign emits a memory-pressure
warning. ~50 bytes per alignment list times C(N, K) per entry — beyond
this, the user almost certainly wants a tighter `max_combinations` cap.
"""

DICT_HASH_LENGTH = 8
"""Number of hex characters to keep from dictionary content hash."""

LENGTH_PENALTY_WEIGHT = 0.5
"""Weight for length mismatch penalty when scoring candidate pronunciations."""

EPSILON_PROBABILITY_BOOST = 100
"""
Probability boost for epsilon (silent) phoneme assignments.

Used in EM alignment to bias non-letter characters (punctuation, etc.)
toward producing no sound. Higher values make it more likely that
apostrophes, hyphens, and other non-letters map to epsilon.
"""

PARALLEL_ENTRY_THRESHOLD = 1000
"""
Minimum entries before using parallel alignment.

Below this threshold, serial processing is faster due to multiprocessing overhead.
"""

MIN_CHUNK_SIZE = 100
"""Minimum chunk size for parallel processing."""

CHUNKS_PER_WORKER = 4
"""Number of chunks per worker for load balancing."""

DEFAULT_MULTIGRAM_MIN_UNIT_MASS = 1e-8
"""
Drop joint units below this probability after each multigram EM M-step.

Keeps the unit table sparse on large lexicons without affecting small
toy training sets (uniform init mass is typically much larger).
"""


# =============================================================================
# Special Characters
# =============================================================================

AETHER = "•"
"""
Padding character for context windows.

Unicode: U+2022 (BULLET)
Used to pad left/right edges when no actual letter exists.
"""

EPSILON = "∅"
"""
Empty phoneme marker.

Unicode: U+2205 (EMPTY SET)
Represents letters that produce no sound (silent letters).
"""

JOIN_CHAR = "₊"
"""
Phoneme joiner character.

Unicode: U+208A (SUBSCRIPT PLUS SIGN)
Used to join multiple phonemes that map to single letter.
Example: 'x' → K₊S (two phonemes joined)
"""


# =============================================================================
# Decision Tree Defaults
# =============================================================================

DEFAULT_MIN_SAMPLES_SPLIT = 2
"""Minimum samples required to split a node."""

DEFAULT_MIN_SAMPLES_LEAF = 1
"""Minimum samples required in a leaf node."""

DEFAULT_STORE_DISTRIBUTIONS = True
"""Whether to store probability distributions at leaves (for n-best)."""

DEFAULT_MIN_DIST_ENTROPY = 0.1
"""Minimum entropy threshold for storing distributions."""

DEFAULT_NBEST_COUNT = 5
"""Default number of n-best pronunciation alternatives to return."""

NBEST_TOP_K_PER_POSITION = 3
"""Top k alternatives per position to keep n-best generation tractable."""

DOWNLOAD_TIMEOUT_SECONDS = 30
"""Timeout in seconds for downloading dictionary files from URLs."""


FILE_ENCODING = "utf-8"
"""Default encoding for text/JSON/model files (non-dictionary)."""

DICT_ENCODING = "utf-8"
"""Default encoding for pronunciation dictionary files."""


# =============================================================================
# Evaluation Split Defaults
# =============================================================================

DEFAULT_SPLIT_SEED = 42
"""Default RNG seed for shuffling a lexicon into train/test splits."""

DEFAULT_TEST_FRACTION = 0.10
"""Default held-out fraction when splitting a lexicon for evaluation."""

DEFAULT_MAX_TEST_ENTRIES = 2000
"""Default cap on held-out test entries (keeps eval runtime bounded)."""
