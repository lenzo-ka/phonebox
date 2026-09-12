"""
Phonebox: grapheme-to-phoneme conversion package.

A fast, lightweight grapheme-to-phoneme conversion system using
decision trees, EM alignment, and optional joint multigram (n:m) models.
"""

from __future__ import annotations

__version__ = "0.2.0"
__package_name__ = "phonebox"

# High-level API (recommended for most users)
from .converter import G2P

# Core classes (for advanced use)
from .core.decision_tree import DecisionTree
from .core.em_align import EMAlign
from .core.multigram_g2p import MultigramG2P
from .core.vectorizer import Vectorizer
from .dictionary import Dictionary
from .join_discovery import JoinCandidate, JoinDiscoveryResult, discover_joins
from .locales import (
    LocaleResolution,
    canonical_locale,
    load_locale_defaults,
    resolve_locale,
    supplement_defaults,
)
from .multigram_training import MultigramTrainingResult, train_multigram
from .pronunciation_scoring import PronunciationScore
from .training import TrainingResult, train_g2p, train_g2p_from_config

__all__ = [
    # High-level API
    "G2P",
    "PronunciationScore",
    "Dictionary",
    # Core API
    "DecisionTree",
    "EMAlign",
    "MultigramG2P",
    "Vectorizer",
    "JoinCandidate",
    "JoinDiscoveryResult",
    "discover_joins",
    "MultigramTrainingResult",
    "train_multigram",
    "TrainingResult",
    "train_g2p",
    "train_g2p_from_config",
    # Locale-aware defaults
    "LocaleResolution",
    "canonical_locale",
    "load_locale_defaults",
    "resolve_locale",
    "supplement_defaults",
]
