"""
Phonebox: grapheme-to-phoneme conversion package.

A fast and accurate grapheme-to-phoneme conversion system using
decision trees, EM alignment, and optional joint multigram (n:m) models.
"""

from __future__ import annotations

__version__ = "0.1.0"
__package_name__ = "phonebox"

# High-level API (recommended for most users)
from .converter import G2P

# Core classes (for advanced use)
from .core.decision_tree import DecisionTree
from .core.em_align import EMAlign
from .core.multigram_g2p import MultigramG2P
from .core.vectorizer import Vectorizer
from .dictionary import Dictionary
from .locales import (
    canonical_locale,
    load_locale_defaults,
    supplement_defaults,
)

__all__ = [
    # High-level API
    "G2P",
    "Dictionary",
    # Core API
    "DecisionTree",
    "EMAlign",
    "MultigramG2P",
    "Vectorizer",
    # Locale-aware defaults
    "canonical_locale",
    "load_locale_defaults",
    "supplement_defaults",
]
