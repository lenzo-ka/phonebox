"""
Core phonebox functionality.

This module contains the core components for grapheme-to-phoneme conversion:
- EMAlign: EM-based letter-to-phoneme alignment (1:1)
- MultigramAligner / MultigramG2P: joint n:m EM + Viterbi decode
- Vectorizer: Feature extraction for context windows
- DecisionTree: G2P decision tree for pronunciation prediction
"""

from __future__ import annotations
