#!/usr/bin/env python
"""Re-export G2PDecisionTree as DecisionTree for a simpler public API."""

from __future__ import annotations

from .g2p_model import G2PDecisionTree as DecisionTree

__all__ = ["DecisionTree"]
