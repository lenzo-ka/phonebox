"""
G2P predictor template for bundling.

The g2p_predict.py template is appended to cartlet's bundled Python
predictor to create a standalone .py executable with the model embedded.

Usage:
    phonebox bundle model.g2p.gz -o g2p.py
"""

from __future__ import annotations
