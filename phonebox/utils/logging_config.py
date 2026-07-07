#!/usr/bin/env python
"""
Logging configuration for g2p.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
