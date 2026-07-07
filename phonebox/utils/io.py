"""Shared I/O utilities for CLI tools and scripts."""

from __future__ import annotations

import contextlib
import sys

from ..constants import FILE_ENCODING


@contextlib.contextmanager
def open_output(path: str | None):
    """Context manager: open *path* for writing, or yield stdout if *path* is None."""
    if path:
        with open(path, "w", encoding=FILE_ENCODING) as f:
            yield f
    else:
        yield sys.stdout


def is_dict_comment(line: str) -> bool:
    """Return True if *line* is empty or a dictionary comment (# or ;;;)."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#") or stripped.startswith(";;;")
