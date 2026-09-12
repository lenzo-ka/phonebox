#!/usr/bin/env python
"""Delegate lexicon review to the installed CLI's shared workflow.

Example: python examples/score_dictionary.py words.dict -m model.g2p.gz
"""

import sys

from phonebox.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["dict", "review", *sys.argv[1:]]))
