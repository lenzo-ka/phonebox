#!/usr/bin/env python
"""Train a model with narrow context window."""

import sys

from phonebox.core.g2p_model import G2PDecisionTree

dt = G2PDecisionTree(
    locale="en_US",
    phoneset_name="cmu",
    remove_stress=True,
    store_distributions=True,
    verbose=False,
    width=3,  # Only 1 letter each side (default is 7)
)

print("Loading...", file=sys.stderr)
with open("data/cmudict/cmudict.dict") as f:
    dt.load_prondict(f)

print("Aligning...", file=sys.stderr)
dt.align()

print("Training...", file=sys.stderr)
dt.train()

print("Exporting...", file=sys.stderr)
dt.export("models/cmudict_narrow.g2p.gz")
print("Done: models/cmudict_narrow.g2p.gz", file=sys.stderr)
