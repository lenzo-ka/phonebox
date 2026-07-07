#!/usr/bin/env python
"""
Simplest possible phonebox usage.
"""

from phonebox import G2P

# Load model and use it
g2p = G2P(model="models/en_US_nostress.g2p.gz")

# Get pronunciation
print("hello:", " ".join(g2p("hello")))
print("world:", " ".join(g2p("world")))
print("python:", " ".join(g2p("python")))

# Batch mode
words = ["grapheme", "phoneme", "pronunciation"]
for word, phones in g2p.pronounce_batch(words):
    print(f"{word}: {' '.join(phones)}")
