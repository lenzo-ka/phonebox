# Phonebox Quick Start Guide

## The Easiest Way - One Command

```bash
# Build PocketSphinx G2P from CMUdict, output as Python executable
phonebox recipe cmudict pocketsphinx -o g2p.py

# Use it immediately
python g2p.py "Hello, world!"
```

This single command:
1. Fetches CMUdict from GitHub
2. Aligns letters to phonemes
3. Trains the decision tree
4. Bundles into a standalone executable

### TTS Preset (keeps stress markers)

```bash
# Primary stress only (default for TTS)
phonebox recipe cmudict tts -o g2p.py

# With secondary stress
phonebox recipe cmudict tts -o g2p.py --keep-secondary
```

## Using the Bundled G2P

```bash
# Command line
python g2p.py "Hello, world!"
# hello   HH AH L OW
# world   W ER L D

# Python library
python -c "
from g2p import G2PPredictor
g2p = G2PPredictor.from_embedded()
print(g2p.pronounce('hello'))  # ['HH', 'AH', 'L', 'OW']
"
```

## Model Only (no bundling)

```bash
phonebox recipe cmudict pocketsphinx -o model.g2p.gz
```

## Using Existing Models

```bash
# Pronounce words
phonebox pronounce hello world -m model.g2p.gz

# Preview text normalization
phonebox normalize "Hello, world!"

# Bundle existing model
phonebox bundle model.g2p.gz -o g2p.py
```

## Step-by-Step Training (for debugging)

```bash
# 1. Fetch dictionary
phonebox dict fetch cmudict

# 2. Align letters to phonemes
phonebox align data/cmudict/cmudict.dict \
  -o alignments.txt \
  --locale en_US \
  --remove-stress

# 3. Vectorize alignments
phonebox vectorize alignments.txt \
  -o vectors.txt \
  --locale en_US

# 4. Train from vectors
phonebox model train en_US \
  --vectors vectors.txt \
  --trainer sklearn \
  -o model.g2p.gz

# 5. Bundle
phonebox bundle model.g2p.gz -o g2p.py
```

## Python API

```python
from phonebox import G2P

# Load model
g2p = G2P(model="model.g2p.gz")

# Get pronunciation
phones = g2p.pronounce("hello")
print(phones)  # ['HH', 'AH', 'L', 'OW']

# N-best alternatives
for pron, score in g2p.pronounce_nbest("read", n=3):
    print(f"{pron} ({score:.3f})")
```

## Common Tasks

```bash
# Benchmark model performance
phonebox model benchmark model.g2p.gz

# Export feature vectors (for external ML tools)
phonebox dict export-vectors cmudict.dict -o vectors.tsv
```

## Need Help?

```bash
phonebox --help
phonebox recipe --help
phonebox model --help
phonebox dict --help
```
