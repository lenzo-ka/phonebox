# Phonebox Quick Start Guide

## Installation

Install in a Python environment supported by the runtime dependencies:

```bash
python -m pip install phonebox
```

Full-package platform support depends on ICU backend wheel availability; see
[installation scope](docs/RELEASING.md#installation-scope). YAML configs need
`phonebox[config]`; opting into scikit-learn needs `phonebox[sklearn]`.
The native workflows below need neither extra. Fetching CMUdict needs network
access; retain its downloaded `LICENSE` when redistributing dictionary data.
See [the task guide index](docs/README.md) for library/API and CLI workflows.

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

# Keep primary and secondary stress (unstressed 0 markers are removed)
phonebox recipe cmudict tts -o g2p.py --keep-secondary
```

## Using the Bundled G2P

Output comments below are illustrative, not fixed results. Pronunciations depend
on the trained model, preset and saved stress policy. The generated Python file
needs only the standard library; training and full-library inference still need
the installed runtime.

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

# Bundle an existing decision-tree model
phonebox bundle model.g2p.gz -o g2p.py
```

Standalone bundling supports decision-tree models; use `phonebox pronounce` or
the full library for multigram models.

## Step-by-Step Training (for debugging)

```bash
# 1. Fetch dictionary
phonebox dict fetch cmudict --data-dir data

# 2. Align letters to phonemes
phonebox align data/cmudict/cmudict.dict \
  -o alignments.txt \
  --locale en_US --phoneset cmu --width 7 \
  --remove-stress

# 3. Vectorize alignments
phonebox vectorize alignments.txt \
  -o vectors.txt \
  --locale en_US --phoneset cmu --width 7 --remove-stress

# 4. Train from vectors
phonebox model train en_US \
  --vectors vectors.txt \
  --trainer native --phoneset cmu --width 7 --remove-stress \
  -o model.g2p.gz

# 5. Bundle
phonebox bundle model.g2p.gz -o g2p.py
```

Prepared alignments/vectors do not carry the primary workflow's complete
configuration. Keep locale, phoneset, stress setting, context width and target
column order consistent across stages; this example uses stressless CMU,
width 7, and the default final target column. Prepared model training is
unpruned by default; primary `phonebox train` prunes by default. For the usual
dictionary-to-model workflow, prefer `phonebox train`.

## Python API

```python
from phonebox import G2P

# Load model
g2p = G2P(model="model.g2p.gz")

# Get pronunciation
phones = g2p.pronounce("hello")
print(phones)  # ['HH', 'AH', 'L', 'OW']

# N-best tree alternatives (default dictionary fallback can return one known variant)
g2p.use_dict_fallback = False
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
