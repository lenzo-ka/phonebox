# Phonebox: Grapheme-to-Phoneme Conversion

Fast, lightweight grapheme-to-phoneme (G2P) conversion using decision trees and
EM alignment. The package also includes **MultigramG2P** (n:m joint Viterbi) in
the library; 1:1 CLI commands use the decision tree path.

Decision-tree G2P is fast, compact, and interpretable; it trades some accuracy
for those properties, and neural G2P methods will generally score better.
CMUdict / PocketSphinx workflows are the main English use case; measured phone
error rates and IPA locale benchmarks are in
[`docs/G2P_EVAL.md`](docs/G2P_EVAL.md).

Phonebox is an alpha `0.x` library. Each `0.X.0` release may make breaking
changes to Python APIs, CLI commands, model tooling, and repository workflows;
treat it like a new major version when upgrading. Patch releases within one
`0.X` line are intended to remain compatible. Pin the minor release when a
stable integration surface is required.

## Features

- **Measured accuracy**: reported phone error rates on CMUdict (see docs/BENCHMARKS.md); neural G2P methods can be more accurate
- **Compact Models**: < 1MB model size for typical 1:1 trees
- **MultigramG2P**: n:m joint EM + Viterbi decode (`MultigramG2P` in Python API)
- **Zero Dependencies**: Bundled Python executable works standalone
- **CMUdict Support**: English pronunciation with PocketSphinx compatibility

## Installation

```bash
pip install phonebox
```

Or from source:

```bash
git clone https://github.com/lenzo-ka/phonebox.git
cd phonebox
pip install -e .
```

## Quick Start

### One Command

```bash
# Build G2P from CMUdict, bundle as Python executable
phonebox recipe cmudict pocketsphinx -o g2p.py

# Use it
python g2p.py "Hello, world!"
```

### TTS Preset (keeps stress)

```bash
phonebox recipe cmudict tts -o g2p.py
```

## Using Bundled G2P

### Command Line

```bash
python g2p.py "Hello, world!"
# hello   HH AH L OW
# world   W ER L D

# Raw mode (no text normalization)
python g2p.py -r "Hello,"
```

### Python Library

```python
from g2p import G2PPredictor

g2p = G2PPredictor.from_embedded()
phones = g2p.pronounce("hello")  # ['HH', 'AH', 'L', 'OW']

# Process text (tokenizes automatically)
for word, phones in g2p.pronounce_text("Hello, world!"):
    print(f"{word}: {' '.join(phones)}")
```

## CLI Commands

```
Quick Start:
  recipe       Build complete G2P from dictionary (one command)

Using Models:
  pronounce    Get pronunciations for words
  normalize    Preview model-independent text tokenization
  bundle       Bundle a decision-tree model as a standalone executable

Building Models:
  train        Train a 1:1 decision-tree model with safe defaults
  train-multigram  Train/export an n:m MultigramG2P model
  model        Model operations (build, train, convert, benchmark)
  dict         Dictionary operations (fetch, export-vectors)

Low-Level:
  align        Align letters to phonemes (EM algorithm)
  vectorize    Convert alignments to feature vectors

Quality:
  check          Validate lexicon against phoneset
  suggest-joins  Discover join candidates (multigram EM)
  compare        1:1 vs n:m eval (locale or all IPA locales)
```

### G2P evaluation (IPA locales)

```bash
phonebox compare all                    # docs/G2P_COMPARE.md
phonebox compare locale --lexicon … --locale it_IT
phonebox train-multigram --locale it_IT --lexicon it_ipa.tsv -o model.g2p.gz
```

Use `phonebox compare all`, `phonebox compare sweep`, and
`phonebox compare units` for evaluation workflows. The same structured
operations are importable from `phonebox.eval`.
See [`docs/G2P_EVAL.md`](docs/G2P_EVAL.md).

### Examples

```bash
# Preview model-independent text tokenization
phonebox normalize "Hello, world!"

# Pronounce with existing model
phonebox pronounce hello world -m model.g2p.gz

# Benchmark model
phonebox model benchmark model.g2p.gz

# Fetch dictionary manually
phonebox dict fetch cmudict
```

## Python API

```python
from phonebox import G2P, MultigramG2P, train_g2p

# Train/export a 1:1 model. Defaults: IPA, native serial training, pruning with
# a 5% validation split, stored leaf distributions, and preserved stress.
training = train_g2p("dictionary.tsv", locale="en", output="model.g2p.gz")
# Use phoneset="cmu", remove_stress=True when CMU stress digits should be removed.

# 1:1 decision tree (CLI: phonebox pronounce)
g2p = G2P(model="model.g2p.gz")
phones = g2p.pronounce("hello")
print(phones)  # ['HH', 'AH', 'L', 'OW']
items = g2p.pronounce_text("Hello, world!")
raw_items = g2p.pronounce_text("Hello, world!", raw=True)

# n:m multigram: the vectorizer is saved with the model and reused by pronounce()
from phonebox.core.vectorizer import Vectorizer
vec = Vectorizer(locale="en_US", phoneset_name="cmu")
mg = MultigramG2P(max_letter_span=2, max_phone_span=2, preprocessor=vec)
mg.train_from_dict("lexicon.tsv")
# phonebox train-multigram … ; phonebox pronounce -m model.g2p.gz (sidecar auto-detect)

# N-best alternatives
for pron, score in g2p.pronounce_nbest("read", n=3):
    print(f"{pron} ({score:.3f})")
```

## Step-by-Step Training

The primary CLI applies the same defaults as `train_g2p` and `G2P.train`:

```bash
phonebox train --locale en --lexicon dictionary.tsv -o model.g2p.gz
```

For prepared-input or debugging workflows:

```bash
# 1. Fetch dictionary
phonebox dict fetch cmudict

# 2. Align letters to phonemes
phonebox align data/cmudict/cmudict.dict -o alignments.txt --remove-stress

# 3. Vectorize alignments
phonebox vectorize alignments.txt -o vectors.txt

# 4. Train from vectors
phonebox model train en_US --vectors vectors.txt -o model.g2p.gz

# 5. Bundle
phonebox bundle model.g2p.gz -o g2p.py
```

## Standalone Deployment

Bundled files have zero dependencies beyond the Python standard library:

```bash
phonebox bundle model.g2p.gz -o g2p.py
python g2p.py "test"
```

Locale arguments accept bare language tags and case-insensitive hyphenated or
underscored forms. See [Locale exemplar inventories](docs/EXEMPLARS.md) for the
exact naming and resolution contract, generated ICU/CLDR data pipeline, and
orthography-only scope.

## Algorithm

1. **EM Alignment**: Expectation-Maximization aligns letters to phonemes
2. **Feature Extraction**: 7-gram letter windows create feature vectors
3. **Decision Tree**: ID3-style tree trained on aligned data
4. **Prediction**: Tree traversal based on letter context

Based on research from CMU:
- [CMU G2P Research](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/lenzo/html/areas/t2p/)
- [ICSLP 1998 Paper](https://www.isca-speech.org/archive/icslp_1998/i98_0561.html)

## License

Phonebox source code is available under the [BSD 2-Clause License](LICENSE).
The generated ICU/CLDR locale artifact is available under the [Unicode License
v3](LICENSE-UNICODE). See [Third-party notices](THIRD_PARTY_NOTICES.md) for its
source, modifications, and versioned provenance.

## Author

Kevin Lenzo ([@lenzo-ka](https://github.com/lenzo-ka))

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
