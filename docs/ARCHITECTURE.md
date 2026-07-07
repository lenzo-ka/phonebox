# Phonebox Package Architecture

## Directory Structure

```
phonebox/
├── __init__.py                 # Package exports: G2P, Dictionary, DecisionTree
├── converter.py                # G2P class - high-level API
├── dictionary.py               # Dictionary class - dict management
│
├── core/                       # Core algorithms
│   ├── decision_tree.py        # DecisionTree alias for G2PDecisionTree
│   ├── g2p_model.py            # G2P decision tree model
│   ├── em_align.py             # 1:1 EM alignment
│   ├── multigram_align.py      # Joint n:m EM (Bisani & Ney)
│   ├── multigram_g2p.py        # MultigramG2P train/decode
│   ├── joint_decode.py         # Unit LM + Viterbi inference
│   ├── multigram_lm.py         # Add-k unit n-gram LM
│   ├── nbest.py                # N-best pronunciation generation
│   └── vectorizer.py           # Feature vectorization
│
├── experiments/                # G2P eval helpers (normalize, equiv, split)
│
├── cli/                        # Command-line tools
│   ├── main.py                 # Entry point for unified CLI
│   └── commands/               # Subcommand implementations
│       ├── align.py            # phonebox align
│       ├── bundle.py           # phonebox bundle
│       ├── check.py            # phonebox check
│       ├── dict.py             # phonebox dict (fetch, process, export-vectors...)
│       ├── model.py            # phonebox model (build, train, benchmark...)
│       ├── normalize.py        # phonebox normalize
│       ├── pronounce.py        # phonebox pronounce (1:1 only)
│       ├── recipe.py           # phonebox recipe
│       ├── suggest_joins.py    # phonebox suggest-joins (multigram EM)
│       ├── train.py            # phonebox train (1:1 safe defaults)
│       └── vectorize.py        # phonebox vectorize
│
├── config/                     # Locale configurations
│   └── locales/                # Per-locale config.json, g2p.xlit, defaults
│       ├── en_US/              # English (US)
│       ├── es_MX/ fr_FR/ de_DE/ it_IT/ pt_BR/ …
│       └── default/            # Default fallback
│
├── cart/                       # Bundled CART predictor template
│   └── g2p_predict.py          # Python predictor template
│
├── tools/                      # Standalone utilities
│   └── count_diphones.py       # Count letter/phone bigrams from EM logs
│
└── utils/                      # Core utilities
    ├── icu_utils.py            # ICU transliteration
    └── logging_config.py       # Logging setup
```

## Class Hierarchy

### High-Level API (Recommended)

```python
from phonebox import G2P, Dictionary

# G2P - simple pronunciation
g2p = G2P(model='path/to/model.g2p.gz')
phones = g2p('hello')

# Dictionary - dict management
dict = Dictionary.fetch('cmudict')
processed = dict.process(remove_stress=True)
model = processed.train_g2p_model('en_US')
```

### Core API (Advanced)

```python
from phonebox import DecisionTree, EMAlign, MultigramG2P, Vectorizer

# 1:1 decision tree
dt = DecisionTree(locale='en_US', phoneset_name='cmu')
dt.load_prondict(file)
dt.align()
dt.train()

# n:m multigram (library; see docs/G2P_EVAL.md for eval scripts)
mg = MultigramG2P(max_letter_span=2, max_phone_span=2)
mg.train_from_dict('it_ipa.tsv')
```

## Key Classes

### G2P (`converter.py`)
- High-level pronunciation interface
- Wrapper around DecisionTree
- Methods: `pronounce()`, `pronounce_batch()`, `train()`, `from_pocketsphinx()`

### Dictionary (`dictionary.py`)
- Dictionary management and processing
- Fetching from CMUdict
- Processing: stress removal, normalization
- Training integration
- Methods: `fetch()`, `process()`, `train_g2p_model()`, `list_available_languages()`

### DecisionTree (`core/decision_tree.py` / `core/g2p_model.py`)
- G2P-specific wrapper around cartlet's generic decision tree
- EM alignment, vectorization, exception dictionary, n-best
- Methods: `load_prondict()`, `align()`, `train()`, `pronounce()`, `export()`

### EMAlign (`core/em_align.py`)
- Expectation-maximization alignment
- Aligns letters to phonemes
- Used internally by DecisionTree

### Vectorizer (`core/vectorizer.py`)
- Feature extraction
- Letter context windows (7-gram)
- ICU transliteration integration
- Phoneme normalization

## CLI Commands

The CLI uses a unified `phonebox` command with subcommands:

| Command | Purpose | Example |
|---------|---------|---------|
| `phonebox recipe` | Build G2P from dictionary (one-shot) | `phonebox recipe cmudict pocketsphinx -o g2p.py` |
| `phonebox pronounce` | Get pronunciations | `phonebox pronounce hello world -m model.g2p.gz` |
| `phonebox normalize` | Preview text normalization | `phonebox normalize "Hello, world!"` |
| `phonebox bundle` | Bundle model into standalone runner | `phonebox bundle model.g2p.gz -o g2p.py` |
| `phonebox model build` | Build complete model | `phonebox model build en_US dict.txt -o model.g2p.gz` |
| `phonebox model train` | Train from aligned data | `phonebox model train en_US --alignments aligned.txt -o model.g2p.gz` |
| `phonebox model benchmark` | Benchmark performance | `phonebox model benchmark model.g2p.gz` |
| `phonebox dict fetch` | Fetch dictionaries | `phonebox dict fetch cmudict` |
| `phonebox dict export-vectors` | Export feature vectors | `phonebox dict export-vectors dict.txt -o vectors.tsv` |
| `phonebox align` | Align letters to phonemes | `phonebox align dict.txt --locale en_US` |
| `phonebox vectorize` | Convert alignments to vectors | `phonebox vectorize alignments.txt -o vectors.txt` |
| `phonebox train` | Train 1:1 G2P (safe defaults) | `phonebox train --locale en_US dict.tsv -o model.g2p.gz` |
| `phonebox train-multigram` | Train n:m MultigramG2P | `phonebox train-multigram --locale it_IT --lexicon it.tsv -o model.g2p.gz` |
| `phonebox compare locale` | Single-locale 1:1 vs n:m eval | `phonebox compare locale --lexicon fr_ipa.tsv --locale fr_FR` |
| `phonebox compare all` | Six-locale eval markdown | `phonebox compare all` |
| `phonebox check` | Validate lexicon vs phoneset | `phonebox check --lexicon lex.tsv --phoneset phones.json` |
| `phonebox suggest-joins` | Discover join candidates | `phonebox suggest-joins --locale fr_FR --lexicon fr.tsv -o joins.json` |

MultigramG2P inference: `phonebox pronounce -m model.g2p.gz` when `.units.json` sidecar
exists. See [`G2P_EVAL.md`](G2P_EVAL.md) for metrics and repo wrappers.

## Data Flow

```
Dictionary Sources
    ├── CMUdict (GitHub)
    └── Local files
         ↓
    [Dictionary.fetch() or load local]
         ↓
    [Dictionary.process()]
         ├── Remove stress
         ├── Normalize
         └── Deduplicate
         ↓
    [Dictionary.train_g2p_model()]
         ├── EMAlign.align()
         ├── Vectorizer.vectorize()
         └── DecisionTree.train()
         ↓
    [DecisionTree.export()]
         ↓
    Trained Model (.g2p.gz)
         ↓
    [G2P.pronounce()]
         ↓
    Phoneme sequence
```

## Module Responsibilities

### `converter.py` (G2P class)
- User-facing API for pronunciation
- Thin wrapper around DecisionTree
- Simple, Pythonic interface

### `dictionary.py` (Dictionary class)
- Dictionary lifecycle management
- Fetching from public repos
- Processing and normalization
- Integration with training
- All dictionary operations in one place

### `core/` (Core Algorithms)
- Self-contained implementations
- Dependencies: cartlet (decision trees), PyICU (optional, transliteration)
- Reusable components

### `cli/` (Command-Line Tools)
- User interface layer
- Orchestrates core functionality
- Uses high-level API (G2P, Dictionary)

### `cart/` (Bundled Predictor)
- Template file for the standalone Python G2P executor
- Combined with model data by the bundler to produce a self-contained `.py`

### `tools/` (Standalone Utilities)
- Specialized analysis tools
- Not part of core workflow
- Can be run independently

### `utils/` (Core Utilities)
- Shared functionality
- ICU transliteration
- Logging configuration

## Design Principles

1. **High-level API first** - G2P and Dictionary classes for 90% of use cases
2. **Core API for advanced** - DecisionTree for full control
3. **Clean imports** - `from phonebox import G2P` just works
4. **Method chaining** - Fluent API where appropriate
5. **Smart caching** - Don't re-download or re-process unless forced
6. **Minimal utils** - Only truly shared code in utils/
7. **Clear organization** - Each directory has a single purpose

## API Examples

### High-Level API
```python
from phonebox import G2P, Dictionary

# Get pronunciations
g2p = G2P(model='model.g2p.gz')
phones = g2p.pronounce('hello')

# Manage dictionaries
d = Dictionary.fetch('cmudict')
processed = d.process(remove_stress=True)
```

### Core API
```python
from phonebox import DecisionTree

# Full control over training
dt = DecisionTree(locale='en_US', phoneset_name='cmu')
dt.load_prondict(file)
dt.align()
dt.train()
dt.export('model.g2p.gz')
```
