# Performance Optimizations and Pronunciation Scoring

This document covers recent improvements to training performance and new tools for dictionary quality validation.

## Performance Improvements

### Parallel Alignment

Alignment is serial by default to avoid duplicating large lexicons across
worker processes. Enable multiprocessing explicitly when memory permits:

```bash
# Safe serial default
phonebox train --locale en_US --lexicon dict.txt -o model.g2p.gz

# Opt in to parallel alignment
phonebox train --locale en_US --lexicon dict.txt -o model.g2p.gz --parallel-align
```

**Performance:**
- Alignment phase: 82s → 43s (48% faster)
- Overall training: 5.6min → 5.1min (10% faster)
- Parallel mode uses worker processes
- Robust Ctrl-C handling with proper cleanup

### Detailed Stage Timing

Training now shows detailed timing for each phase:

```
Loading dictionary...
  OK loaded in 1.4 sec
Aligning letters and phonemes...
  OK aligned in 42.6 sec
Training decision tree...
building tree from 469663 unique observations
tree built: 163521 nodes (242.8 sec, 673.5 nodes/sec)
  OK trained in 242.9 sec
Building exceptions dictionary...
Found 26856 exceptions (5.9 sec)
============================================================
Success! Model trained in 338.3 seconds (5.6 min)
============================================================
```

Progress updates appear every 30 seconds during tree building for long-running builds.

## Distributions Enabled by Default

Models now include probability distributions by default, enabling:
- N-best pronunciation generation
- Per-phoneme confidence scores
- Pronunciation likelihood scoring

```bash
# Distributions are enabled by default
phonebox train --locale en_US --lexicon dict.txt -o model.g2p.gz
```

**Model size impact:** +27% (425KB → 540KB)
**Runtime impact:** None (distributions are sorted for fast 1-best lookup)

## Ordered pronunciation scoring

Scoring measures compatibility with the trained CART model. It sums the mass
of complete ordered emission sequences matching a candidate pronunciation.
Each cooked letter position emits one leaf label; epsilon emits no phone and
joined labels can emit several. Equivalent paths are summed, and positions
cannot be reused or skipped without a learned silent label. Model exceptions
are excluded. Saved spelling rules, joins and optional stress policy determine
the representation; candidate scoring does not change that policy.

```python
from phonebox import G2P

model = G2P(model="model.g2p.gz")
details = model.score_pronunciation_details("read", ["R", "IY1", "D"])
print(details.score, details.log_probability, details.supported)
print(details.to_dict())  # numeric values, public phones, null for no path
score = model.score_pronunciation("read", ["R", "IY1", "D"])
```

Only two methods are supported:

| Method | Meaning |
| --- | --- |
| `geometric` (default) | exp(log sequence mass / cooked letter positions) |
| `product` | Raw sequence probability under independent position emissions |

Unknown methods raise `ValueError`. This alpha API replaces the previous
unordered per-phone maximum aggregation and removes arithmetic, harmonic and
minimum methods. Reversed or repeated phones need their own complete matching
path; finding those phones somewhere in the word is insufficient.

`PronunciationScore` records `score`, `probability`, `log_probability`,
`positions`, effective public `phones`, `supported`, and `method`. No matching
path has score/probability zero, log probability `None`, and supported false.
No cooked positions is unsupported. A supported sequence can have raw probability
zero from floating-point underflow while retaining finite log probability;
use `supported` and log probability to distinguish that case. Log-space forward
summation keeps long-sequence calculations stable. Deterministic leaves are
point masses, so a model without ambiguous distributions has coarser scores.

Geometric normalization reduces length bias across words; for variants of one
word, position count is fixed and both methods give the same ranking. Neither
score is a calibrated probability of correctness. There are no universal
thresholds defining common, rare or erroneous pronunciations. Zero support can
be a legitimate acronym, foreign pronunciation, or sparse training support.

### Training and saved-material bounds

A rich-context model may memorize an unusual pronunciation in its own training
lexicon and assign it mass one. `train_g2p(..., width=1)` is an explicit coarse
orthographic-compatibility baseline; it also has sparse-data and memorization
limits. A stronger outlier assessment needs held-out words (keeping every
variant of a word in the same group) and empirical ranking evidence. Scores
prioritize manual review rather than declare errors or detect abbreviations.

Phonebox `.g2p.gz`, `.jsonl`, and native `.cart` loading restore their embedded
preprocessing metadata. Native CART probabilities are stored as float32, so
round-trip numerical comparisons need tolerance. Metadata absent from an old
artifact is not invented. Multigram candidate likelihood is unsupported; its
EM unit distribution alone is not the full saved unit-language-model likelihood.

## Dictionary Validation Tools

### score_dictionary.py

Validate pronunciation dictionaries by scoring every entry against a trained model.

#### Basic Usage

```bash
# Find worst pronunciations (potential errors)
python examples/score_dictionary.py --bottom 50

# Score entire dictionary
python examples/score_dictionary.py -o scored.tsv

# Show highest confidence pronunciations
python examples/score_dictionary.py --top 100
```

#### Output Format (TSV)

```
word    pronunciation    score    model_prediction    match    orig_rank
CAT     K AE T          0.000    K AE T              =        1
READ    R EH D         -0.050    R EH D              =        1
READ(2) R IY D         -5.756    R EH D              !        2
```

**Columns:**
- `word`: Dictionary entry with instance number
- `pronunciation`: Phonemes from dictionary
- `score`: Average log probability (higher = better)
- `model_prediction`: What model would predict
- `match`: `=` if matches, `!` if differs
- `orig_rank`: Original variant number in dictionary

### Reordering Pronunciations

Reorder pronunciation variants within each word by model confidence:

```bash
# Reorder variants by score
python examples/score_dictionary.py --reorder -o reordered.tsv

# Show only words where ranking changed
python examples/score_dictionary.py --rank-changes-only -o changes.tsv
```

**Reordered output adds `new_rank` column:**

```
word    pronunciation    score    model_pred    match    orig_rank    new_rank
abs     AE B Z          0.000    AE B Z        =        2            1
abs(2)  EY B IY EH S   -9.000    AE B Z        !        1            2
```

This shows `abs` was originally variant #2 but became #1 after reordering (better score).

**Use cases:**
1. **Quality control**: Find pronunciation errors in dictionaries
2. **Variant ordering**: Rank pronunciations by likelihood
3. **Dictionary cleanup**: Identify and fix irregular entries
4. **Model validation**: See what the model considers unusual

### Examples from CMUdict

**Words with rank changes (5,631 entries):**
```
abs:     (2) → (1)  Score improved from rank 2 to 1
asap:    (2) → (1)  Better pronunciation moved up
anfal:   (2) → (1)  Original #1 was very poor
```

**Worst scored entries (< -10):**
- Acronyms spelled out: FYI, BBC, CNN
- Unusual abbreviations: AOL(2), AWB(2)
- Likely errors: Wrong phoneme sequences

### Integration with Unix Tools

The TSV format integrates with standard tools:

```bash
# Extract just word + pronunciation
cut -f1-2 reordered.tsv > dict.txt

# Sort by score
sort -t$'\t' -k3,3n scored.tsv | head -100

# Find specific words
grep "^READ" reordered.tsv

# Count perfect scores
awk -F'\t' '$3 > -0.01 {count++} END {print count}' scored.tsv
```

## Performance Summary

| Phase         | Before  | After   | Speedup |
|--------------|---------|---------|---------|
| Alignment    | 82s     | 43s     | 1.9x    |
| Tree build   | 243s    | 243s    | 1.0x    |
| Total        | 5.6min  | 5.1min  | 1.1x    |

**Model size:**
- Without distributions: 425KB
- With distributions: 540KB (+27%)

**Dictionary quality:**
- 50.6% perfect pronunciations (score > -0.01)
- 83.1% good pronunciations (score > -1.0)
- 0.2% very poor (score < -10.0, likely errors)

## Saving and Loading Alignments

Alignment is the slowest part of training. Save alignments for fast iteration:

```python
from phonebox.core.g2p_model import G2PDecisionTree

# First time: align and save
dt = G2PDecisionTree(locale="en_US", phoneset_name="cmu")
with open("dict.txt") as f:
    dt.load_prondict(f)
dt.align()
dt.save_alignments("alignments.txt")  # Save for reuse

# Later: load alignments and iterate quickly
dt = G2PDecisionTree(width=1, store_distributions=True)  # Try different params
with open("alignments.txt") as f:
    dt.load_alignments(f)  # ~1 second vs ~40 seconds
dt.train()
```

Pre-computed alignments are included in `data/cmudict/alignments.txt`.

## Future Optimizations

Potential areas for further speedup:

1. **Parallel tree building** - Split by target phoneme (4-8x speedup possible)
2. **Numpy vectorization** - Replace Python lists in hot paths (40-50% faster)
3. **Cythonize entropy calculations** - Compile hot loops (30% faster)
4. **Cache entropy computations** - Avoid redundant calculations (needs smart caching)

The parallel alignment was the "easy win" that gave good speedup with minimal complexity.
