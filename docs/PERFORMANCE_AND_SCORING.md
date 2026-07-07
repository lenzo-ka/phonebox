# Performance Optimizations and Pronunciation Scoring

This document covers recent improvements to training performance and new tools for dictionary quality validation.

## Performance Improvements

### Parallel Alignment (2x Speedup)

The EM alignment phase now runs in parallel using multiprocessing:

```bash
# Enabled by default
phonebox model build en_US dict.txt -o model.g2p.gz

# Disable via config
phonebox model build en_US dict.txt -o model.g2p.gz -c no_parallel.yaml
```

**Performance:**
- Alignment phase: 82s → 43s (48% faster)
- Overall training: 5.6min → 5.1min (10% faster)
- Uses (cpu_count - 1) worker processes
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
phonebox model build en_US dict.txt -o model.g2p.gz
```

**Model size impact:** +27% (425KB → 540KB)
**Runtime impact:** None (distributions are sorted for fast 1-best lookup)

## Pronunciation Likelihood Scoring

Score how likely a pronunciation is according to the model's learned letter-to-phone distributions.

### API Usage

```python
from phonebox import G2P

g2p = G2P("model.g2p.gz")

# Score a pronunciation (returns probability 0-1)
score = g2p.score_pronunciation("READ", ["R", "IY", "D"])
print(f"Likelihood: {score:.4f}")

# Compare pronunciations
read_present = g2p.score_pronunciation("READ", ["R", "IY", "D"])
read_past = g2p.score_pronunciation("READ", ["R", "EH", "D"])

# Choose scoring method
score = g2p.score_pronunciation("the", ["DH", "AH"], method="geometric")
```

### Scoring Methods

The `method` parameter controls how per-phone probabilities are combined:

| Method | Formula | Best For |
|--------|---------|----------|
| `geometric` (default) | ⁿ√(p₁ × p₂ × ... × pₙ) | Balanced scoring |
| `product` | p₁ × p₂ × ... × pₙ | Strict matching |
| `arithmetic` | (p₁ + p₂ + ... + pₙ) / n | Lenient scoring |
| `harmonic` | n / (1/p₁ + 1/p₂ + ... + 1/pₙ) | Penalize weak links |
| `min` | min(p₁, p₂, ..., pₙ) | Worst-case focus |

**Geometric mean** (default) is recommended because:
- Handles varying pronunciation lengths fairly
- Doesn't collapse to zero from one rare phone
- Penalizes consistently weak pronunciations

### Score Interpretation

Scores are probabilities between 0 and 1:
- `> 0.5` = Very likely pronunciation
- `0.1 - 0.5` = Common pronunciation
- `0.01 - 0.1` = Less common but valid
- `< 0.01` = Rare or irregular
- `0` = Contains impossible letter→phone mapping

### How It Works

1. Gets model's per-letter probability distributions (width=1 model recommended)
2. For each phone in the target pronunciation, finds its probability in the corresponding letter's distribution
3. Combines per-phone probabilities using the selected method

### Width=1 Models for Scoring

For pronunciation scoring, train a model with `width=1` (single letter context):

```python
dt = G2PDecisionTree(
    width=1,  # Pure letter-to-phone probabilities
    store_distributions=True,
)
```

This gives clean letter→phone probability distributions without context interference.

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
