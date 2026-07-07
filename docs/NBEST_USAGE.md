# N-Best and Confidence Scoring - Usage Guide

## Overview

Phonebox supports **confidence scores** and **n-best pronunciation lists** for grapheme-to-phoneme conversion. This feature helps identify:

- **Uncertain predictions**: Phonemes with low confidence scores
- **Alternative pronunciations**: Multiple valid ways to pronounce ambiguous words
- **Model reliability**: Per-phoneme confidence for quality assessment

## Quick Start

### Training with Distributions

Distributions are stored by default. To train a model:

```bash
phonebox model build en_US my_dict.txt -o model.g2p.gz
```

This stores probability distributions at ambiguous decision tree leaves, enabling confidence scores and n-best generation.

### Using Confidence Scores

#### CLI

```bash
# Get 1-best with confidence scores
phonebox pronounce hello -m model.g2p.gz --with-confidence
# Output: hello   HH AH L OW   0.95 0.82 0.98 0.87
```

#### Python API

```python
from phonebox import G2P

g2p = G2P(model='model.g2p.gz')
phones, confidences = g2p.pronounce_with_confidence('hello')
# phones = ['HH', 'AH', 'L', 'OW']
# confidences = [0.95, 0.82, 0.98, 0.87]

# Flag uncertain predictions
for phone, conf in zip(phones, confidences):
    marker = "❓" if conf < 0.7 else ""
    print(f"{phone:4} {conf:.2f} {marker}")
```

### Using N-Best Lists

#### CLI

```bash
# Get top 3 pronunciations with scores
phonebox pronounce read -m model.g2p.gz --nbest 3
# Output:
# read   R IY D   0.6500
# read   R EH D   0.3500
# read   R IH D   0.0500
```

#### Python API

```python
from phonebox import G2P

g2p = G2P(model='model.g2p.gz')
nbest = g2p.pronounce_nbest('read', n=3)

for phones, score in nbest:
    print(f"{' '.join(phones):20} (confidence: {score:.1%})")
# Output:
# R IY D               (confidence: 65.0%)
# R EH D               (confidence: 35.0%)
# R IH D               (confidence: 5.0%)
```

## Training Options

### `store_distributions` (config key, default `true`)

Controls whether probability distributions are stored at tree leaves. Set in
a YAML config passed via `--config`, or as a constructor argument to
`DecisionTree` / `G2P.train(...)`.

**Effect:**
- Model size: ~5-10% increase
- Training time: No significant change
- Inference time: 1-best unchanged, n-best adds ~2-5ms/word

**When to use:**
- You need confidence scores
- You want n-best alternatives
- Quality assessment is important

**When to skip:**
- Model size is critical
- Only need 1-best pronunciation
- Speed is paramount

### `min_dist_entropy` (config key)

Minimum entropy threshold for storing distributions (default: 0.1 bits). Set
this in your YAML/TOML/JSON config (passed via `--config` to `phonebox model
build`) or as a constructor argument to `DecisionTree`:

**Lower values:**
- Store more distributions
- Larger model size
- More granular confidence

**Higher values:**
- Store fewer distributions
- Smaller model size
- Only store highly ambiguous cases

## Use Cases

### 1. Quality Control

Flag low-confidence predictions for human review:

```python
g2p = G2P(model='model.g2p.gz')

with open('words.txt') as f:
    for word in f:
        word = word.strip()
        phones, confs = g2p.pronounce_with_confidence(word)

        min_conf = min(confs) if confs else 1.0
        if min_conf < 0.6:
            print(f"WARNING:  Low confidence on '{word}': {min_conf:.2f}")
            print(f"   Pronunciation: {' '.join(phones)}")
```

### 2. Speech Recognition

Generate pronunciation lattices for ASR:

```python
def generate_lattice(g2p, word, min_score=0.1):
    """Generate pronunciation lattice for ASR."""
    nbest = g2p.pronounce_nbest(word, n=10)

    # Filter out very low probability paths
    lattice = [(phones, score) for phones, score in nbest if score >= min_score]

    return lattice

# Use in ASR
word = "read"
lattice = generate_lattice(g2p, word)
# Returns: [(['R', 'IY', 'D'], 0.65), (['R', 'EH', 'D'], 0.35)]
```

### 3. Text-to-Speech

Choose pronunciations based on context:

```python
def contextual_pronunciation(g2p, word, context):
    """Select pronunciation based on context."""
    nbest = g2p.pronounce_nbest(word, n=5)

    # Simple heuristic: past tense context
    if 'past' in context or 'read' in context and word == 'read':
        # Prefer 'R EH D' for past tense
        for phones, score in nbest:
            if 'EH' in phones:
                return phones

    # Default to 1-best
    return nbest[0][0] if nbest else []

# Example
pron = contextual_pronunciation(g2p, 'read', 'I read the book yesterday')
# Returns: ['R', 'EH', 'D']  (past tense)
```

### 4. Model Evaluation

Compute oracle accuracy (best possible accuracy if n-best includes correct):

```python
def oracle_accuracy(g2p, test_data, n=5):
    """Compute oracle accuracy with n-best."""
    correct = 0
    total = 0

    for word, true_phones in test_data:
        nbest = g2p.pronounce_nbest(word, n=n)

        # Check if any alternative matches
        if any(phones == true_phones for phones, _ in nbest):
            correct += 1
        total += 1

    return correct / total

# Oracle accuracy is typically 95-99% with n=5
# vs 85-92% for 1-best accuracy
```

## Model Format

The model format is the same in both modes; only the leaf representation
differs:

| `store_distributions` | Leaf type | Confidence | N-Best |
|-----------------------|-----------|------------|--------|
| `True` (default)      | dict of `{class: probability}` | per-phoneme probabilities | up to many alternatives |
| `False`               | single class label | always 1.0 | always 1 result |

Both kinds of model load through the same `G2P(model=...)` / `DecisionTree`
loader. If you call `pronounce_with_confidence()` or `pronounce_nbest()` on a
model trained without distributions, you get 1.0 / single-result fallbacks,
not an error.

## Performance

### Model Size

Tested on CMUdict (133k words):

| Configuration | Model Size | Increase |
|--------------|-----------|----------|
| Baseline (no dists) | 2.4 MB | - |
| With distributions (entropy=0.1) | 2.6 MB | +8% |
| With distributions (entropy=0.05) | 2.8 MB | +17% |

### Inference Speed

Tested on 10,000 words:

| Operation | Time per word |
|-----------|--------------|
| 1-best (no dists) | 0.12 ms |
| 1-best (with dists) | 0.12 ms |
| with_confidence | 0.14 ms |
| nbest (n=5) | 0.18 ms |
| nbest (n=10) | 0.24 ms |

**Key insight:** 1-best speed unchanged. Sorted distributions enable fast `next(iter(dict))` lookup.

## API Reference

### DecisionTree

```python
class DecisionTree:
    def __init__(
        self,
        ...,
        store_distributions: bool = True,
        min_dist_entropy: float = 0.1,
    ):
        """
        Args:
            store_distributions: Enable confidence scoring (default: True)
            min_dist_entropy: Minimum entropy to store distribution
        """

    def pronounce_with_confidence(self, word: str) -> Tuple[List[str], List[float]]:
        """Get pronunciation with per-phoneme confidence scores."""

    def pronounce_nbest(self, word: str, n: int = 5) -> List[Tuple[List[str], float]]:
        """Get n-best pronunciations sorted by score."""
```

### G2P (High-level API)

```python
class G2P:
    def pronounce_with_confidence(self, word: str) -> Tuple[List[str], List[float]]:
        """Get pronunciation with confidence scores."""

    def pronounce_nbest(self, word: str, n: int = 5) -> List[Tuple[List[str], float]]:
        """Get n-best pronunciations with scores."""
```

## CLI Reference

### Training

```bash
phonebox model build en_US dict.txt -o model.g2p.gz [options]
phonebox model train en_US --dict dict.txt -o model.g2p.gz [options]
```

Distributions are stored by default. The entropy threshold (`min_dist_entropy`)
can be set via a YAML config file with `--config`.

### Inference

```bash
phonebox pronounce WORD [options]

Options:
  --with-confidence    Output confidence scores with 1-best
  --nbest N           Output n-best pronunciations with scores
```

## Examples

### Complete Workflow

```python
from phonebox import G2P

# 1. Train with distributions
g2p_train = G2P.train(
    'cmudict.dict',
    locale='en_US',
    store_distributions=True,
    output='model.g2p.gz'
)

# 2. Load trained model
g2p = G2P(model='model.g2p.gz')

# 3. Use confidence scoring
phones, confs = g2p.pronounce_with_confidence('algorithm')
print(f"Pronunciation: {' '.join(phones)}")
print(f"Min confidence: {min(confs):.2f}")

# 4. Get alternatives
for phones, score in g2p.pronounce_nbest('either', n=3):
    print(f"{' '.join(phones):20} {score:.1%}")
```

### Batch Processing with Quality Metrics

```python
from phonebox import G2P
import csv

g2p = G2P(model='model.g2p.gz')

with open('words.txt') as infile, open('output.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['word', 'pronunciation', 'min_conf', 'avg_conf', 'quality'])

    for line in infile:
        word = line.strip()
        phones, confs = g2p.pronounce_with_confidence(word)

        min_conf = min(confs) if confs else 0
        avg_conf = sum(confs) / len(confs) if confs else 0

        if min_conf >= 0.8:
            quality = 'high'
        elif min_conf >= 0.6:
            quality = 'medium'
        else:
            quality = 'low'

        writer.writerow([
            word,
            ' '.join(phones),
            f'{min_conf:.2f}',
            f'{avg_conf:.2f}',
            quality
        ])
```

## Troubleshooting

### "Model doesn't support n-best"

**Cause:** Model was trained without distributions

**Solution:** Retrain (distributions are enabled by default):
```bash
phonebox model build en_US dict.txt -o new_model.g2p.gz
```

### "All confidences are 1.0"

**Cause:** Either:
1. Model trained without distributions, OR
2. Data is not ambiguous enough (all leaves are deterministic)

**Solution:**
1. Check training used `store_distributions: true` (the default)
2. Try a lower `min_dist_entropy` threshold in your config
3. Some datasets simply don't have ambiguity

### "N-best only returns 1 result"

**Cause:** Model doesn't have distributions (see above)

**Solution:** Retrain with default settings (distributions are on by default)

### "Model size too large"

**Solution:** Increase entropy threshold to store fewer distributions.
Set `min_dist_entropy: 0.2` in your config YAML.

## Best Practices

1. **Training**: Distributions are stored by default; disable only if model size is critical
2. **Thresholds**: Use default entropy=0.1 unless model size is critical
3. **N-best**: Use n=3-5 for most applications; n>10 has diminishing returns
4. **Confidence**: Flag < 0.7 for review, < 0.5 for manual correction
5. **Testing**: Use oracle accuracy to evaluate n-best coverage

## See Also

- [NBEST_IMPLEMENTATION_PLAN.md](NBEST_IMPLEMENTATION_PLAN.md) - Implementation details
- [NBEST_CHECKLIST.md](NBEST_CHECKLIST.md) - Features and validation
- [ACCURACY.md](ACCURACY.md) - Model accuracy metrics
- [README.md](../README.md) - Main documentation
