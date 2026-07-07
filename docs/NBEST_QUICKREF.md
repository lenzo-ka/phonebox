# N-Best and Confidence Scoring - Quick Reference

## Training

```bash
# Training (distributions stored by default, enabling n-best)
phonebox model build en_US dict.txt -o model.g2p.gz
```

## CLI Usage

```bash
# Standard 1-best
phonebox pronounce hello -m model.g2p.gz
# Output: hello   HH AH L OW

# With confidence scores
phonebox pronounce hello -m model.g2p.gz --with-confidence
# Output: hello   HH AH L OW   0.95 0.82 0.98 0.87

# N-best alternatives
phonebox pronounce read -m model.g2p.gz --nbest 3
# Output:
# read   R IY D   0.6500
# read   R EH D   0.3500
# read   R IH D   0.0500
```

## Python API

### Load Model

```python
from phonebox import G2P

g2p = G2P(model='model.g2p.gz')
```

### Standard Pronunciation

```python
phones = g2p.pronounce('hello')
# Returns: ['HH', 'AH', 'L', 'OW']
```

### With Confidence Scores

```python
phones, confidences = g2p.pronounce_with_confidence('hello')
# phones: ['HH', 'AH', 'L', 'OW']
# confidences: [0.95, 0.82, 0.98, 0.87]

# Always: len(phones) == len(confidences)
# Always: 0.0 <= conf <= 1.0
```

### N-Best Alternatives

```python
nbest = g2p.pronounce_nbest('read', n=5)
# Returns: [(phones, score), ...]
# Example: [(['R', 'IY', 'D'], 0.65), (['R', 'EH', 'D'], 0.35)]

# Sorted by score (descending)
# Length: min(n, available alternatives)
```

### Train from Code

```python
g2p = G2P.train(
    'dict.txt',
    locale='en_US',
    store_distributions=True,  # Enable n-best
    output='model.g2p.gz'
)
```

## Common Patterns

### Flag Low Confidence

```python
phones, confs = g2p.pronounce_with_confidence(word)

if min(confs) < 0.7:
    print(f"WARNING:  Low confidence on '{word}'")
```

### Quality Classification

```python
phones, confs = g2p.pronounce_with_confidence(word)
min_conf = min(confs)

if min_conf >= 0.8:
    quality = "HIGH"
elif min_conf >= 0.6:
    quality = "MEDIUM"
else:
    quality = "LOW"
```

### Batch Processing

```python
results = []
for word in words:
    phones, confs = g2p.pronounce_with_confidence(word)
    results.append({
        'word': word,
        'phones': phones,
        'min_conf': min(confs),
        'avg_conf': sum(confs) / len(confs)
    })
```

### Get Best Alternative

```python
nbest = g2p.pronounce_nbest(word, n=5)
if nbest:
    best_phones, best_score = nbest[0]
```

### Check All Alternatives

```python
for phones, score in g2p.pronounce_nbest(word, n=5):
    print(f"{' '.join(phones):20} {score:.1%}")
```

## Modes

| `store_distributions` | Confidence | N-Best | Notes |
|-----------------------|------------|--------|-------|
| `true` (default)      | per-phoneme probabilities | up to many alternatives | full feature support |
| `false`               | always 1.0 | always 1 result | smaller / faster, no scoring |

## Thresholds and Interpretation

### Confidence Score Interpretation

| Confidence | Interpretation | Action |
|-----------|----------------|--------|
| ≥ 0.90 | High confidence | Accept |
| 0.70 - 0.89 | Medium confidence | Review if critical |
| < 0.70 | Low confidence | Flag for review |

### N Parameter Guidelines

| N | Use Case |
|---|----------|
| 1 | Same as 1-best |
| 3-5 | Most applications |
| 5-10 | ASR, TTS with alternatives |
| > 10 | Diminishing returns |

## Performance

### Model Size

| Config | Size |
|--------|------|
| No distributions | 100% (baseline) |
| With distributions | 105-110% |

### Speed (per word)

| Operation | Time |
|-----------|------|
| 1-best | 0.12 ms |
| with_confidence | 0.14 ms |
| nbest (n=5) | 0.18 ms |

## Error Handling

### Model Doesn't Support N-Best

```python
# Check if model has distributions
from phonebox import G2P
g2p = G2P(model='model.g2p.gz')
phones, confs = g2p.pronounce_with_confidence("test")
if any(c < 1.0 for c in confs):
    print("[x] N-best supported")
else:
    print("[ ] Model may lack distributions")
```

### Empty Results

```python
phones, confs = g2p.pronounce_with_confidence(word)
if not phones:
    print(f"No pronunciation for '{word}'")
```

## Examples

### Quality Control Pipeline

```python
import csv

with open('words.txt') as f, open('output.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerow(['word', 'pronunciation', 'confidence', 'quality'])

    for line in f:
        word = line.strip()
        phones, confs = g2p.pronounce_with_confidence(word)

        min_conf = min(confs) if confs else 0
        quality = 'HIGH' if min_conf >= 0.8 else \
                  'MEDIUM' if min_conf >= 0.6 else 'LOW'

        writer.writerow([
            word,
            ' '.join(phones),
            f'{min_conf:.2f}',
            quality
        ])
```

### ASR Lattice Generation

```python
def asr_lattice(word, min_score=0.05):
    """Generate pronunciation lattice for ASR."""
    nbest = g2p.pronounce_nbest(word, n=10)
    return [(p, s) for p, s in nbest if s >= min_score]
```

## Documentation

- **User Guide:** [docs/NBEST_USAGE.md](NBEST_USAGE.md)
- **Examples:** [examples/nbest_example.py](../examples/nbest_example.py)

## Common Issues

**Q: All confidence scores are 1.0**
A: Model trained with `store_distributions: false` or data has no ambiguity

**Q: N-best only returns 1 result**
A: Same as above - retrain with `store_distributions: true` (the default)

**Q: Model size too large**
A: Increase `min_dist_entropy` threshold in your config

**Q: Low confidence but pronunciation looks correct**
A: Model uncertain due to limited training data for that pattern

## Tips

[x] **DO:**
- Leave `store_distributions: true` (the default) unless you need a smaller / faster model
- Use confidence < 0.7 threshold for flagging
- Request n=3-5 for most applications

[ ] **DON'T:**
- Use very large n (>10) - diminishing returns
- Expect confidence to sum to 1.0 (per-position independence)
- Assume low confidence means wrong (could be valid but unusual)

---

**Quick Links:**
- Train: `phonebox model build en_US dict.txt -o model.g2p.gz`
- Confidence: `g2p.pronounce_with_confidence(word)`
- N-best: `g2p.pronounce_nbest(word, n=5)`
