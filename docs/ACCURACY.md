# Understanding G2P Accuracy Metrics

## Two Types of Accuracy

### 1. Pure G2P Accuracy (~95%)
**What it measures:** Decision tree model quality alone
**How:** Disables exceptions dictionary, tests only g2p predictions
**Typical:** 95-97% phoneme-level accuracy

```python
# Test pure model
g2p.use_dict_fallback = False
accuracy = test(g2p)  # ~95%
```

**This measures:**
- How well the model learns grapheme-to-phoneme patterns
- Generalization to unseen words
- Algorithm quality

### 2. Hybrid Accuracy (~99.5%)
**What it measures:** Combined dict + g2p system
**How:** Uses exceptions dictionary first, falls back to g2p
**Typical:** 99-99.5% accuracy

```python
# Test hybrid system
g2p.use_dict_fallback = True  # Default
accuracy = test(g2p)  # ~99.5%
```

**This measures:**
- Real-world performance
- What users actually experience
- Combined system quality

## Why Hybrid is High

The exceptions dictionary contains **correct pronunciations from the training dictionary**:

- **26,673 words** (in en_US model) have exact pronunciations
- These are words the g2p model mispronounced during training
- They're stored as lookups (instant, 100% accurate for those words)

**Hybrid accuracy is high by design** - it's not cheating, it's smart engineering.

## What Matters for Different Use Cases

### Research / Algorithm Development
→ **Pure g2p accuracy** (95-97%)
- Measures algorithm quality
- Comparable across implementations
- Shows model learning capability

### Production / User-Facing
→ **Hybrid accuracy** (99-99.5%)
- What users experience
- Real-world performance
- Includes system optimizations

## Generalization & Pruning

Phonebox has two complementary anti-overfitting knobs:

### Pre-pruning (always on)

The splitter refuses to grow branches that lack support:

- `min_samples_split` - skip splits below this row count
- `min_samples_leaf` - reject splits that would create a too-small leaf
- `min_confidence` - skip a split that doesn't sharpen the leaf distribution
  enough to justify the extra node

Tune these in your YAML config or pass them to `DecisionTree(...)`. They run
during training and shape what gets built.

### Post-pruning (opt-in, recommended)

After the tree is built, hold out a slice of the training data and walk the
tree bottom-up: collapse any subtree whose children don't help on the
held-out rows. This is "reduced-error pruning" and it directly attacks
overfitting because the validation rows weren't used to choose splits.

```bash
# CLI: prune with a 5% validation hold-out
phonebox model build en_US dict.txt -o model.g2p.gz --prune

# Custom split, plus a 5% held-out test slice for honest accuracy
phonebox model build en_US dict.txt -o model.g2p.gz \
  --prune --validation-split 0.05 --test-split 0.05

# Recipe: prune end-to-end
phonebox recipe cmudict pocketsphinx -o g2p.py --prune
```

```python
# Python API
from phonebox import G2P
g2p = G2P.train(
    "dict.txt", locale="en_US",
    prune=True, validation_split=0.05, test_split=0.05,
)
```

Or in YAML:

```yaml
prune: true
validation_split: 0.05
test_split: 0.05
```

Pruning typically trades a small drop in training-fit accuracy for a
matching gain on truly unseen words, and shrinks the tree as a bonus.

## Multiple Pronunciations

Some words have multiple correct pronunciations:
- `tomato` → T AH M EY T OW or T AH M AA T OW
- `either` → IY DH ER or AY DH ER

**How we handle it:**
- Exceptions dict picks pronunciation **closest to g2p prediction**
- Provides consistency (same model always gives same output)
- May not capture all variants (limitation)

For full multi-pronunciation support, would need:
- N-best lists
- Pronunciation confidence scores
- Or separate handling of homographs

## Measuring Your Model

```python
from phonebox import G2P

g2p = G2P(model='model.g2p.gz')

# Check hybrid vs pure
print(f"Exceptions loaded: {len(g2p._dt.exceptions):,}")
print(f"Coverage: ~{100 * len(g2p._dt.exceptions) / 135000:.1f}% from dict")
print(f"Fallback: ~{100 - (100 * len(g2p._dt.exceptions) / 135000):.1f}% from g2p")

# Test on words
test_words = ['hello', 'world', 'test']
for word in test_words:
    in_dict = word in g2p._dt.exceptions
    source = "dict" if in_dict else "g2p"
    print(f"{word}: {source}")
```

## Bottom Line

- **Pure g2p: ~95%** - measures model quality
- **Hybrid: ~99.5%** - what users get
- **Pruning fights overfitting** - opt-in via `--prune`, recommended on real data
- **Exceptions are smart** - not cheating, good engineering
- **Both metrics matter** - for different purposes
