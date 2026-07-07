# Understanding G2P Accuracy

Phonebox reports two very different things, and it matters which one you mean.

## Pure G2P (generalization)

**What it measures:** the decision-tree model alone, on words it did *not* see
in training — i.e. how well it generalizes grapheme→phoneme patterns.

**How:** disable the exceptions dictionary and predict from the tree only.

**Measured on CMUdict** (`G2PDecisionTree`, 133k-entry train split, held-out
2,000-word test, single best pronunciation; see
[`G2P_EVAL.md`](G2P_EVAL.md#measured-cmudict-results-11-decision-tree)):

| Phones | PER% | pos% |
|--------|-----:|-----:|
| with stress   | 12.96 | 81.8 |
| stress stripped | 11.36 | 81.8 |

- **PER%** = phone error rate (edit distance / phone count).
- **pos%** = position accuracy, the per-position phone match rate — the accuracy
  of the underlying letter→phone classifier the tree is trained to be. See
  [why pos% is meaningful](G2P_EVAL.md#why-position-accuracy-pos).

```python
# Pure model (no dictionary fallback)
g2p.use_dict_fallback = False
```

This is the number to compare across G2P implementations, because it reflects
algorithm quality rather than dictionary coverage.

## Hybrid (memorize + generalize)

**What it measures:** the production path — look the word up in an exceptions
dictionary first, and fall back to the tree only for words not in it.

**How:** `use_dict_fallback = True` (the default). The exceptions table is saved
inside the model.

```python
# Hybrid system (default)
g2p.use_dict_fallback = True
```

The exceptions dictionary holds correct pronunciations from the training
dictionary, so the hybrid system effectively **memorizes the training entries
and generalizes (via the tree) to the rest**. On any word the dictionary
covers, output is exact; on the remainder it is the pure-g2p tree.

This is not cheating — it is the sensible design for a lexicon-backed G2P. But
its headline accuracy is a function of **coverage**, so it is not comparable to
a pure-g2p number and depends entirely on the test set and dictionary.

### Why real-text accuracy is high — and where it isn't

On running text, most tokens are common words the dictionary already covers, so
the *effective* error rate on real text is much lower than the pure-g2p rate:
coverage does most of the work. The errors that remain are dominated by
**genuine pronunciation ambiguity** — homographs / heteronyms such as *read*,
*lead*, *bass*, *live*, *wind*, *tear* — where the correct phones depend on
meaning or context that spelling alone does not carry. No orthography-only G2P
(tree or neural) resolves those without additional context, so they set a floor
on achievable accuracy.

## Comparison to neural G2P

Neural sequence-to-sequence G2P is generally **more accurate** than a decision
tree and more robust to spelling variation. Phonebox trades that accuracy for a
small, fast, interpretable, dependency-free model. Pick the decision tree when
size, speed, determinism, or deployability matter more than the last few points
of accuracy; pick neural when accuracy is paramount. (Published neural results
are cited in [`BENCHMARKS.md`](BENCHMARKS.md).)

## Generalization & Pruning

Phonebox has two complementary anti-overfitting knobs.

### Pre-pruning (always on)

The splitter refuses to grow branches that lack support:

- `min_samples_split` — skip splits below this row count
- `min_samples_leaf` — reject splits that would create a too-small leaf
- `min_confidence` — skip a split that doesn't sharpen the leaf distribution
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

# Custom split, plus a 5% held-out test slice for honest measurement
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

Pruning typically trades a small drop in training-fit for a matching gain on
truly unseen words, and shrinks the tree as a bonus.

## Multiple Pronunciations

Some words have several correct pronunciations:

- `tomato` → `T AH M EY T OW` or `T AH M AA T OW`
- `either` → `IY DH ER` or `AY DH ER`

How phonebox handles it:

- The exceptions dictionary picks the pronunciation **closest to the g2p
  prediction**, for a deterministic single output.
- It does not, on its own, capture all variants. For that, use the n-best API
  (`pronounce_nbest`) or handle homographs with external context.

## Measuring your own model

```python
from phonebox import G2P

g2p = G2P(model="model.g2p.gz")

n_exc = len(g2p._dt.exceptions)
print(f"Exceptions memorized: {n_exc:,}")

# Evaluate pure vs hybrid on a held-out list you control, and report PER/pos%
# so the numbers are reproducible (see phonebox/eval/g2p_compare.py).
```

## Bottom line

- **Pure g2p** (generalization): measured PER ≈ 11–13% / pos% ≈ 82% on held-out
  CMUdict — this is the algorithm's quality.
- **Hybrid** (memorize + generalize): exact on covered words, tree elsewhere;
  effective real-text accuracy is high because coverage is high.
- **Ambiguity** sets a floor: homographs need context no spelling-only model has.
- **Neural G2P is more accurate**; the decision tree wins on size, speed,
  determinism, interpretability, and zero dependencies.
