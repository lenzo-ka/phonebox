# Training performance and pronunciation scoring

Training and scoring use the same saved model configuration. See
[Library and CLI workflows](WORKFLOWS.md) for public entry points and the
primary versus prepared-input training contract.

## Training choices

Primary dictionary training uses the native trainer, serial alignment,
probability distributions, and pruning by default. Stress removal is optional.
For a CMU-tagged dictionary:

```bash
phonebox train --locale en --phoneset cmu --lexicon dict.txt -o model.g2p.gz

# Explicit alternatives when appropriate for the data and available memory
phonebox train --locale en --phoneset cmu --lexicon dict.txt -o model.g2p.gz \
  --parallel-align --remove-stress
```

Parallel alignment uses worker processes and can increase memory consumption.
Its runtime benefit depends on the corpus, configuration, and machine; this
guide makes no fixed speedup or model-size claim. Stored leaf distributions
support candidate scoring, n-best prediction, and per-phone confidence. A
model's scores reflect the distributions it actually retained.

The [recorded CMUdict comparison](https://github.com/lenzo-ka/phonebox/blob/cf1433439738ae6f0499b24b106db85b4ed44c7b/docs/CMUDICT_COMPARISON.md)
provides historical measurements with source revision, dependencies, corpus
hash, split, model-specific filtering, and reproduction instructions. Those
measurements describe that snapshot; they do not establish current training
timings or pronunciation-score calibration.

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

## Score candidate JSONL

The existing batch API scores candidate pronunciations without deciding whether
they are errors:

```python
from phonebox import G2P
from phonebox.pronunciation_analysis import score_entries

model = G2P(model="model.g2p.gz", use_dict_fallback=False)
entries = [{"word": "read", "prons": ["R IY1 D", "R EH1 D"]}]
scored = list(score_entries(model, entries))
```

Each result retains the entry's other fields and stores numeric values in its
`prons` mapping. Variants are ordered by sequence log probability, with input
order retained for ties. This prevents underflowed product probabilities from
creating false ranking ties.

For the installed command, put one such input object on each line:

```bash
phonebox score-prons candidates.jsonl -m model.g2p.gz -o scored.jsonl
phonebox score-prons candidates.jsonl -m model.g2p.gz --method product
```

JSON scores are numbers, not formatted strings. Use the detailed scoring API
when finite log probability and support status are needed to distinguish
underflow from an unsupported sequence. Generic lexicon, validation, and
training operations are described in [WORKFLOWS](WORKFLOWS.md).

## Reuse prepared alignments

Primary training can write an alignment checkpoint through `alignments_out`:

```python
from phonebox import DecisionTree, train_g2p

train_g2p("dict.txt", locale="en", phoneset="cmu",
          alignments_out="alignments.txt")

prepared = DecisionTree(locale="en", phoneset_name="cmu", width=7)
with open("alignments.txt", encoding="utf-8") as source:
    prepared.load_alignments(source)
prepared.train()
```

Prepared alignments are user-generated files, not bundled training data.
Keep their spelling normalization, joins, phoneset, and stress policy consistent
with the consuming model. Prepared training has its own explicit configuration
and does not reconstruct all primary-workflow settings from the file. Reusing
alignments avoids rerunning that stage; no fixed runtime saving is promised.
