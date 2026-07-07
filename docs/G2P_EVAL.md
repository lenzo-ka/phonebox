# G2P evaluation (1:1 vs MultigramG2P)

Compare **G2PDecisionTree** (1:1, cart tree) and **MultigramG2P** (n:m, joint EM
+ unit n-gram Viterbi) on held-out IPA lexicon slices. Use the **`phonebox compare`**
CLI or the thin repo-root wrappers (`compare_g2p*.py`).

## Environment

```bash
export PHONEDECODING_LEXICON_DIR=/path/to/lexicons/processed
export PHONEDECODING_G2P_DIR=/path/to/build/g2p   # join-on compare only
```

Lexicons: `es_ipa.tsv`, `fr_ipa.tsv`, `de_ipa.tsv`, `en_ipa.tsv`, `pt_ipa.tsv`,
`it_ipa.tsv`.

## CLI

| Command | Output | Purpose |
|---------|--------|---------|
| `phonebox compare locale …` | stdout | Single locale; all eval flags |
| `phonebox compare all` | `docs/G2P_COMPARE.md` | Six locales; pretrained 1:1 + train n:m |
| `phonebox compare all --no-config-joins` | `docs/G2P_COMPARE_NO_JOINS.md` | Both models on train split; joins off |
| `phonebox train-multigram …` | model + sidecars | Train/export n:m for `phonebox pronounce` |

Repo-root wrappers (same logic): `compare_g2p.py`, `compare_g2p_all.py`,
`compare_g2p_sweep.py`, `dump_units.py`, `run_g2p_experiments.py`.

## Other scripts

| Script | Output | Purpose |
|--------|--------|---------|
| `compare_g2p_sweep.py` | `docs/G2P_SWEEP.md` | n:m span × LM order (joins **on**) |
| `dump_units.py` | `docs/G2P_UNITS.md` | Top EM multigram units (joins **on**) |
| `run_g2p_experiments.py` | `docs/experiments/` | it_IT / pt_BR train-normalize A/B tests |
| `run_full_eval.sh` | above (except no-joins) | Sequential regen |

## Key flags (`compare locale`)

- **`--no-config-joins`** — `Vectorizer.disable_config_joins()`; ignores
  `--baseline-model`; trains 1:1 on the train split (fair vs n:m).
- **`--baseline-model`** — load pretrained `.g2p.gz` for 1:1 (join-on eval).
- **`--relaxed-per`** — PERr column using locale phone-equivalence sets
  (`phonebox/experiments/equiv.py`: it_IT e/ɛ o/ɔ; pt_BR reduction/allophony).
- **`--parallel-align`** — parallel multigram EM / Viterbi on large lexicons.
- **`--use-exceptions`** — train-split lexicon lookup on both models (hybrid).

Locale `config.json` may set `"multigram": {"max_letter_span": 3}` (fr_FR,
de_DE). Compare scripts read this via `Vectorizer.multigram_config()`.

## Metrics (compare scripts)

| Column | Meaning |
|--------|---------|
| **WER%** | Word error: predicted cooked phones ≠ gold (strict) |
| **WERr%** | Word “hit” if pred matches gold **or** any lexicon variant |
| **PER%** | Phone error rate: normalized Levenshtein edit distance / phone count |
| **PERr%** | PER with locale-specific phone equivalence (eval only) |
| **pos%** | Position accuracy: per-position phone match rate (see below) |

### Why position accuracy (pos%)

A 1:1 decision tree is, at heart, a *per-position classifier*: for each letter,
in its context window, it predicts exactly one phone (or `∅` for a silent
letter). **pos%** measures exactly that — the fraction of positions where the
predicted phone is correct, compared slot-by-slot with no re-alignment,
normalized by the longer of the predicted/gold lengths. It is the accuracy of
the thing the model is actually trained to do, before epsilon-removal and
joining collapse the output into the final phone string.

- **pos%** answers *"how good is the underlying letter→phone classifier?"* — one
  wrong phone costs exactly one position, so it degrades gracefully.
- **PER%** answers *"how wrong is the final phone string?"* — via edit distance,
  where a single insertion or deletion can shift and penalize the whole tail.

Because a 1:1 model's output is inherently letter-aligned, pos% is an honest
accuracy for it. For n:m models, where predicted and gold lengths diverge, PER%
is the better lens; treat pos% as informative only there.

## Measured CMUdict results (1:1 decision tree)

Pure G2P — **no** exceptions dictionary — `G2PDecisionTree` trained on a
133,166-entry train split of CMUdict and evaluated on a held-out 2,000-word
slice (`seed 42`, single best pronunciation). Inline `#` comments are stripped
by `parse_dict_line`.

| Phones | PER% | WER% | pos% |
|--------|-----:|-----:|-----:|
| with stress (`AH0`/`AH1`/`AH2` kept) | 12.96 | 53.8 | 81.8 |
| stress stripped (`0/1/2` removed)    | 11.36 | 44.9 | 81.8 |

Reproduce (with stress):

```bash
phonebox compare locale --lexicon data/cmudict/cmudict.dict \
    --locale en_US --phoneset cmu --skip-multigram --parallel-align
```

For the stress-stripped row, strip the trailing `0/1/2` from each phone first
(pos% is unchanged because the position of every phone is the same; only the
stress distinctions that PER/WER counted disappear).

These are **pure-generalization** numbers on unseen words. Production use adds an
exceptions dictionary that is consulted first and falls back to the tree — i.e.
it *memorizes the training entries and generalizes to the rest*, so real-world
error on in-vocabulary words is lower (bounded below by these figures). The
exceptions table is saved in the model; see `docs/ACCURACY.md`.

These numbers differ from other docs that use different splits or the
hybrid (dict + tree) path. See frozen baseline:
[`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md).

## Library API (n:m)

```python
from phonebox.core.multigram_g2p import MultigramG2P
from phonebox.core.vectorizer import Vectorizer

vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
mg = MultigramG2P(max_letter_span=2, max_phone_span=2)
mg.train_from_dict("it_ipa.tsv")
letters = vec.cook_letters("ciao", g2p=True)
phones = mg.pronounce_letters(letters, word="ciao")
```

CLI training and inference:

```bash
phonebox train-multigram --locale it_IT --lexicon it_ipa.tsv -o model.g2p.gz
phonebox pronounce ciao -m model.g2p.gz   # auto-detects .units.json sidecar
```

1:1 inference: `phonebox pronounce -m tree-only.g2p.gz` (no sidecar).

## Related CLI

- **`phonebox suggest-joins`** — EM join discovery (`MultigramAligner` only;
  pass **`--locale`** to pick `multigram` spans from locale config).
- **`phonebox check`** — lexicon vs phoneset validation (NFC, xenophones).
