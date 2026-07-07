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
| **pos%** | Position-wise phone match rate (phoneme accuracy) |

These differ from `docs/ACCURACY.md` / `docs/BENCHMARKS.md` (CMUdict, hybrid
on/off, different splits). See frozen baseline:
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
