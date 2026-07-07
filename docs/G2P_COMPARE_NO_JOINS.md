# G2P comparison: 1:1 vs n:m (no config joins)

Generated: 2026-05-22 13:30 UTC

## Setup

- Seed: 42
- Test cap: 2000 entries (10% split, shuffled)
- 1:1: **G2PDecisionTree trained on train split** (same xlit, no ``config.json`` joins)
- n:m: MultigramG2P v3 joint decode + LM, EM iterations=15
- Locale ``config.json`` letter/phone joins: **off** (multigram units from EM only; xlit still on)
- Parallel align: True
- Exceptions / lexicon lookup: off (pure G2P)

Pretrained ``*-ipa.g2p.gz`` models are **not** used (they were trained with joins). Compare to [`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md) for join-on + pretrained 1:1.
Multigram trains on the train split only; test words are never in its exception table.

WERr = any cooked pronunciation variant in the lexicon counts as correct.

### vs baseline ([`G2P_COMPARE_BASELINE.md`](G2P_COMPARE_BASELINE.md))

Baseline uses locale config joins and a **pretrained** 1:1 tree; this run trains **both** models on the
train split with joins off. Not directly comparable on 1:1 (different training), but shows n:m without
hand-tuned joins.

| Locale | Baseline Δ PER (1:1−n:m) | No-joins Δ PER | Baseline n:m PER% | No-joins n:m PER% |
|--------|--------------------------|----------------|-------------------|-------------------|
| es_MX | −0.12 (1:1) | −0.27 (1:1) | 0.75 | 0.72 |
| fr_FR | +1.95 (n:m) | +0.06 (n:m) | 7.37 | 7.37 |
| de_DE | +1.70 (n:m) | −0.43 (1:1) | 1.34 | 1.51 |
| en_US | +0.29 (n:m) | −3.07 (1:1) | 15.51 | 13.34 |
| pt_BR | −1.83 (1:1) | −2.20 (1:1) | 4.29 | 4.15 |
| it_IT | +0.24 (n:m) | +1.19 (n:m) | 4.92 | 4.76 |

## Summary

| Locale | 1:1 WER% | 1:1 PER% | n:m WER% | n:m PER% | Δ PER (1:1−n:m) | Winner |
|--------|----------|----------|----------|----------|-----------------|--------|
| es_MX | 3.20 | 0.46 | 5.15 | 0.72 | -0.27 | 1:1 |
| fr_FR | 56.15 | 7.43 | 50.65 | 7.37 | +0.06 | n:m |
| de_DE | 8.70 | 1.08 | 11.75 | 1.51 | -0.43 | 1:1 |
| en_US | 45.25 | 10.27 | 50.15 | 13.34 | -3.07 | 1:1 |
| pt_BR | 10.50 | 1.95 | 22.40 | 4.15 | -2.20 | 1:1 |
| it_IT | 34.60 | 5.95 | 31.20 | 4.76 | +1.19 | n:m |

## es_MX

- Lexicon: `es_ipa.tsv` (45584 entries, 2000 test)
- Multi-pron words: 4
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | pos% |
|---|---|---|---|---|---|
| G2PDecisionTree | 27.5 | 3.20 | 3.20 | 0.46 | 99.35 |
| MultigramG2P | 29.1 | 5.15 | 5.15 | 0.72 | 99.04 |

## fr_FR

- Lexicon: `fr_ipa.tsv` (428675 entries, 2000 test)
- Multi-pron words: 183396
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | pos% |
|---|---|---|---|---|---|
| G2PDecisionTree | 1129.8 | 56.15 | 5.40 | 7.43 | 91.78 |
| MultigramG2P | 468.1 | 50.65 | 8.10 | 7.37 | 91.62 |

## de_DE

- Lexicon: `de_ipa.tsv` (314077 entries, 2000 test)
- Multi-pron words: 547
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | pos% |
|---|---|---|---|---|---|
| G2PDecisionTree | 330.6 | 8.70 | 8.65 | 1.08 | 95.95 |
| MultigramG2P | 375.2 | 11.75 | 11.65 | 1.51 | 97.38 |

## en_US

- Lexicon: `en_ipa.tsv` (135111 entries, 2000 test)
- Multi-pron words: 8254
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | pos% |
|---|---|---|---|---|---|
| G2PDecisionTree | 282.9 | 45.25 | 37.15 | 10.27 | 82.44 |
| MultigramG2P | 196.0 | 50.15 | 46.65 | 13.34 | 81.42 |

## pt_BR

- Lexicon: `pt_ipa.tsv` (66144 entries, 2000 test)
- Multi-pron words: 147
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | PERe% | pos% |
|---|---|---|---|---|---|---|
| G2PDecisionTree | 120.6 | 10.50 | 9.35 | 1.95 | 1.71 | 96.84 |
| MultigramG2P | 60.5 | 22.40 | 12.80 | 4.15 | 2.31 | 97.12 |

## it_IT

- Lexicon: `it_ipa.tsv` (21561 entries, 2000 test)
- Multi-pron words: 631
- 1:1 model: `train-split G2PDecisionTree`
- Config joins: off

| Model | train_s | WER% | WERr% | PER% | PERe% | pos% |
|---|---|---|---|---|---|---|
| G2PDecisionTree | 22.5 | 34.60 | 19.00 | 5.95 | 3.83 | 91.06 |
| MultigramG2P | 34.0 | 31.20 | 11.80 | 4.76 | 2.21 | 96.17 |
