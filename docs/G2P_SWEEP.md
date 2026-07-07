# n:m sweep: letter span × LM order

Generated: 2026-05-21 15:44 UTC

## Setup

- Seed: 42, max test: 2000
- EM iterations: 15, parallel align: True
- LM: stdlib add-k (k=0.1) n-gram
- Locales: fr_FR, de_DE, en_US, pt_BR
- Letter spans: [2, 3]
- LM orders: [2, 3]

Each cell shows ``WER% / PER%`` (lower is better). PER is primary.

## fr_FR

| letter_span \ lm_order | 2 | 3 |
|---|---|---|
| 2 | 50.50 / 7.41 | 68.20 / 12.83 |
| 3 | 50.80 / 7.37 | 62.90 / 11.07 |

## de_DE

| letter_span \ lm_order | 2 | 3 |
|---|---|---|
| 2 | 10.95 / 1.53 | 30.50 / 4.44 |
| 3 | 10.15 / 1.34 | 34.45 / 5.13 |

## en_US

| letter_span \ lm_order | 2 | 3 |
|---|---|---|
| 2 | 51.10 / 15.51 | 67.35 / 22.82 |
| 3 | 51.15 / 16.71 | 62.75 / 21.49 |

## pt_BR

| letter_span \ lm_order | 2 | 3 |
|---|---|---|
| 2 | 22.55 / 4.29 | 48.25 / 9.59 |
| 3 | 23.45 / 4.30 | 46.70 / 8.94 |
