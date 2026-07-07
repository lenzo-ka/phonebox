# G2P experiments: it_IT

Updated: 2026-05-21 23:02 UTC

Baseline numbers (unaltered lexicon, seed 42) are frozen in [`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md).

## Train-normalize policies

| Policy | Train gold change | Why |
|--------|-------------------|-----|
| `baseline` | none | Control; matches baseline compare. |
| `spelling_gated` | ɛ→e, ɔ→o only when word has no è/ò | Plain spelling dominates; open vowels stay when orthography marks them. |
| `collapse_open` | always ɛ→e, ɔ→o | Upper bound; removes open/closed entirely on train. |

## Train-split audit (phones changed before cook)

| Policy | Entries changed | Phone token edits |
|--------|-----------------|-------------------|
| `baseline` | 0 / 19561 | 0 |
| `collapse_open` | 4423 / 19561 | 4495 |
| `spelling_gated` | 4216 / 19561 | 4288 |

## Metrics (test gold = original lexicon)

| Policy | 1:1 WER% | 1:1 PER% | n:m WER% | n:m PER% | Δ PER | 1:1 PERr% | n:m PERr% |
|--------|----------|----------|----------|----------|-------|-----------|-----------|
| `baseline` | 32.25 | 5.16 | 31.55 | 4.92 | +0.24 | 2.76 | 2.37 |
| `collapse_open` | 32.25 | 5.16 | 35.25 | 5.38 | -0.22 | 2.76 | 2.40 |
| `spelling_gated` | 32.25 | 5.16 | 34.15 | 5.24 | -0.08 | 2.76 | 2.40 |

PERr uses locale relaxed phone-equivalence (see `phonebox/experiments/equiv.py`).

### Top phone substitutions (1:1, baseline test)

| gold | pred | count |
|------|------|-------|
| `ɛ` | `e` | 128 |
| `ɔ` | `o` | 111 |
| `e` | `ɛ` | 100 |
| `s` | `z` | 43 |
| `o` | `ɔ` | 42 |
| `dz` | `ts` | 28 |
| `i` | `j` | 27 |
| `j` | `i` | 24 |
| `t` | `ts` | 21 |
| `j` | `o` | 20 |
| `z` | `s` | 17 |
| `ts` | `j` | 17 |
| `ʃ₊ʃ` | `ʃ` | 15 |
| `o` | `n` | 15 |
| (length) | 7 phones | 15 |
| (length) | 5 phones | 15 |
| `ts₊ts` | `ts` | 14 |
| (length) | 6 phones | 14 |
| (length) | 8 phones | 12 |
| (length) | 9 phones | 11 |
| `tʃ` | `k` | 11 |
| `o` | `j` | 10 |
| `dʒ` | `d` | 10 |
| (length) | 10 phones | 9 |
| `u` | `w` | 9 |

### Top phone substitutions (n:m, baseline test)

| gold | pred | count |
|------|------|-------|
| `ɛ` | `e` | 215 |
| `ɔ` | `o` | 143 |
| `j` | `i` | 46 |
| `z` | `s` | 45 |
| `e` | `ɛ` | 37 |
| `i` | `j` | 21 |
| `t` | `ts` | 19 |
| `ts` | `j` | 19 |
| `j` | `o` | 18 |
| `s` | `z` | 17 |
| `w` | `u` | 16 |
| `t` | `ts₊ts` | 15 |
| `tʃ` | `t` | 14 |
| `ts₊ts` | `ts` | 13 |
| `dz` | `ts` | 12 |
| (length) | 10 phones | 11 |
| `o` | `ɔ` | 11 |
| `o` | `n` | 10 |
| `ts` | `a` | 10 |
| (length) | 8 phones | 10 |
| `dʒ` | `d` | 10 |
| (length) | 7 phones | 9 |
| (length) | 9 phones | 8 |
| `tʃ` | `k` | 7 |
| (length) | 5 phones | 7 |
