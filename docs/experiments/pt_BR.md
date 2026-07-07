# G2P experiments: pt_BR

Updated: 2026-05-21 23:04 UTC

Baseline numbers (unaltered lexicon, seed 42) are frozen in [`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md).

## Train-normalize policies

| Policy | Train gold change | Why |
|--------|-------------------|-----|
| `baseline` | none | Control. |
| `surface_final` | final `o`/`ʊ`→`u` when spelling ends in plain `o` | ~99% of `-o` words already have `u`; fixes mixed gold. |
| `do_du` | `surface_final` + `… d o`→`… d u` at `-do`/`-ado` | Targets dominant n:m confusion (`d o` vs `d u`). |
| `citation_expand` | final `u`→`o` on plain `-o` words | Contrast: full vowels on train (expected to hurt n:m). |

## Train-split audit (phones changed before cook)

| Policy | Entries changed | Phone token edits |
|--------|-----------------|-------------------|
| `baseline` | 0 / 64144 | 0 |
| `citation_expand` | 10020 / 64144 | 10020 |
| `do_du` | 193 / 64144 | 193 |
| `surface_final` | 193 / 64144 | 193 |

## Metrics (test gold = original lexicon)

| Policy | 1:1 WER% | 1:1 PER% | n:m WER% | n:m PER% | Δ PER | 1:1 PERr% | n:m PERr% |
|--------|----------|----------|----------|----------|-------|-----------|-----------|
| `baseline` | 14.30 | 2.45 | 22.55 | 4.29 | -1.83 | 2.07 | 2.44 |
| `citation_expand` | 14.30 | 2.45 | 35.15 | 6.10 | -3.64 | 2.07 | 2.36 |
| `do_du` | 14.30 | 2.45 | 22.65 | 4.29 | -1.83 | 2.07 | 2.45 |
| `surface_final` | 14.30 | 2.45 | 22.65 | 4.29 | -1.83 | 2.07 | 2.45 |

PERr uses locale relaxed phone-equivalence (see `phonebox/experiments/equiv.py`).

### Top phone substitutions (1:1, baseline test)

| gold | pred | count |
|------|------|-------|
| `a` | `ɐ` | 18 |
| `ʁ` | `ɾ` | 18 |
| (length) | 6 phones | 15 |
| `j` | `s` | 13 |
| `ĩ` | `j` | 13 |
| `z` | `s` | 12 |
| `ɐ` | `a` | 12 |
| `ɔ` | `o` | 10 |
| `ɾ` | `ʁ` | 10 |
| `ɛ` | `e` | 10 |
| `e` | `ɪ` | 9 |
| (length) | 7 phones | 9 |
| `õ` | `o` | 8 |
| `e` | `i` | 8 |
| `ʃ` | `s` | 8 |
| `ã` | `ɐ̃` | 8 |
| (length) | 8 phones | 7 |
| (length) | 9 phones | 7 |
| (length) | 5 phones | 7 |
| `k` | `ʃ` | 7 |
| `ẽ` | `e` | 7 |
| `ĩ` | `i` | 7 |
| `ẽ` | `ĩ` | 6 |
| (length) | 4 phones | 6 |
| (length) | 11 phones | 6 |

### Top phone substitutions (n:m, baseline test)

| gold | pred | count |
|------|------|-------|
| `i` | `e` | 101 |
| `u` | `o` | 63 |
| `ɐ̃` | `a` | 43 |
| `z` | `s` | 39 |
| `tʃ` | `t` | 35 |
| `ĩ` | `i` | 34 |
| `w̃` | `m` | 20 |
| `dʒ` | `d` | 18 |
| `ɾ` | `ʁ` | 17 |
| `ɛ` | `e` | 16 |
| `ɐ` | `a` | 15 |
| `õ` | `o` | 14 |
| (length) | 6 phones | 13 |
| `ẽ` | `e` | 12 |
| (length) | 5 phones | 11 |
| `ʁ` | `ɾ` | 11 |
| (length) | 7 phones | 10 |
| `ɔ` | `o` | 9 |
| `j` | `s` | 9 |
| (length) | 4 phones | 9 |
| `ũ` | `u` | 8 |
| `ã` | `ɐ̃` | 7 |
| `e` | `i` | 6 |
| `s` | `z` | 6 |
| `o` | `u` | 6 |
