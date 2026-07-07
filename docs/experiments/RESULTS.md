# Experiment results summary

Last run: 2026-05-21 (seed 42, 2000 test, EM 15, parallel align).

**Frozen baseline:** [`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md) — do not edit.

## Key findings

### it_IT (open/closed vowels)

| Policy | n:m WER% | n:m PER% | vs baseline n:m |
|--------|----------|----------|-----------------|
| baseline | 31.55 | 4.92 | — |
| spelling_gated | 34.15 | 5.24 | worse |
| collapse_open | 35.25 | 5.38 | worse |

- **1:1 metrics are unchanged** across policies (pretrained tree; train normalize only affects n:m training).
- Train edits: `spelling_gated` 4216 entries, `collapse_open` 4423 entries (ɛ→e, ɔ→o).
- Normalizing train gold to mid vowels **does not help** n:m when test gold keeps ɛ/ɔ; it teaches the wrong targets.
- Error analysis: top confusions remain **ɛ↔e** and **ɔ↔o** (n:m worse than 1:1 on these). See [it_IT.md](it_IT.md).

**Conclusion:** Lexicon train-normalize (policy A) is not the right lever for Italian; fixes need spelling-side cues (already in `g2p.xlit`), richer context, or post-decode rules — not collapsing open vowels on train.

### pt_BR (reduction / allophony)

| Policy | n:m WER% | n:m PER% | vs baseline n:m |
|--------|----------|----------|-----------------|
| baseline | 22.55 | 4.29 | — |
| surface_final | 22.65 | 4.29 | ~same |
| do_du | 22.65 | 4.29 | ~same |
| citation_expand | 35.15 | 6.10 | much worse |

- Train edits for `surface_final` / `do_du`: only **193** entries (lexicon already ~99% consistent on final `u`).
- `citation_expand` (u→o on train) confirms n:m needs **surface reduction** in gold, not citation vowels.
- n:m top errors vs gold: **i→e**, **u→o** (over-full vowels), **tʃ→t**, not only `d o`→`d u`. See [pt_BR.md](pt_BR.md).
- PERr (o/u, e/i, ʁ/ɾ, …): n:m **2.44%** vs 1:1 **2.07%** — gap shrinks in relaxed metric but 1:1 still wins.

**Conclusion:** pt train-normalize on the small inconsistent subset is insufficient; closing the gap needs **letter context in n:m decode** (or more train data with consistent reduction), not more dict joins.

## Reproduce

```bash
export PHONEDECODING_LEXICON_DIR=…/processed
export PHONEDECODING_G2P_DIR=…/build/g2p
python run_g2p_experiments.py --parallel-align
```

Per-run JSON: `results/<locale>_<policy>.json`.
