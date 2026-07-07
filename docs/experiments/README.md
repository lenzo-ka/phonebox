# G2P experiments (it_IT, pt_BR)

Last run: 2026-05-21 23:04 UTC

Frozen pre-experiment metrics (config joins **on**, pretrained 1:1):
[`G2P_COMPARE_BASELINE.md`](../G2P_COMPARE_BASELINE.md).

Fair join-off comparison (both models on train split, no ``config.json`` joins):
[`G2P_COMPARE_NO_JOINS.md`](../G2P_COMPARE_NO_JOINS.md).

Train-split **phone** normalization policies live in `phonebox/experiments/normalize.py`. Test evaluation always uses the **original** lexicon pronunciations.

## Summary

See **[RESULTS.md](RESULTS.md)** for outcomes. Short version: it_IT train-normalize **hurt** n:m;
pt_BR normalize touched only **193** train entries and did not move n:m (citation_expand hurt as expected).

## Locale reports

- [it_IT.md](it_IT.md) — open/closed vowels (ɛ/e, ɔ/o)
- [pt_BR.md](pt_BR.md) — final reduction and `d o`→`d u`

## Raw JSON

Per-run metrics: `results/<locale>_<policy>.json`

## Reproduce

```bash
export PHONEDECODING_LEXICON_DIR=…/processed
export PHONEDECODING_G2P_DIR=…/build/g2p
python run_g2p_experiments.py --parallel-align
```
