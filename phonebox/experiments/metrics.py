"""Shared metric glossary for G2P compare output."""

G2P_METRICS_FOOTER = """\
## Metrics

| Column | Meaning |
|--------|---------|
| WER% | Word error (strict cooked-phone match) |
| WERr% | Word hit if pred matches gold or any lexicon variant |
| PER% | Normalized phone edit distance / phone count |
| PERr% | PER with locale phone-equivalence (it_IT, pt_BR only) |
| pos% | Position-wise phone match rate |

See [`G2P_EVAL.md`](G2P_EVAL.md) for script usage and flag reference."""
