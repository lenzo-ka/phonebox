# Posterior decomposition scattering: complete development result

The prespecified unigram-q teacher and left-anchor CART projection do not support the proposed gain with independent-position prediction: posterior training has higher variant PER than its matched hard-projection control in all four conditions. Both remain worse than epsilon-CART. These negative results are retained; baseline settings and test predictions are unchanged.

| Condition | Supervision | Dev variant WER % | Dev variant PER % | Admitted / skipped | Targets + fit (s) | Model bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| french | hard-q | 31.9667 | 6.8780 | 78,588 / 18 | 177.07 | 193144 |
| french | posterior-q | 33.3333 | 6.9885 | 78,588 / 18 | 564.04 | 3887623 |
| french | epsilon | 14.1221 | 2.6885 | 77,555 / 1,051 | 265.59 | 91743 |
| italian | hard-q | 25.9970 | 4.0224 | 72,448 / 20 | 129.15 | 133882 |
| italian | posterior-q | 29.5796 | 4.6151 | 72,448 / 20 | 564.96 | 3719216 |
| italian | epsilon | 16.7500 | 2.2452 | 70,527 / 1,941 | 177.29 | 59230 |
| cmudict-preserved | hard-q | 61.1633 | 18.6094 | 112,032 / 42 | 211.58 | 547150 |
| cmudict-preserved | posterior-q | 62.5506 | 19.3105 | 112,032 / 42 | 619.84 | 7441448 |
| cmudict-preserved | epsilon | 53.6838 | 13.3886 | 109,929 / 2,145 | 197.70 | 367282 |
| cmudict-removed | hard-q | 52.8996 | 14.7028 | 111,775 / 42 | 192.59 | 436755 |
| cmudict-removed | posterior-q | 54.7523 | 15.2503 | 111,775 / 42 | 714.77 | 7565001 |
| cmudict-removed | epsilon | 42.4300 | 9.7079 | 109,675 / 2,142 | 202.19 | 277800 |

Posterior-minus-hard PER changes (percentage points):
- french: +0.1105.
- italian: +0.5927.
- cmudict-preserved: +0.7011.
- cmudict-removed: +0.5475.

All 12 metric rows were independently recomputed from their saved predictions on the entire fixed development splits and matched exactly. The matched hard/posterior pair shares fitted q, preprocessing and admission within each condition. Epsilon-CART admission differs and is reported. No significance, novelty or independent test-generalization claim is made.

Frozen source: `000eb75`. Shared q training is accounted separately from target preparation and CART fitting. Full settings and hypothesis are in [the experiment protocol](CART_DECOMPOSITION_EXPERIMENT.md). Raw models, predictions and receipts remain private.

The teacher already estimates q using soft EM. This experiment concerns retaining its final gold-conditioned alignment uncertainty through CART targets. Marginal targets discard complete-path correlations; a failure here does not falsify averaging over complete possibilities during EM. Teacher quality, projection and leaf storage remain separate mechanisms. The runtime lattice and leaf-retention ablations have changed, explicitly declared objectives; their outcomes do not retroactively alter this independent-position hypothesis.

French preparation excludes explicitly liaison-marked variants and supplies no following-word context. It measures citation forms, not contextual liaison. The Italian and CMU replications show the observed regression is not confined to French; they do not identify its entire cause.
