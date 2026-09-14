# Why posterior decomposition can lose, and what improves it

These are development diagnostics, not held-out test results. The original
weighted-training protocol and frozen models remain unchanged. French evaluates
citation forms without following-word context; the Italian replication makes
liaison an insufficient explanation for the regression.

## Marginalization loses alignment correlations

Gold-conditioned alignment posteriors provide a control without CART learning:
every complete admitted alignment emits the supplied gold pronunciation. Yet
concatenating the independently most probable positional targets gets 429 of
7,657 supported French development spellings wrong (5.60%), and 407 of 7,380
Italian spellings wrong (5.51%). There were respectively 26 and 17 unsupported
spellings. This establishes a projection/decision failure; it does not attribute
the entire learned-model regression to that failure.

For example, suppose one phone can occupy any of three letters, equally likely.
Every alignment emits that phone, but each position has epsilon probability
2/3. Taking each position's mode emits nothing. Averaging over complete paths is
sound; taking modes after discarding their correlations changes the objective.
A support-only lattice can still admit all-epsilon combinations, so legality
alone does not recover the original joint posterior.

## Fixed-tree decoding results

Any-reference-variant phone error rate (%), on all development spellings:

| Language / training | Independent CART | Unit lattice | Lattice + 0.1% backoff |
| --- | ---: | ---: | ---: |
| French hard q | 6.8780 | 12.4774 | 5.1866 |
| French posterior q | 6.9885 | 5.4496 | 4.7463 |
| Italian hard q | 4.0224 | 7.0157 | 3.2382 |
| Italian posterior q | 4.6151 | 3.8735 | 3.4988 |

The epsilon-CART reference remains better: French 2.6885%, Italian 2.2452%.
Posterior training wins against hard training after backoff in French, but loses
in Italian. The evidence for posterior training is mixed rather than uniformly
negative or positive.

Without backoff, the hard/soft lattice has 737/90 French and 378/51 Italian
spellings with no legal path. Backoff reduces these to 0/0 and 1/1. Empty output
and absent paths are distinct: a legal path can emit only epsilon. All measured
predictions completed without exceptions after the normalization correction.

Backoff explicitly interpolates each positional distribution with a
letter-conditioned target prior projected from q: `(1-alpha)*CART + alpha*prior`.
This changes the scoring objective; it is not an extra hidden q multiplier.
Alphas 0, .001, .01 and .1 were examined on a fixed 1,000-spelling development
sample. The smallest positive value, .001, was then used for full-development
measurement. This is development selection, not independent confirmation.

The diagnostic inventory/prior comes from a separately frozen native q model.
Prepared data, aligner implementation, convergence trace and inventory size
match the weighted teacher, but the original weighted q was not persisted, so
bitwise identity is not independently established.

## CART also collapses possibilities

With the current trainer defaults, `store_distributions=True` still permits a
leaf to collapse to its best class above 95% confidence or below 0.1 bits of
entropy. Cartlet also omits class probabilities below 1e-8 without renormalizing.
These gates remove alternatives before decoding. The lattice accepts and
normalizes small total-mass deficits; that numerical correction does not restore
omitted labels. Explicit backoff does restore some missing support.

Posterior trees also grow substantially: French hard/soft have 44,003/142,682
decision nodes, Italian 30,487/129,008. Fractional alternatives prevent many pure
leaves. This suggests testing regularization; size alone does not prove
statistical overfitting.

## Next controlled experiments

1. The [leaf-retention ablation](CART_LEAF_RETENTION_RESULTS.md) now demonstrates
   better no-backoff lattice PER/coverage with unchanged point predictions.
   Added gains over explicit backoff are small and mixed; retain that limit.
2. Test a simpler teacher/projection. Unrestricted 2:2 q creates arbitrary
   two-letter/two-phone fusions, expanding anchored target labels and ambiguity.
   A single-letter/multiple-phone teacher is a useful first ablation; a restricted
   1:M/M:1 teacher is a separate comparison. Their benefit is a hypothesis.
3. Add an output sequence or length model to reject mutually incompatible
   positional combinations, including legal all-epsilon paths. Evaluate its
   changed objective explicitly rather than equating unit legality with full
   correlation preservation.
4. Regularize posterior trees using validation grouped by spelling, preserving
   all variants within a group and avoiding positional-row leakage.

Soft EM already estimates the alignment teacher. These findings concern how its
uncertainty survives projection, tree storage and decoding; they do not falsify
averaging over complete possibilities during EM.

Private reproducibility artifacts: `/private/tmp/phonebox-decomposition-diagnostics-full`
(gold oracle and fixed-tree lattice), `/private/tmp/phonebox-cart-backoff-diagnostics`
(sample alpha sweep), and `/private/tmp/phonebox-cart-backoff-diagnostics-full`
(full-development backoff). Frozen corrected decoder: `3af505a`.
