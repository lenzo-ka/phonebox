# Posterior decomposition scattering: interim development result

The French result does not support the proposed accuracy gain under the
prespecified unigram-q teacher and left-anchor projection. Posterior training
has slightly higher error than the matched hard-projection control, and both
are substantially worse than epsilon-CART. This is a retained negative result,
not a reason to replace the original baseline or change the ongoing protocol.

| French supervision | Development WER % | Development PER % | Admitted / skipped | Target preparation + CART fit (s) | Model bytes |
| --- | ---: | ---: | --- | ---: | ---: |
| Hard q-Viterbi projection | 31.9667 | 6.8780 | 78,588 / 18 | 177.07 | 193,144 |
| Posterior q projection | 33.3333 | 6.9885 | 78,588 / 18 | 564.04 | 3,887,623 |
| Epsilon-CART reference | 14.1221 | 2.6885 | 77,555 / 1,051 | 265.59 | 91,743 |

Posterior-minus-hard differences: **+0.1105 PER percentage points** and
**+1.3666 WER percentage points**. No statistical-significance claim is made.
Posterior preparation recorded 567,832 ambiguous admitted letter positions.
Repeated pairs retain their multiplicity. The two projected models share the
same fitted q and admission; the epsilon reference has different admission.

All three metric rows were independently recomputed from their saved
predictions on all 7,683 development spellings and matched exactly. No test
predictions were generated. Frozen source: `000eb75`. Full settings, pinned
preparation, hypothesis and interpretation limits are in
[CART_DECOMPOSITION_EXPERIMENT.md](CART_DECOMPOSITION_EXPERIMENT.md).

The French lexicon preparation excludes explicitly liaison-marked variants.
These runs provide no following-word context and do not measure contextual
liaison prediction. Unmarked pronunciation ambiguity can remain. French is
therefore a citation-form decomposition experiment, not a complete model of
French pronunciation. Identical preparation keeps the hard/posterior contrast
matched, but results should not be generalized to liaison-aware systems.

The alignment teacher itself already uses soft EM. This experiment concerns
carrying its final alignment uncertainty through CART supervision, rather than
using soft versus hard EM to estimate q. Marginal targets discard correlations
between positions; independent CART predictions may combine targets that never
co-occurred in one complete alignment. That is one plausible failure mechanism,
not yet a demonstrated cause. Teacher quality, projection policy and model
complexity remain other possible explanations.

Both CMU conditions and the Italian epsilon reference remain in progress. Their outcomes, including
losses and null results, will be appended without tuning on these measurements.
The separate lattice extension will compare coherent supported path decoding
against the existing independent-position prediction objective.

## Italian projected-target result

Italian also shows a loss from posterior training. This indicates that the
observed regression is not confined to the French condition; it does not
identify its cause. Both projected models admit the same 72,448 pairs and
skip 20. Metrics were recomputed exactly on all 7,397 development spellings.

| Italian supervision | Development WER % | Development PER % |
| --- | ---: | ---: |
| hard-q | 25.9970 | 4.0224 |
| posterior-q | 29.5796 | 4.6151 |

Posterior-minus-hard PER: +0.5927 percentage points. No significance claim or test tuning.
