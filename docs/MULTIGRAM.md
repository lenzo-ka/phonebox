# Joint-unit language modeling

Phonebox's n:m model learns letter/phone units with EM alignment, then counts
sequences of those units on Viterbi training paths. Inference maximizes the
complete unit-LM sequence score, including the end marker. Alignment unit
probabilities supply candidate support, without another probability factor.

`MultigramLM(order=...)`, `MultigramG2P(lm_order=...)`, the public
`train_multigram` API and `phonebox train-multigram --lm-order` accept orders
1–8. Order counts the current unit: order 8 can condition on seven preceding
joint units. This differs from a letter span, a phone span, or a CART context
window. Increasing order enables controlled development-set comparisons; it
does not establish that order explains differences from other G2P systems.

## Counts and smoothing

Training stores sparse n-grams only through the requested order. Context totals
are derived from outgoing counts, rather than serialized independently. The
fixed prediction events are observed units, declared alignment units (including
silent-phone units absent from Viterbi paths), and EOS. SOS is a single initial
context marker and is never predicted. Unknown prediction units are rejected.

For an observed context, add-k smoothing uses `(count + k) / (context total +
k * number of events)`. An unseen context backs off to its shorter suffix.
The default remains `k=0.1`; extending the order does not change this smoothing
method. In particular, this is not modified Kneser–Ney smoothing. Each event
probability and the EOS probability use the same normalized event support.
Extreme finite positive `k` values use log-domain arithmetic when a direct
probability would overflow or underflow. Scores remain finite even when their
exponentiated probabilities are too small for a floating-point number; smoothing
values are not clamped and the saved count format is unchanged.

Higher orders increase the number and size of sparse count keys. Memory grows
with the distinct observed n-grams, not with a fully enumerated vocabulary
power, but sparse storage does not make a large training corpus free of cost.

## Search and reproducibility

The decoder retains histories of up to `order - 1` units at each input position.
Different segmentations and phone choices can create many histories as order
increases. `decode_beam=0` (CLI `--decode-beam 0`) preserves exact search and
remains the default. A positive integer expands only the best-scoring histories
per position; this is approximate search and can discard the globally best
pronunciation. The beam controls expanded histories, not a hard bound on all
resident candidate states or total process memory. No beam is enabled silently.

For example, train with `lm_order=8, decode_beam=64` in the API, or add
`--lm-order 8 --decode-beam 64` to `phonebox train-multigram`. Select settings on
fixed development data, preserve the chosen order and beam in exported model
metadata, and evaluate test data only after freezing the configuration.

Model version 7 / LM count format 3 serializes the requested sparse levels,
validates event/context counts on load, and retains the `unit-lm-with-eos`
objective. Earlier model layouts require retraining; see
[release migration](RELEASING.md#multigram-scoring-artifacts). Historical reports
and models keep their original source, order and search settings; adding this
capability does not rewrite their measurements.

## Repeated prediction

Prepare a reusable snapshot when predicting many words from one trained model:

```python
from phonebox import MultigramG2P

model = MultigramG2P.load("model.g2p")
predictor = model.prepare_predictor()
pronunciations = [predictor.pronounce(word) for word in words]
```

`MultigramPredictor.pronounce_letters(letters, word=...)` accepts already cooked
letter tokens, like the model method. The snapshot preserves the language model,
ordered candidate units, preprocessing, exception lookup, and decoding beam.
Later changes or retraining of `model` do not change existing predictors; call
`prepare_predictor()` again to refresh. Returned phone lists can be modified
without changing the snapshot.

Preparation copies the model's inference state once and builds the candidate
index and encoded unit IDs once. It costs additional time and memory that grow
with model size, which can be amortized over a batch; single-word model methods
continue to use current model state. The pronunciation CLI and the two-way
comparison evaluator prepare once and reuse the result for their input stream.
Preparation and prediction should be measured separately when reporting speed.
The public `joint_decode` function remains available for direct mutable inputs.

For callers already holding encoded unit IDs, `MultigramLM.log_prob_unit_id`
uses the same probability calculation and vocabulary validation as `log_prob`,
without encoding the unit again.
