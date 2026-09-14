# Development experiment: retaining decomposition uncertainty in CART

## Hypothesis and possible falsification

Retaining posterior uncertainty across gold-compatible n:m decompositions in
contextual CART training reduces held-out phone error compared with projecting
one q-Viterbi alignment. This is a hypothesis, not an established improvement
or novelty claim. A null result or regression is a reportable outcome.

The primary measure is shared any-variant PER on complete held-out development
spellings. Any-variant WER, missing predictions, training admission, target
ambiguity, runtime and model size are secondary. Report percentage-point
changes against the matched hard-projection control for each condition; retain
losses and mixed results. No new test decoding or test-based tuning.

## Declared comparison

Four unchanged substantial public benchmark train/dev splits: WikiPron French,
WikiPron Italian, CMUdict stress preserved, CMUdict stress removed. Use the
pinned preparation API and identical atomic phones/case policy. Whole spelling
groups remain in one split. Data/test hashes may be recorded for identity, but
test predictions and test metrics are not generated in this experiment.

Three contextual CART predictors, width 7, native trainer, no pruning, no
internal row split, dictionary lookup/storage disabled:

1. Epsilon-CART reference: existing epsilon alignment, cap 100, combination cap
   5,000. Its admission can differ; report that separately.
2. Hard projected n:m targets: one q-Viterbi path per training pronunciation,
   full phone group at the first consumed letter, epsilon at remaining letters.
3. Posterior projected n:m targets: forward/backward marginalization over all
   supported gold-compatible q paths with the same projection policy. Fractional
   target counts sum to one per admitted letter position.

Fit one shared joint-unit aligner per condition on training only: letter/phone
spans 2:2, min phone span 0, cap 100, convergence threshold 1e-4, default unit
pruning. Freeze q before preparing either projected control. Both projected
models must admit identical source pairs; a mismatch stops the comparison.
Preprocessing is the benchmark identity vectorizer, with static joins disabled.
Record actual preprocessing, retained support and the probability floor.

## Interpretation limits

The experiment isolates posterior versus hard target preparation under a fixed
unigram q teacher. It does not establish benefits of higher-order alignment,
new smoothing, learned input chunks, or runtime lattice decoding. CART still
predicts independently per letter; posterior target marginals lose complete-path
correlations and may produce incoherent combinations. The primary hypothesis
can fail even if soft EM improves estimation of the teacher itself.

Compute costs include shared alignment separately from target preparation and
CART fitting. Aggregate repeated pairs without changing their multiplicity.
Generated target uncertainty is distinct from a calibrated uncertainty claim.
Development findings are exploratory evidence; broad claims require further
validation. Preserve exact source/data receipts, settings, predictions and raw
logs privately, and write up both favorable and unfavorable outcomes.

## Relationship to prior work

The proposed connection is a generalization of epsilon scattering: preserve
probability over decompositions through CART supervision rather than commit to
one alignment. Existing many-to-many alignment/local-classifier methods provide
relevant context, but an exact novelty determination requires a fuller review.
See Black, Lenzo and Pagel (1998), *Issues in Building General Letter to Sound
Rules*, and Jiampojamarn, Kondrak and Sherif (2007), *Applying Many-to-Many
Alignments and Hidden Markov Models to Letter-to-Phoneme Conversion*:
<https://aclanthology.org/N07-1047/>.
