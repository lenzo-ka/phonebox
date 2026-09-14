# Posterior decomposition scattering for CART

Experimental training method: generalize epsilon scattering to learned joint
letter/phone units, retaining uncertainty across supported gold alignments.

For `ab → A B`, both `a → A; b → B` and `ab → A B` can contribute. Each path
is weighted by its alignment-q likelihood conditional on the complete spelling
and gold phones. Forward/backward inference sums all supported paths without
listing them. A unit emits its full phone group at its first letter position;
remaining letters emit epsilon. Posterior target mass sums to one at each
letter position, so longer decompositions do not give a word extra weight.
Repeated source pairs keep their multiplicity. Their posteriors are computed
once per distinct pair, and feature contexts are reused across pronunciations
of the same spelling. Identical projected target
sequences accumulate their probability mass.

The normal Cartlet trainer consumes fractional weights on ordinary contextual
letter features. This is n:m alignment supervision with a CART predictor,
not a joint-unit language model or a lattice decoder. Runtime CART predictions
remain independent across positions and can combine targets that do not form
one coherent alignment. A sequence decoder is a separate extension.

```python
from phonebox import train_g2p

result = train_g2p(
    "words.dict", locale="fr", output="model.g2p.gz",
    alignment_method="decomposition-posterior", prune=False,
    decomposition_iterations=100, max_letter_span=2, max_phone_span=2,
    use_dict_fallback=False,
)
print(result.metrics["decomposition"])
```

```sh
phonebox train --locale fr --lexicon words.dict -o model.g2p.gz \
  --alignment-method decomposition-posterior --no-prune \
  --decomposition-iterations 100 --max-letter-span 2 --max-phone-span 2
```

The public workflow disables static locale letter/phone joins so both merged
and unmerged learned alternatives remain possible. Locale transliteration and
requested stress normalization still apply and are saved for prediction.
Raw reserved phone join/epsilon syntax is rejected. The cooked-pair library
API is `G2PDecisionTree.train_decomposition_from_pairs`; callers may supply
an already fitted aligner and must use matching preprocessing. Preparation is
also available through `prepare_decomposition_vectors` in the decomposition
module. Unsupported pairs are skipped with explicit reason counts. Dictionary
corrections use admitted source pairs rather than a fabricated epsilon history.

Split complete spelling groups before alignment. Internal pruning/test row
splits are disabled because posterior alternatives for one spelling must stay
together. Epsilon alignment checkpoints are unavailable for this method.
Admission, ambiguity, unit support, preprocessing and observed alignment
history are recorded in metrics and exported training metadata. Runtime
formats and the default epsilon training method are unchanged.

The posterior is over support retained after EM pruning and its probability
floor, within the declared span limits. It cannot resurrect removed units.
The unigram q teacher does not use joint-LM history; changing downstream LM
order would not change these posteriors. No corpus accuracy improvement or
novelty claim has been established. Development-only comparisons must precede
any new test evaluation.
