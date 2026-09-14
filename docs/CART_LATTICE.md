# CART-scored decomposition lattice

Experimental, opt-in prediction following posterior decomposition training.
Reuse the learned unit inventory while selecting a legal complete path from
contextual CART target probabilities. Existing per-position pronunciation is
unchanged.

```python
lattice = result.model.prepare_decomposition_lattice()
phones = result.model.pronounce_lattice("word", lattice=lattice)
```

New posterior-trained models save supported letter/phone units, but do not save
teacher q weights for this decoder. Inventory validation and model round trips
use the existing CART formats; ordinary models need explicit units supplied to
`prepare_decomposition_lattice(units)`.

Each candidate consumes its learned letter span. Its score is the sum of log
CART probabilities for the full phone group at the first consumed letter and
epsilon at each remaining letter. A merged unit and its unmerged alternatives
therefore compete using contextual classifier evidence. The inventory is
indexed once and owned by the prepared lattice. The shared joint decoder
performs exact Viterbi search; no beam, probability floor or extra q factor is
introduced. Missing labels have zero probability. With no supported path,
`CartDecompositionLattice.decode` returns None and `pronounce_lattice` returns
an empty prediction; dictionary policy follows the model's lookup setting.

This prevents unsupported combinations of unit boundaries and targets. It does
not learn cross-position correlations, add a unit language model, or sum all
paths yielding the same pronunciation. Equivalent projected target paths can
receive identical scores; candidate order breaks ties deterministically. CART
scores are marginal probabilities, not a calibrated complete pronunciation
posterior.

Weighted training must be assessed before attributing benefits to this decoder.
Use the identical trained tree, learned inventory, preprocessing and complete
development population for independent versus lattice prediction. Report path
coverage, WER/PER, model/support size and preparation/prediction costs, retaining
missing predictions and negative outcomes. No corpus benefit is claimed yet.
French citation-form evaluation remains separate from context-aware liaison.
