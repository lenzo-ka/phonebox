# Library and CLI workflows

Library functions accept Python arguments and return reusable results. Installed
commands parse paths/options and present those results. Use `phonebox <command>
--help` for the option catalog; APIs below are the corresponding workflow owners.

| Operation | Public Python entry point | Installed command |
| --- | --- | --- |
| Fetch/process dictionaries | `phonebox.Dictionary.fetch`, `Dictionary.process` | `phonebox dict fetch`, `dict process` |
| Train from a dictionary | `phonebox.train_g2p`, `train_g2p_from_config`, `G2P.train` | `phonebox train` |
| Train n:m model | `phonebox.train_multigram` | `phonebox train-multigram` |
| Discover spelling joins | `phonebox.discover_joins` | `phonebox suggest-joins` |
| Validate a supplied inventory | `phonebox.validation.validate_lexicon_file` | `phonebox check` |
| Load/predict with a tree | `phonebox.G2P` | `phonebox pronounce` |
| Load/predict with n:m units | `phonebox.MultigramG2P.load`, `pronounce` | `phonebox pronounce` (sidecar detection) |
| Bundle a tree | `phonebox.bundler.bundle_g2p` | `phonebox bundle` |
| Compare models | `phonebox.eval.g2p_compare.run_compare`, `g2p_compare_all.run_compare_all` | `phonebox compare locale`, `compare all` |
| Sweep/inspect units | `phonebox.eval.g2p_sweep.run_g2p_sweep`, `phonebox.eval.multigram_units.analyze_multigram_units` | `phonebox compare sweep`, `compare units` |
| Dictionary accuracy | `phonebox.eval.accuracy.evaluate_accuracy` | `phonebox compare accuracy` |
| Review/reorder lexicon variants | `phonebox.pronunciation_analysis.review_lexicon`, `review_lexicon_file` | `phonebox dict review` |
| Score/triage candidates | `phonebox.pronunciation_analysis.score_entries`, `triage_entries` | `phonebox score-prons`, `find-suspicious` |
| Read orthographic exemplars | `phonebox.exemplars.get_exemplars` | Python API |
| Generate pinned exemplar data | `phonebox.dev.exemplars.generate_exemplars` | `phonebox exemplars generate` (development extra) |

## Primary and prepared training

`train_g2p` returns a `TrainingResult` containing the trained model and workflow
metrics; `G2P.train` returns a `G2P` predictor. Primary dictionary training defaults
to IPA-tagged phones, preserves stress, and prunes using 5% validation rows.
Select `phoneset="cmu"` for CMU stress semantics and `remove_stress=True` only
when removal is wanted. Validation used for pruning is distinct from test rows;
request `test_split` explicitly for a held-out test result.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from phonebox import train_g2p

with TemporaryDirectory() as directory:
    lexicon = Path(directory) / "tiny.dict"
    lexicon.write_text("cat K AE1 T\ndog D AO1 G\n", encoding="utf-8")
    result = train_g2p(lexicon, locale="en", phoneset="cmu", prune=False)
    print(result.model.pronounce("cat"))
```

Prepared input is deliberate lower-level plumbing: `EMAlign` and `Vectorizer`
produce alignments/vectors. `DecisionTree.load_alignments` consumes alignments;
for vectors, use `Vectorizer.load_vectors_file` and `parse_vectors_to_data`, then
`DecisionTree.load_vectors_data(X, y, counts)` and `train` (`phonebox align`,
`vectorize`, `model train`). Prepared
training retains CMU/unpruned defaults. Keep locale, phoneset, stress policy,
and odd context width consistent between stages. Width defaults to 7; pass a
nondefault vector width to `model train` as well as the vector-producing stage.
Prepared files do not automatically carry the full primary workflow's settings.

`train_multigram` returns a `MultigramTrainingResult` with model, training metrics,
effective spans, and exported sidecar paths when output is requested. Without
output, paths are `None`. `discover_joins` returns candidate/settings/history
results; it discovers joins from raw spelling rather than model-cooked spelling.
Neither workflow invents a pronunciation inventory or fetches training data.

## Validation and text processing

```python
from phonebox.validation import validate_lexicon, format_lexicon_validation

result = validate_lexicon(["cat K AE1 T\n", "cat(2) K AE2 T\n"],
                          ["K", "AE1", "AE2", "T"])
assert not result.has_errors(strict=True)
print(format_lexicon_validation(result))
```

Validation uses the shared dictionary comment/variant parser, preserves raw phone
forms, and reports NFC-equivalent mismatches, missing/unused phones, and exact
counts. Normalization mismatches fail; missing phones warn unless strict checking
is requested. It does not rewrite the lexicon or strip stress.

`phonebox normalize` previews model-independent text tokenization. It does not
promise to show a saved model's letter preprocessing. Model prediction replays
saved letter rules and joins; `Vectorizer` is the lower-level cooking API.
Orthographic exemplar membership is separate from explicit locale spelling-rule
policy. Standalone bundling supports trees with supported portable preprocessing;
n:m inference uses the full library.

[Lexicon review](LEXICON_REVIEW.md) accepts dictionary lines directly, preserves
all source origins after effective-phone deduplication, and assigns full-population
ranks before filtering. JSONL candidate scoring is a separate input adapter.
