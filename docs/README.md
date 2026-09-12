# Phonebox guides

Start with the [project README](../README.md) and [quick start](../QUICKSTART.md).
Full-package installation depends on available ICU backend wheels; see
[installation scope and 0.2.0 migration](RELEASING.md). Exported CART bundles
provide a separate standard-library-only inference path.

| Task | Guide |
| --- | --- |
| Choose the library API or installed command | [Library and CLI workflows](WORKFLOWS.md) |
| Obtain training data and annotate French liaison | [Dictionary data](DATA.md) |
| Map phones and deduplicate pronunciation variants | [Dictionary processing](DICTIONARY_PROCESSING.md) |
| Review and reorder a lexicon using saved model scores | [Lexicon review](LEXICON_REVIEW.md) |
| Understand ordered sequence scores | [Performance and scoring](PERFORMANCE_AND_SCORING.md) |
| Use confidence and pronunciation alternatives | [N-best usage](NBEST_USAGE.md), [quick reference](NBEST_QUICKREF.md) |
| Deploy a tree without runtime dependencies | [Bundling](BUNDLING.md), [deployment](DEPLOYMENT.md) |
| Read or regenerate orthographic locale inventories | [ICU exemplars](EXEMPLARS.md) |
| Record dictionary identity and external provenance | [Content identifiers](DICT_VERSION_SCHEMA.md), [versioning examples](VERSIONING_EXAMPLES.md) |
| Run model evaluations | [Evaluation commands](G2P_EVAL.md), [accuracy interpretation](ACCURACY.md) |
| Reproduce the public CMUdict comparison | [CMUdict comparison](CMUDICT_COMPARISON.md) |
| Understand package organization | [Architecture](ARCHITECTURE.md) |
| Prepare an authorized release | [Release procedure](RELEASING.md) |

## Recorded experiments

The [older comparison](G2P_COMPARE.md), [baseline](G2P_COMPARE_BASELINE.md),
[no-join comparison](G2P_COMPARE_NO_JOINS.md), [span sweep](G2P_SWEEP.md),
[unit report](G2P_UNITS.md), and [normalization experiments](experiments/README.md)
are historical records. Their metrics apply to their original data, settings,
and code, not automatically to the current release. The locale experiments
require caller-supplied dictionaries and, where specified, trained models;
Phonebox does not package those inputs. Use the public CMUdict workflow for
an independently downloadable comparison and record a new snapshot when rerunning.
