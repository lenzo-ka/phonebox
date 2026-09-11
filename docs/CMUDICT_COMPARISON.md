# CART and multigram G2P on CMUdict

> These measurements are a reproducible snapshot, not a promise about future releases. Run the refresh command below after implementation changes.

Phonebox exposes two trainable grapheme-to-phoneme models. `G2PDecisionTree` aligns one cooked grapheme position to one target position and learns CART decisions from spelling context. `MultigramG2P` learns joint n:m grapheme/phone units and decodes unit sequences with a language model. Both use `Vectorizer` for the same locale normalization, stress policy, and configured joins in this benchmark.

## Reproduce

```console
python -m pip install -e '.[dev]'
phonebox compare cmudict --refresh docs/cmudict-comparison.json
phonebox compare cmudict --check docs/cmudict-comparison.json docs/CMUDICT_COMPARISON.md
```

Data: [CMUdict](https://github.com/cmusphinx/cmudict) commit `74790861f652b15e4ac49015a90074ad62a27690`, `cmudict.dict` SHA-256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`. Its [license](https://github.com/cmusphinx/cmudict/blob/74790861f652b15e4ac49015a90074ad62a27690/LICENSE) permits research and commercial use. CMUdict is Copyright Carnegie Mellon University, which requests acknowledgement of its origin.

Split: seed 1729, 10% held out, capped at 10000 normalized word identities. All variants of a cooked spelling remain on one side. Identical pronunciations after the selected phone mapping are deduplicated. Exceptions are disabled.
The CART row uses the shared native, serial, unpruned benchmark helper. This is an explicit evaluation setting; the primary training workflow currently prunes by default. Exact helper behavior belongs to the recorded Phonebox source revision.

WER is error against the first deterministic gold variant; WERv accepts any gold variant. PER uses the first variant and divides edits by its phone count. PERv selects the gold variant with the fewest phone edits for each word (ties use lexical phone order), then divides total edits by total phones in those references. It may exceed 100% when insertions outnumber reference phones. The policy is identical for both models. Artifact size includes every file required to reload the exported model.
Training time includes each model's phone cooking, alignment, and fit. The CART export is gzip-compressed while multigram uses JSON sidecars, so size is the current complete export footprint rather than normalized complexity. Prediction errors and empty outputs are counted explicitly.

## Results

| Stress | Model | Train words | Test words | Train s | Size bytes | WER% | WERv% | PER% | PERv% | Errors | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| preserved | G2PDecisionTree | 116052 | 10000 | 283.29 | 391037 | 54.23 | 52.62 | 13.77 | 13.23 | 0 | 0 |
| preserved | MultigramG2P | 116052 | 10000 | 227.62 | 17116009 | 65.61 | 64.32 | 20.75 | 20.27 | 0 | 0 |
| removed | G2PDecisionTree | 116052 | 10000 | 252.64 | 312958 | 43.72 | 41.66 | 11.50 | 10.86 | 0 | 0 |
| removed | MultigramG2P | 116052 | 10000 | 214.90 | 14382955 | 67.57 | 66.41 | 26.30 | 25.71 | 0 | 0 |

## Interpretation

- With stress preserved, G2PDecisionTree has the lower variant-aware phone error (13.23%).
- With stress removed, G2PDecisionTree has the lower variant-aware phone error (10.86%).
- Variant-aware WER/PER credit alternate held-out pronunciations; compare them within the same stress condition.
- Stress-preserved and stress-removed rows are different prediction tasks; their error rates are not direct measures of one task improving.
- Training times are one observed run on the recorded environment, not a throughput guarantee.
- Export byte totals reflect the current gzip CART and JSON-sidecar multigram formats, not normalized algorithm complexity.

## Snapshot provenance

- Phonebox revision: `78485ef21a6a92629773d07c5ee8ff925f680c99` (clean)
- Workload code SHA-256: `2af02335711e79446ad5d47c31fdd3dc97afa33a52815d6be1553f9ea1b17e56`
- Python: CPython 3.12.12
- Platform: macOS-27.0-arm64-arm-64bit
- Dependencies: {'cartlet': '0.5.0', 'icukit': '0.4.0'}
- Multigram EM iterations: 10; spans 2:2
