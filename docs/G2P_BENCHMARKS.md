# Measured G2P comparisons

Generated from aggregate benchmark JSON. See [protocol and reproduction instructions](REPRODUCIBLE_BENCHMARKS.md).

WER accepts any reference pronunciation; PER uses the minimum-edit reference and its phone count. Missing predictions remain errors. Each condition uses identical prepared splits across systems. Scores measure held-out model predictions with dictionary lookup disabled. These are measured baseline runs, not historical-paper replications.

## CMUdict, stress preserved

Train/dev/test words: 104,447 / 11,605 / 10,000.

| System | Held-out WER (%) | Held-out PER (%) | Empty predictions | Train + export (s) | Predict (s) | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phonebox CART | 53.70 | 13.32 | 0 | 195.71 | 0.58 | 367282 |
| Phonebox n:m | 54.55 | 15.24 | 0 | 295.04 | 121.08 | 14937665 |
| Sequitur | 51.56 | 13.79 | 0 | 16307.39 | 76.33 | 1605862 |
| Phonetisaurus | 33.69 | 8.88 | 0 | 107.28 | 4.11 | 40716873 |

Not measured in this artifact: DeepPhonemizer autoregressive.

Prepared split SHA-256:

- train: `1e354b709ae151e8a2ad259e44e212c555531a443f6195a5304ba469c96d6a91`
- dev: `7e2b6b3e0efde4284806dc7876e3ec52d096ada3cfc5acba8e66ba3906897566`
- test: `356328cf3a8b87ce01cc087c0f129f6eff41b2ebddce6351a6fb64f862b86dfc`

## CMUdict, stress removed

Train/dev/test words: 104,447 / 11,605 / 10,000.

| System | Held-out WER (%) | Held-out PER (%) | Empty predictions | Train + export (s) | Predict (s) | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phonebox CART | 42.48 | 9.76 | 0 | 188.05 | 0.57 | 277800 |
| Phonebox n:m | 43.33 | 11.17 | 0 | 253.28 | 53.41 | 12473370 |
| Sequitur | 41.67 | 10.40 | 0 | 15602.03 | 30.52 | 1068896 |
| Phonetisaurus | 25.86 | 6.40 | 0 | 134.93 | 3.02 | 36904262 |

Not measured in this artifact: DeepPhonemizer autoregressive.

Prepared split SHA-256:

- train: `46f838a971dc02f9218655f3a49c47157baf726f6104e59ebfcfcdef7d64b372`
- dev: `12832f603f88bd3a2e9f617d31903bba8c95823e4075f5973866d1ed4cce420f`
- test: `f51ca936430c9dd430fb7b62d73ee793a6b243b99e2b9f4b1ee6204cb5ed8636`

## French

Train/dev/test words: 69,106 / 7,683 / 8,525.

| System | Held-out WER (%) | Held-out PER (%) | Empty predictions | Train + export (s) | Predict (s) | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phonebox CART | 14.21 | 2.64 | 0 | 247.70 | 0.54 | 91743 |
| Phonebox n:m | 14.10 | 2.88 | 9 | 272.40 | 20.10 | 7365562 |
| Sequitur | 9.72 | 1.85 | 0 | 7169.42 | 29.60 | 655007 |
| Phonetisaurus | 6.38 | 1.26 | 0 | 91.44 | 2.69 | 24186353 |

Not measured in this artifact: DeepPhonemizer autoregressive.

Prepared split SHA-256:

- train: `54b3f98964fd02918d97037b9fa8fef8ec22fdc45ac09b0fea228df13de854c9`
- dev: `d096ca206e6cd3c23f3c0cfa43426c7e0243a11bce4d735ae50046709e126a94`
- test: `7a8ff9bb882f43fbc5d0751d24fce8d5ee6bf8cacbc288729577f5e809cf0c69`

## Italian

Train/dev/test words: 66,573 / 7,397 / 8,222.

| System | Held-out WER (%) | Held-out PER (%) | Empty predictions | Train + export (s) | Predict (s) | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phonebox CART | 16.91 | 2.28 | 0 | 114.82 | 0.46 | 59229 |
| Phonebox n:m | 21.80 | 2.97 | 2 | 240.53 | 9.79 | 5646436 |
| Sequitur | 21.60 | 2.84 | 1 | 8223.61 | 9.57 | 322640 |
| Phonetisaurus | 7.81 | 1.06 | 1 | 215.84 | 4.19 | 20333774 |

Not measured in this artifact: DeepPhonemizer autoregressive.

Prepared split SHA-256:

- train: `b16886176309e1712570fd7a7fdbcfb9d2eae602043a0a14611a6225373e0055`
- dev: `568af51511d5b59b995fc9643121eb958799defd9336a2ac2d589c5623cec13c`
- test: `e4ca4ed19aaa2ab408199a6fb7acb5d87f5142cb104601e08630ba448d632f40`

## Scope

Full configuration, source/dependency provenance, training admission counts, and error accounting are retained in the accompanying JSON. Train + export includes the recorded training procedure (including development selection where used) and export; external training already includes export. Timings across different shared machines are descriptive, not a speed ranking. These are lexicon-type evaluations, not running-text accuracy measurements.
