# Fixed-tree CART lattice: complete development comparison

Across all four conditions, lattice decoding lowers posterior-tree variant PER relative to independent prediction. The hard trees lose PER in every condition despite lower WER, largely accompanied by many absent legal paths. This is an opt-in score/objective comparison, not a change to the original weighted-training protocol or baseline.

| Condition | Training | Decoder | Dev variant WER % | Dev variant PER % | No legal path |
| --- | --- | --- | ---: | ---: | ---: |
| french | hard-q | independent | 31.9667 | 6.8780 | — |
| french | hard-q | lattice | 26.7474 | 12.4774 | 737 |
| french | posterior-q | independent | 33.3333 | 6.9885 | — |
| french | posterior-q | lattice | 24.4436 | 5.4496 | 90 |
| italian | hard-q | independent | 25.9970 | 4.0224 | — |
| italian | hard-q | lattice | 23.0499 | 7.0157 | 378 |
| italian | posterior-q | independent | 29.5796 | 4.6151 | — |
| italian | posterior-q | lattice | 24.8209 | 3.8735 | 51 |
| cmudict-preserved | hard-q | independent | 61.1633 | 18.6094 | — |
| cmudict-preserved | hard-q | lattice | 59.1555 | 24.6275 | 1247 |
| cmudict-preserved | posterior-q | independent | 62.5506 | 19.3105 | — |
| cmudict-preserved | posterior-q | lattice | 59.6209 | 17.8996 | 107 |
| cmudict-removed | hard-q | independent | 52.8996 | 14.7028 | — |
| cmudict-removed | hard-q | lattice | 51.0987 | 20.6077 | 1118 |
| cmudict-removed | posterior-q | independent | 54.7523 | 15.2503 | — |
| cmudict-removed | posterior-q | lattice | 50.9263 | 13.9462 | 96 |

All16 saved prediction metric rows independently recomputed exactly. Runtime scoring multiplies projected per-position CART probabilities on legal joint-unit paths; no q factor, LM, beam or support floor is added. Zero-probability labels can prevent any complete path. Best-path selection is not pronunciation-level probability aggregation or recovery of full alignment correlations.

Frozen corrected source`3af505a`. Support from earlier native unit artifacts matched prepared data, aligner source, observed EM trace and inventory size; the original weighted teacher q was not persisted, so bitwise original q identity is not independently established. This limitation is retained. Raw artifacts remain private at `/private/tmp/phonebox-cart-lattice-dev-corrected-20260913`. No test prediction, significance or novelty claim.

See [the mechanism diagnosis](CART_DECOMPOSITION_DIAGNOSIS.md) and [the controlled leaf-retention result](CART_LEAF_RETENTION_RESULTS.md) for the separate coverage/backoff probes.
