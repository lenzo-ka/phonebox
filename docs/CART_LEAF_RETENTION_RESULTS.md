# Retaining CART leaf alternatives: controlled development result

Disabling confidence/entropy leaf collapse improves lattice PER and path coverage in both French and Italian, while all independent-position predictions remain identical. This directly demonstrates that stored alternatives can help the decoder without changing point predictions. With explicit .001 backoff already present, the added gain is tiny in French and slightly negative in Italian.

| Language | Leaf gates | Decoder | Dev variant WER % | Dev variant PER % | No legal path |
| --- | --- | --- | ---: | ---: | ---: |
| french | default | independent | 33.3333 | 6.9885 | 0 |
| french | default | lattice | 24.4436 | 5.4496 | 90 |
| french | default | lattice-backoff-.001 | 24.1963 | 4.7463 | 0 |
| french | retained | independent | 33.3333 | 6.9885 | 0 |
| french | retained | lattice | 24.1442 | 4.8894 | 33 |
| french | retained | lattice-backoff-.001 | 24.1442 | 4.7349 | 0 |
| italian | default | independent | 29.5796 | 4.6151 | 0 |
| italian | default | lattice | 24.8209 | 3.8735 | 51 |
| italian | default | lattice-backoff-.001 | 24.6451 | 3.4988 | 1 |
| italian | retained | independent | 29.5796 | 4.6151 | 0 |
| italian | retained | lattice | 24.6992 | 3.6221 | 20 |
| italian | retained | lattice-backoff-.001 | 24.6722 | 3.5033 | 1 |

Primary retained-minus-default no-backoff lattice PER: French **-0.5602** percentage points, Italian **-0.2513**. No legal path counts fall from 90 to 33 and 51 to 20. A valid empty-emission path remains distinct from an absent path. Backoff comparisons: French **-0.0114** points, Italian **+0.0046**; retain the mixed result. Epsilon-CART remains better (French 2.6885%, Italian 2.2452%); no test-generalization or significance claim.

## Controlled change

Default gates: `min_confidence=.95`, `min_dist_entropy=.1` bits. Retained gates: `min_confidence=1`, `min_dist_entropy=0`. Pure leaves can still be deterministic, and Cartlet still omits probabilities below 1e-8. This is not preservation of every tiny nonzero class.

Fit one 2:2 q teacher per language, cap 100, and prepare fractional posterior rows once. Both trees reuse identical X/y/counts (verified prepared SHA), width 7/native, no pruning/internal row split/dictionary lookup. Original teacher trace matched exactly; refitted default control reproduced the original independent metrics before fitting the retained tree. Within this ablation both trees use the same directly persisted q and support. Independent saved prediction dictionaries for default/retained also match exactly, not only their aggregate metrics.

Backoff is a separate score: `(1-alpha)*normalized_CART + alpha*letter_conditioned_projected_q_prior`, with alpha .001 fixed from the earlier development probe. There was no new alpha search. Unit lattice legality alone does not recover full training-path correlations.

All 12 saved prediction metric rows independently recomputed exactly on full development splits. Source frozen `dd1d45d`; the protocol and raw artifacts remain private. The original 12-row weighted and 16-row lattice comparisons were verified too: 40 rows total. The verification receipt remains private.

The result supports preserving uncertain leaves for this decoder. It does not isolate teacher errors, establish posterior training always beats hard training, or show that larger trees are calibrated. Next experiments should separately test simpler teacher spans and sequence-dependent scoring.
