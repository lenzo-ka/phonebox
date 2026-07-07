# G2P Benchmarks

## Measured results (this repo)

CMUdict, 1:1 `G2PDecisionTree`, **pure g2p** (no exceptions dictionary), trained
on a 133,166-entry split and evaluated on a held-out 2,000-word test, single
best pronunciation (`seed 42`). Full method and reproduce command in
[`G2P_EVAL.md`](G2P_EVAL.md#measured-cmudict-results-11-decision-tree).

| Phones | PER% | WER% | pos% |
|--------|-----:|-----:|-----:|
| with stress   | 12.96 | 53.8 | 81.8 |
| stress stripped | 11.36 | 44.9 | 81.8 |

- **PER%** phone edit rate · **WER%** whole-word error · **pos%** position
  accuracy (see [`ACCURACY.md`](ACCURACY.md) and
  [`G2P_EVAL.md`](G2P_EVAL.md#why-position-accuracy-pos)).
- These are **pure-generalization** numbers on unseen words. The production
  hybrid path (exceptions dictionary first, tree fallback) is exact on covered
  words, so effective error on real text — where most tokens are covered — is
  lower. That is coverage, not a different algorithm; see
  [`ACCURACY.md`](ACCURACY.md).

Not independently re-measured here (treat as approximate): model size and
per-word latency. Trained 1:1 English trees are typically well under 1 MB and
run in pure Python on CPU with no GPU.

## How decision-tree G2P compares

Decision-tree / CART G2P is a classic, lightweight approach. Neural
sequence-to-sequence G2P is generally **more accurate** and more robust to
spelling variation, at the cost of model size, compute, and dependencies.
Phonebox sits at the small-fast-interpretable end of that trade-off.

Published reference points (external results, **not** our measurements; metrics
and test sets differ, so they are not directly comparable to each other or to
the CMUdict PER above — context, not a leaderboard):

| System | Reported | Type |
|--------|----------|------|
| Montreal Forced Aligner (Arabic / Bulgarian) | 95.4% / 97.3% accuracy | decision tree |
| LiteG2P ([arXiv:2303.01086](https://arxiv.org/abs/2303.01086)) | "comparable to leading models" | transformer |
| r-G2P ([arXiv:2202.11194](https://arxiv.org/abs/2202.11194)) | 2.73–9.09% WER reduction | neural, robustness |
| Multilingual multi-task ([Interspeech 2018](https://www.isca-archive.org/interspeech_2018/ni18_interspeech.html)) | 97.7% syllable accuracy | neural |

## When to choose phonebox

Good fit:

- Compact deployment; fast, deterministic, CPU-only inference
- Interpretable rules you can inspect
- Zero runtime dependencies; embedded / serverless / edge
- A lexicon-backed hybrid where most real-text tokens are already covered

Prefer neural G2P when:

- Maximum accuracy on out-of-vocabulary words is the priority
- Robustness to typos / spelling variation matters
- Model size and GPU/compute are not constraints

## References

1. Montreal Forced Aligner — pretrained G2P models
2. LiteG2P — lightweight transformer (arXiv:2303.01086)
3. r-G2P — robust G2P (arXiv:2202.11194)
4. Multilingual neural multi-task G2P (Interspeech 2018)
