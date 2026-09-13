# G2P methods in the reproducible comparison

This guide describes the four implementations used by the
[benchmark protocol](REPRODUCIBLE_BENCHMARKS.md). It separates their modeling
approaches from this comparison's small, fixed configuration choices. Identical
prepared inputs and held-out references make the task comparable; they do not
make the models, training budgets, or searches equivalent. These runs are
baseline observations, not a state-of-the-art claim or a reproduction of a
published paper's percentages.

## Overview

| System | Alignment and training units | Prediction model | How a pronunciation is chosen |
|---|---|---|---|
| Phonebox CART | Each cooked letter position receives one phone label or epsilon; a label can represent an explicitly joined phone sequence | Entropy-based classification tree over a letter context window | Predict one label per position, concatenate emissions, remove epsilon and split explicit joins |
| Phonebox multigram | Joint units pair a nonempty letter span with a phone span, including an empty phone side | EM unit probabilities plus an n-gram model over Viterbi-aligned joint units | Dynamic programming chooses a complete matching letter segmentation and its phone sequence |
| Sequitur G2P | Joint-sequence units; this benchmark inherits one-symbol pairs and input/output epsilon units | Discounted joint-sequence model with increasing history order | The authors' translator searches for the best output sequence for the spelling |
| Phonetisaurus | Joint alignment units, then an n-gram corpus of aligned joint symbols | MITLM backoff n-gram model converted to a weighted finite-state transducer | The authors' decoder searches the WFST with the input spelling |

“Joint” means that a model unit represents both an orthographic subsequence and
its pronunciation subsequence. A silent letter may therefore be represented
by an empty phone side rather than by a phonetic symbol. Epsilon and separators
are implementation markers, not additional reference phones.

## Phonebox CART

Phonebox's [`EMAlign`](../phonebox/core/em_align.py) enumerates placements of
silent-letter labels while preserving the order of the supplied phone labels.
It alternates choosing a best alignment under a letter-to-label model and
re-estimating that model. This is a hard-alignment procedure, rather than the
full forward-backward expectation calculation used by the multigram aligner.
The alignment model uses count smoothing and a silence bias for apostrophes
and hyphens.

[`G2PDecisionTree`](../phonebox/core/g2p_model.py) converts the resulting
alignments to context-window classification examples and trains Cartlet's
native entropy-based tree. The tree learns letter-context decisions; it is
not a language model over output pronunciation history. Prediction emits a
label at each cooked letter position. The vectorizer's uncooking operation
removes epsilon and expands explicitly joined phone labels. Saved vocabulary
checks can also cause an unsupported context to emit epsilon.

In the general library workflow, configured phone joins can make a multi-phone
emission one training label. The shared-identity benchmark deliberately applies
no joins. Consequently entries with more phone tokens than letter positions
cannot be admitted by this aligner. It also rejects candidates whose number
of epsilon-placement combinations exceeds the configured limit. These are
model-specific admission limits: the shared dataset retains such candidates,
and results report actual retained and skipped entries.

The benchmark uses width **7**, at most **10** alignment iterations,
`max_combinations=5000`, native tree training, and **no pruning**. This is a
fixed baseline, not the default pruned primary training workflow and not a
search for the best context width or tree settings. See the
[`benchmark adapter`](../phonebox/eval/benchmark.py) and each result's settings
for the executable configuration.

## Phonebox multigram

[`MultigramAligner`](../phonebox/core/multigram_align.py) estimates joint-unit
probabilities with forward-backward EM over possible alignments. Its letter
side consumes at least one token; the phone side may be empty. After fitting,
training pairs are Viterbi-aligned into joint-unit sequences. Only successfully
aligned paths train the subsequent unit language model.

[`MultigramLM`](../phonebox/core/multigram_lm.py) counts unit n-grams and uses
add-k smoothing, with **k=0.1** in the current implementation. When a context
has not been observed, it falls back to a shorter context. This is not the
same smoothing procedure as Sequitur's discount adjustment or MITLM's
modified Kneser–Ney estimator.

[`joint_decode`](../phonebox/core/joint_decode.py) keeps hypotheses by input
position and joint-unit history. Each transition must match the next letter
span and adds the log unit probability and the unit LM log score. It chooses
a complete path rather than greedily emitting a phone for each letter. A
word with no complete segmentation in the saved unit inventory returns an
empty prediction; occurrences of its individual letters in training do not
guarantee a composable unit path.

The benchmark caps letter and phone spans at **2**, allows **0** phones per
unit, runs at most **10** EM iterations, and uses a **bigram** unit LM.
Current inherited defaults are convergence threshold `1e-4`, minimum unit
mass `1e-8`, and decode beam **0**: no beam truncation. These choices are
recordable implementation settings, not properties required by the general
n:m approach. The native LM supports orders 1–3; this baseline does not tune
that order on the test set.

## Sequitur G2P

Sequitur implements the joint-sequence approach of **Bisani and Ney (2008)**.
The benchmark invokes the authors' program as a separate executable; Phonebox
does not reimplement its estimator or decoder. At the pinned source revision,
[`ModelTemplate`](https://github.com/sequitur-g2p/sequitur-g2p/blob/7bd56d5d502325e0be3f14b7d898720a39db3338/sequitur.py)
defaults to size templates **(1,1), (1,0), (0,1)**. The adapter supplies no size
override. Thus this configuration uses single-symbol pairing, deletion, and
insertion units, rather than inheriting Phonebox's 2-token span caps.

Training uses the upstream EM procedure, with no `--viterbi` approximation
switch. Because an explicit development file is supplied,
[`SequiturTool`](https://github.com/sequitur-g2p/sequitur-g2p/blob/7bd56d5d502325e0be3f14b7d898720a39db3338/SequiturTool.py)
selects its default discount adjuster. That adjuster optimizes discounts using
development log likelihood; it is not a fixed add-k constant.

The adapter trains orders **1, 2, and 3** in succession, ramping up the previous
model. Each stage uses UTF-8, minimum **1** and maximum **10** iterations.
The selected stage minimizes development PER, then development WER, then
order. Upstream development-likelihood model selection within training is
separate from that adapter-level choice. Test references participate in
neither selection. Prediction uses the authors' normal one-best translator;
no custom search-stack limit is supplied.

These restricted orders and iteration limits are declared turnaround choices,
not the highest configuration attainable by Sequitur or a match to the
original paper's experimental protocol.

## Phonetisaurus

Phonetisaurus implements the joint n-gram WFST approach described by **Novak,
Minematsu, and Hirose (2016)**. The adapter uses the authors' alignment and
decoder tools, **MITLM** for n-gram estimation, and **OpenFst** for WFST
infrastructure. No SRILM installation is required.

The pipeline is alignment to joint-symbol sequences, n-gram estimation to
ARPA, then ARPA-to-WFST conversion. The spelling and phone token boundaries
from shared preparation are preserved. The WFST represents the learned
joint-symbol language model; it must not be described as universally storing
the complete training dictionary.

The benchmark sets span caps **2/2**, `seq1_del=false`, `seq2_del=true`,
alignment iterations **11**, `restrict=true`, and `grow=false`. The pinned
[`aligner flags`](https://github.com/AdolfVonKleist/Phonetisaurus/blob/f08d3dfb10b8d619e665a9581d2a327bcc2504f7/src/bin/phonetisaurus-align.cc)
define restriction as initializing **M-to-1 and 1-to-N** links. Span caps
therefore do not imply unrestricted 2-to-2 initialization. Score penalization
is enabled and penalization during EM is disabled; the inherited EM
termination threshold is `1e-10`.

The fixed joint-symbol LM order is **8**. The adapter supplies no smoothing
override; pinned
[`estimate-ngram`](https://github.com/mitlm/mitlm/blob/553edca763a8e142edd8ef6d51404bbf43b79c95/src/estimate-ngram.cpp)
defaults to **ModKN**, modified Kneser–Ney smoothing, producing a backoff
n-gram model. This order is not numerically equivalent to a CART window or
a Phonebox span cap, and the benchmark does not search a full LM-order grid.

The model-only `phonetisaurus-g2pfst` invocation requests one hypothesis,
beam **10000**, comparison threshold **99**, probability-mass setting **0**,
and no printed scores. Alignment, estimation, and WFST conversion all count
toward model-production time. The final WFST is the measured model artifact.
Tool availability, build provenance, and successful command/output checks
are required separately from this description of the source contract.

## Held-out models and deployment dictionaries

The comparison disables dictionary lookup and excludes Phonebox exceptions
from exported model artifacts. All WER/PER results concern disjoint held-out
spellings. Missing paths or predictions remain in the evaluation population.

Deployment can instead use dictionary corrections. CART's automatic exceptions
are based on admitted alignment data: incorrectly predicted single-reference
words and multiple-reference words receive a selected reference closest to
the model output. This is neither complete input memorization nor preservation
of every alternative or necessarily the first variant. Correctly predicted
single-reference words need no stored correction. Lookup keys depend on saved
preprocessing. The multigram model accepts an explicit exceptions dictionary
but does not automatically memorize the whole training lexicon. The
Phonetisaurus application wrapper can substitute dictionary pronunciations;
the benchmark uses the direct model-only decoder and supplies no such lexicon.

Dictionary-backed running-text accuracy additionally depends on token coverage,
the covered-word reference policy, and pronunciation context. It cannot be
inferred directly from this held-out dictionary-type comparison. See the
protocol's coverage discussion for that distinction and the timing caveat:
runs on different platforms do not establish a controlled speed ranking.

## Neural approaches beyond this classical matrix

The four-system matrix does not cover the full family of trainable neural
G2P methods. Encoder–decoder sequence-to-sequence models encode a spelling
and generate a phone sequence conditioned on that input and previously emitted
phones. Recurrent implementations can use learned attention instead of fixed
precomputed alignment; alignment-aware neural variants also exist.
[Yao and Zweig (2015)](https://www.microsoft.com/en-us/research/publication/sequence-to-sequence-neural-net-models-for-grapheme-to-phoneme-conversion/)
study both sequence generation and alignment-informed recurrent approaches.

Transformers replace recurrent sequence processing with attention-based
representations. Their character-level G2P application is studied by
[Wu et al. (2021)](https://aclanthology.org/2021.eacl-main.163/).
Neural edit/transduction models instead learn weights or probabilities for
operations relating input and output strings; an example evaluated on G2P is
[Neural String Edit Distance (2022)](https://aclanthology.org/2022.spnlp-1.6/).
These are distinct modeling families, not synonyms for a particular installed
tool or evidence that one family wins on every dictionary.

No measured neural result is supplied by the classical adapters described
above. A runnable neural comparison must separately identify the authors'
implementation and license, freeze training and decoding settings, train on
the same prepared input without outside pretrained pronunciation data, and
use the same held-out metrics. Development selection and accelerator/software
provenance must be reported. Architecture citations alone cannot substitute
for that experiment.

### Runnable neural protocol: DeepPhonemizer

The optional [DeepPhonemizer author implementation](https://github.com/axelspringer/DeepPhonemizer/tree/5dce7e27556aef4426f5623baf6351d266a30a73)
uses an autoregressive Transformer: character encoder, phoneme-token decoder,
and teacher-forced cross-entropy training, without a separately extracted joint
alignment. The accepted configuration has four encoder and four decoder layers,
width512, feed-forward1024, four heads and dropout0.1; greedy decoding stops on
the end token or the upstream100-step cap. Train-only symbol vocabularies retain
atomic phones and original spelling case. Whole words with unseen text symbols
are explicit prediction failures; unknown dev phone targets fail preflight.

From-scratch Adam training uses learning rate0.0001, warmup10000 updates, batch32
and max500 epochs. Full shared dev PER, then WER, selects the earliest best
checkpoint each epoch. Plateau learning rate halves with patience10; dev early
stopping requires30 nonimproving observations after warmup. The author CTC
alternative is not the selected protocol because fixed character repetition
cannot represent some supplied training targets. No gold target is shortened
or discarded to satisfy that constraint.

The [isolated toolchain, patch, dependency lock and receipt recipe](BENCHMARK_TOOLCHAINS.md#optional-neural-toolchain-deepphonemizer)
identify the external MIT code and narrow training/device corrections. Dictionary
lookup and optimizer payload are absent from the measured inference artifact.
MPS is seeded but not bit deterministic; platform/device and complete training
and development costs are reported. A tiny compatibility test or resource
profile supplies no neural accuracy result. Published performance reproduction,
exhaustive neural tuning, and superiority over the classical methods are not
claimed by this protocol.

## Sources, citations, and reproducibility

* Bisani, M., and Ney, H. (2008). *Joint-sequence models for grapheme-to-phoneme
  conversion*. Speech Communication, 50, 434–451.
  [DOI](https://doi.org/10.1016/j.specom.2008.01.002).
  [Authors' implementation, pinned revision](https://github.com/sequitur-g2p/sequitur-g2p/tree/7bd56d5d502325e0be3f14b7d898720a39db3338).
* Novak, J. R., Minematsu, N., and Hirose, K. (2016). *Phonetisaurus: Exploring
  grapheme-to-phoneme conversion with joint n-gram models in the WFST framework*.
  Natural Language Engineering, 22(6), 907–938.
  [Publisher article](https://www.cambridge.org/core/journals/natural-language-engineering/article/phonetisaurus-exploring-graphemetophoneme-conversion-with-joint-ngram-models-in-the-wfst-framework/F1160C3866842F0B707924EB30B8E753).
  [Authors' implementation, pinned revision](https://github.com/AdolfVonKleist/Phonetisaurus/tree/f08d3dfb10b8d619e665a9581d2a327bcc2504f7).
* [MITLM pinned source](https://github.com/mitlm/mitlm/tree/553edca763a8e142edd8ef6d51404bbf43b79c95)
  and [OpenFst project](https://www.openfst.org/) document the separate estimator
  and WFST dependencies. Build instructions and license facts are in
  [toolchains](BENCHMARK_TOOLCHAINS.md) and
  [third-party notices](../THIRD_PARTY_NOTICES.md).

Phonebox implementation links above describe the current library, not a claim
that every general modeling approach is original to this project. Each measured
JSON receipt records its actual prepared splits, settings, source fingerprint,
dependency/tool identities, training admission, timings, and held-out counts.
Those receipts and the protocol define the experiment more precisely than a
method name alone.
