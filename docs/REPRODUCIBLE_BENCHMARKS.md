# Reproducing G2P comparisons

The developer benchmark compares Phonebox's traditional CART model and n-to-m
multigram model with the authors' Sequitur and Phonetisaurus implementations.
Each system receives the same prepared training, development, and test data.
Results describe the recorded source revisions and settings; they are not a
claim of state-of-the-art performance. The [methods guide](G2P_METHODS.md)
explains each alignment, learning, decoding, and dictionary-lookup approach.

## Data and protocol

CMUdict is identified by a Git commit and file digest, rather than a package
release number. Its pronunciation variants remain in the same spelling group
when splitting. Stress-preserved and stress-removed experiments are separate
conditions; duplicate pronunciations are removed after the stress mapping.
The primary matrix has two stress conditions where the source supplies stress:
original labels preserved, and all stress removed. For CMUdict, preserved means
keeping its `0`, `1`, and `2` labels distinctly; stripped removes all three from
phone labels. This covers stress-sensitive output, such as TTS, and stress-free
output often used for ASR without multiplying the comparison into a factorial
study. Secondary-to-unstressed or secondary-to-primary mappings can be explored
separately through dictionary phone mapping, with the exact transform recorded
and duplicate variants removed afterward. They are not extra default rows.

The selected WikiPron snapshots already remove stress upstream. Their results
are therefore stress-free IPA observations, not a claimed stress-preserved
Italian/French condition. Missing source stress cannot be reconstructed by
renaming an experimental setting or inferring it from spelling.

This comparison reserves development data as well as test data and preserves
unjoined target phone tokens. The [archived two-way CMUdict snapshot](archive/cmudict-489097f/README.md)
used locale-aware joins and a train/test-only split. Its cooked target atoms and
model scoring differ, so the two reports are not a controlled before/after
comparison. See [CMUdict workflows](CMUDICT_COMPARISON.md).

The additional witnesses use substantial filtered WikiPron pronunciation
lexica, identified by immutable source revisions and file hashes. Whole NFC-casefold spelling
groups remain together during deterministic train/dev/test splitting, while
model inputs preserve their original case. This prevents differently cased
versions of a spelling from leaking between splits. We report
each language separately, without averaging unrelated phone inventories into a
single headline. Small task slices are diagnostic material, not the release
comparison.

French requires a further distinction: these are dictionary-entry pronunciations,
not contextual speech. A single spelling provides no following-word context for
liaison selection. The comparison excludes entire French pronunciation variants containing the
explicit linking marker `‿`, retaining other unmarked variants. The upstream
phone-inventory notes identify this as a boundary/liaison annotation rather
than a phoneme. We record the exclusion counts rather than silently delete
the marker from the phone sequence. The comparison neither invents liaison
annotations nor enables Phonebox's French input sentinel. Its score therefore does not measure contextual liaison accuracy.
A separate liaison-aware experiment would need explicitly annotated training
phones and appropriate contextual test material; see
[French liaison annotations](DATA.md#french-liaison-annotations).

Shared preparation preserves NFC spelling and the supplied phone token
boundaries. CMUdict spellings are lowercased. The IPA lexica retain case and
accents, with no additional stress stripping. These experiments deliberately
use identity letter processing, rather than the shipped locale rewriting rules.
The loader checks the prepared split intersections before training. It does not
silently repair a contaminated split or merge accented spellings through a
model-specific locale rule.

CART uses epsilon scattering and local context decision trees; the multigram
model learns variable-length letter-to-phone units and their sequence model.
Sequitur supplies an independent joint-sequence implementation, and
Phonetisaurus uses joint n-grams compiled into a weighted finite-state
transducer. All four predict from the model alone: dictionary fallback and
reference-pronunciation substitution are disabled.

Native CART and multigram alignment allow at most **100** iterations with
existing stopping criteria: CART stops at zero changes (or below its configured
change ratio); multigram stops when relative observed likelihood change is
below `1e-4`. Results store the actual full trace, stopping criterion, iteration
count and cap-censoring in `training.convergence`. Multigram likelihoods are
observed before each M-step, not a separate final-model rescore. A successful
run or a fixed cap alone does not demonstrate convergence. Older cap-10 native
snapshots are initial diagnostics, not results of this final protocol.

Multigram inference uses the complete unit-LM sequence score including EOS;
alignment q supplies candidate support, without an extra unigram factor.
Results record model/LM versions and the saved scoring identifier dynamically.
Historical q-plus-LM snapshots retain their original semantics and source pins.

The benchmark fixes model settings before testing. Sequitur's order is selected
from orders 1–3 using development phone error rate, then word error rate, then
the smaller order. Each order uses the upstream convergence test with minimum
20 and initial maximum 100 iterations; hitting the cap triggers a fresh restart
of that order with maximum 200, before test decoding. Results record remaining
iteration limits rather than asserting convergence. See the
[toolchain protocol](BENCHMARK_TOOLCHAINS.md) for controls and stopping evidence.
Phonetisaurus uses its upstream example order of eight.
These choices are bounded baseline configurations, not an exhaustive tuning
study. Test results do not determine hyperparameters.

Word error rate accepts any recorded reference pronunciation. Phone error rate
uses edit distance divided by reference phones; for multiple references it
selects the minimum-edit reference, breaking ties deterministically. Missing
and empty predictions remain in the test denominator. Training admission
counts distinguish supplied examples from examples a model can align.

## Dataset quality checks

The selected WikiPron files have roughly 89,000 Italian and 97,000 French
pronunciations before preparation. Both pass strict two-column parsing, NFC
spelling checks, and upstream phone-inventory validation, with no malformed
rows or duplicate spelling/pronunciation pairs in the pinned source.
The loader verifies the exact file hashes and reports subsequent exclusions,
variant deduplication, spelling-group membership, and split counts.

Upstream "filtered" means phone tokens passed an inventory whitelist; it is
not a guarantee that every dictionary transcription is correct. The source
includes inflected forms, proper names, loanwords, and pronunciation variants.
We retain phonetic distinctions and do not clean the test references according
to model errors. Entries with more phones than letters remain in the shared
data: each trainer's actual admission or alignment failures are reported.

## Held-out accuracy and dictionary-backed pronunciation

**Every reported WER and PER is on held-out spellings with dictionary lookup
disabled.** Training, development, and test spellings are disjoint after shared
preparation. Development references select only the declared model setting;
test references are used only for scoring. This measures generalization to
unseen dictionary entries, not the error rate of a deployed dictionary-backed
pronouncer on running text.

Phonebox's CART deployment path supports a compact exceptions list. After
training, it stores corrections for admitted training words that the model
mispronounces, plus a selected reference pronunciation for words with multiple
variants. A correctly predicted word with one reference needs no stored
correction. Selection favors the reference closest to the model's prediction;
it does not preserve every alternative or necessarily the first input variant.
The automatic list is built from admitted alignment data, so omitted training
entries must not be described as memorized. Lookup also depends on the saved
preprocessing and matching spelling keys.

The multigram model supports an explicit exceptions dictionary, but its basic
trainer does not automatically memorize the full lexicon. Phonetisaurus's
application layer likewise supports dictionary substitution; the benchmark
uses its model-only decoder and supplies no reference lexicon. The benchmark
therefore measures the same unseen-word task for all four systems.

In a dictionary-backed application, frequent covered words can use stored
pronunciations, and the model handles uncovered words. High **token coverage**
can therefore make practical error much lower than held-out dictionary-type
error. Conceptually, overall accuracy is

`coverage × dictionary accuracy + (1 − coverage) × OOV accuracy`.

The quantities must refer to the same target material and scoring policy.
Word-frequency distributions motivate retaining frequent forms, but a
held-out lexicon test does not measure running-text coverage or its OOV
accuracy. A stored pronunciation also need not be the appropriate variant
in every context. We report no deployment accuracy estimate without that
additional evidence.

## Developer API and CLI

Install the developer environment with `python -m pip install -e '.[dev]'`.
The benchmark driver adds no runtime dependencies. External training programs
need their own native builds; a Python decoder wheel alone is insufficient.
The [verified Linux toolchain recipes](BENCHMARK_TOOLCHAINS.md) record source pins.

Run each experiment in its own ignored directory:

```sh
phonebox compare benchmark --dataset italian --system cart \
  --work-dir .cache/benchmarks/italian-cart \
  --output .cache/benchmarks/italian-cart.json
phonebox compare benchmark --dataset italian --system multigram \
  --work-dir .cache/benchmarks/italian-multigram \
  --output .cache/benchmarks/italian-multigram.json
phonebox compare benchmark --dataset italian --system sequitur \
  --sequitur-executable .cache/toolchains/sequitur/venv/bin/g2p.py \
  --work-dir .cache/benchmarks/italian-sequitur \
  --output .cache/benchmarks/italian-sequitur.json
phonebox compare benchmark --dataset italian --system phonetisaurus \
  --phonetisaurus-prefix .cache/toolchains/phonetisaurus/prefix \
  --work-dir .cache/benchmarks/italian-phonetisaurus \
  --output .cache/benchmarks/italian-phonetisaurus.json
phonebox compare benchmark-report .cache/benchmarks/italian-*.json \
  --output .cache/benchmarks/italian.md
```

Repeat with `--dataset french` or `--dataset cmudict`. For the second CMUdict
condition, add `--remove-stress` and use new work and result paths. The CLI help
lists the required external executables and cache options. Progress goes to
standard error; aggregate measurements go to the requested JSON file.

The CLI delegates to the same library functions:

```python
from pathlib import Path
from phonebox.eval.benchmark_data import load_dataset
from phonebox.eval.benchmark import run_benchmark
from phonebox.eval.benchmark_report import render_benchmark_report

data = load_dataset("italian", Path(".cache/benchmarks/data"))
result = run_benchmark(data, "cart", Path(".cache/benchmarks/api-italian-cart"))
markdown = render_benchmark_report([result])
```

`PreparedDataset.metadata` records source licenses, hashes, preparation, and
split populations. `run_benchmark` returns a JSON-serializable result. The report
renderer checks matching prepared splits and test populations, rejects duplicate
system rows, and labels unmeasured systems. Models, logs, and raw predictions
stay in the experiment directory; only aggregate result JSON and its rendering
belong in the public results snapshot.

## Interpreting and replicating results

Compare measured systems within the same prepared dataset and split hashes.
Record model settings, source and dependency versions, training admission,
prediction failures, model size, and timing alongside WER and PER. Training plus export is reported as the complete model-production time, including
any declared development selection. Runs may use different platforms; the JSON
records each environment. These timings are descriptive, not a controlled speed
ranking.

Published paper results are relevant context. A replication claim additionally
requires matching the original dictionary revision, split, normalization,
reference handling, metric definition, and training protocol. Running newer
CMUdict through the authors' software is a new controlled comparison; it does
not by itself replicate the historical number. Where those details cannot be
matched, published scores must be labeled separately rather than inserted into
the measured-results table.

## Sources, credit, and licensing

The benchmark downloads data into an ignored cache. The package contains source
manifests and aggregate measurements, not the downloaded dictionaries, trained
external models, or word-level predictions. Data licenses remain separate from
Phonebox's software license.

- [CMUdict](https://github.com/cmusphinx/cmudict) provides its own
  [license](https://github.com/cmusphinx/cmudict/blob/74790861f652b15e4ac49015a90074ad62a27690/LICENSE).
  The loader records the exact revision and digest.
- [WikiPron](https://github.com/CUNY-CL/wikipron/tree/d282e848a211ea31cfd730f0ced8bc8cdab9e83d)
  provides filtered Italian and French pronunciation lexica. Cite
  [Lee et al. (2020), *Massively multilingual pronunciation mining with
  WikiPron*](https://aclanthology.org/2020.lrec-1.521/).
  Its software is Apache 2.0, but the pronunciation data have
  [Wiktionary's separate licensing terms](https://en.wiktionary.org/wiki/Wiktionary:Copyrights),
  including CC BY-SA 4.0. Preserve source attribution and the applicable data
  license when sharing prepared data; do not apply the software license to it.
- [Sequitur](https://github.com/sequitur-g2p/sequitur-g2p) is GPL-2.0-only.
  Cite Bisani and Ney (2008), *Joint-sequence models for grapheme-to-phoneme
  conversion*, Speech Communication 50(5), 434–451,
  [doi:10.1016/j.specom.2008.01.002](https://doi.org/10.1016/j.specom.2008.01.002).
- [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) is BSD-3-Clause.
  Cite Novak, Minematsu, and Hirose (2016), *Phonetisaurus: Exploring
  grapheme-to-phoneme conversion with joint n-gram models in the WFST framework*,
  Natural Language Engineering 22(6), 907–938.
  Its training toolchain uses [OpenFst](https://www.openfst.org/) (Apache 2.0)
  and [MITLM](https://github.com/mitlm/mitlm) (BSD-3-Clause).
  For MITLM, cite Hsu and Glass (2008), *Iterative Language Model Estimation:
  Efficient Data Structure & Algorithms*, Interspeech, 841–844.

External tools are installed separately for development and invoked as
subprocesses. They are not Phonebox runtime dependencies or vendored package
contents. Retain their licenses and notices with any redistributed copies.
This build uses MITLM; it does not require SRILM.
