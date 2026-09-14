# Phonebox

Train grapheme-to-phoneme (G2P) models, generate pronunciations, and review
pronunciation lexicons through a shared Python API and command-line interface.
Phonebox provides compact CART decision trees and n:m multigram models, with
saved locale preprocessing for repeatable inference.

- **Train and predict:** learn from your lexicon, preserve or remove stress,
  and generate single or n-best pronunciations.
- **Review dictionaries:** score pronunciation variants, inspect low-support
  entries, and reorder variants by model likelihood with TSV or JSON output.
- **Process orthography:** resolve locale names and use versioned ICU/CLDR
  exemplar data and spelling policies. Orthographic inventories are separate
  from the phone inventory you choose for training.
- **Deploy compact trees:** export a standalone Python predictor that uses only
  the standard library. The full training package has dependencies; multigram
  models use the full package.
- **Measure results:** compare CART and multigram models on held-out CMUdict
  with recorded data, settings, admission counts, and model sizes.

Phonebox is **alpha**. Each `0.X.0` release may break APIs, commands, or model
workflows; patch releases within a minor line are intended to remain compatible.
See the [0.3.0 changes](https://github.com/lenzo-ka/phonebox/blob/main/CHANGELOG.md)
and [upgrade guide](https://github.com/lenzo-ka/phonebox/blob/main/docs/RELEASING.md).

## Install

Requires Python 3.11 or newer:

```bash
pip install phonebox
```

The full package uses cartlet and ICU. Installation requires compatible backend
wheels for your Python and platform; see
[installation scope](https://github.com/lenzo-ka/phonebox/blob/main/docs/RELEASING.md#installation-scope).
Generated standalone tree bundles use only the Python standard library.

Optional extras:

```bash
pip install 'phonebox[config]'          # YAML training configuration
pip install 'phonebox[sklearn]'         # Optional scikit-learn trainer
```

## Alignment and the CART workflow

The CART path uses **epsilon scattering**, a term coined by Kevin Lenzo in the
[1998 work with Alan Black and Vincent Pagel](#references). Epsilon denotes an
empty phone emission: a spelling position can participate in an alignment
without producing a sound. The current workflow is:

1. **Prepare the units.** Apply the selected spelling normalization and rewrite
   rules, optional stress removal, and configured letter/phone joins. A joined
   unit can represent several letters or phones.
2. **Align by epsilon scattering.** Enumerate placements of empty phone targets
   across the processed input positions while preserving phone order. Score
   these alternatives with letter–phone probabilities, select the best alignment
   for each entry, and re-estimate the probabilities from those choices. Iterate
   until the stopping criterion or iteration limit is reached.
3. **Train the tree.** Turn each aligned position into a spelling-context window
   and its phone target, count repeated vectors, and train CART to predict a
   phone unit or epsilon. Boundary padding supplies context at word edges.
4. **Predict.** Reuse the saved preprocessing and context windows, predict the
   targets, remove epsilon emissions, and expand joined phone units into the
   output pronunciation.

This CART aligner requires at least as many processed input positions as phone
units; entries exceeding its alignment-combination limit are also omitted.
Locale-specific input padding, such as the French liaison sentinel, is separate
from empty phone emissions. Liaison targets still require explicit
[pronunciation-side annotations](https://github.com/lenzo-ka/phonebox/blob/main/docs/DATA.md#french-liaison-annotations).
The multigram path instead learns n:m joint units and decodes their sequences.

Epsilon scattering illustrates a broader possibilia approach: make admissible
alternatives explicit, then use evidence to choose among them. Here the
alternatives are alignment candidates within the configured model and limits.

## Train an English model and bundle it

This recipe downloads CMUdict, trains a decision tree, and writes a standalone
predictor. Training takes time; this is not a download of a pretrained model.
The PocketSphinx preset removes stress. The `tts` preset retains primary stress;
add `--keep-secondary` to retain secondary stress too.

```bash
phonebox recipe cmudict pocketsphinx -o g2p.py
python g2p.py "Hello, world!"
```

Use the generated predictor from Python:

```python
from g2p import G2PPredictor

predictor = G2PPredictor.from_embedded()
for word, phones in predictor.pronounce_text("Hello, world!"):
    print(word, " ".join(phones))
```

See [data preparation](https://github.com/lenzo-ka/phonebox/blob/main/docs/DATA.md)
for CMUdict licensing and French liaison annotation requirements, and
[bundling](https://github.com/lenzo-ka/phonebox/blob/main/docs/BUNDLING.md)
for deployment details.

## Train from your own lexicon

Dictionary lines contain a spelling followed by whitespace-separated phones.
Choose the phone inventory to match your data. For CMU/ARPAbet input:

```bash
phonebox train --locale en --phoneset cmu \
  --lexicon words.dict -o model.g2p.gz
phonebox pronounce hello world -m model.g2p.gz
```

The equivalent Python workflow uses the same training defaults:

```python
from phonebox import G2P, train_g2p

training = train_g2p(
    "words.dict", locale="en", phoneset="cmu", output="model.g2p.gz"
)
predictor = G2P(model="model.g2p.gz")
print(predictor.pronounce("hello"))
```

Primary training preserves stress, uses the native serial CART trainer, and
prunes with a 5% validation split. Pass `--remove-stress` or
`remove_stress=True` when desired. The default phone inventory is IPA; select
`cmu` explicitly for CMU stress syntax. Locale names accept bare language codes
and case-insensitive hyphen or underscore forms such as `en`, `en-US`, and
`en_US`. See [locale resolution and ICU data](https://github.com/lenzo-ka/phonebox/blob/main/docs/EXEMPLARS.md).

For n:m alignment and decoding:

```bash
phonebox train-multigram --locale en --phoneset cmu \
  --lexicon words.dict -o multigram.g2p.gz
phonebox pronounce hello -m multigram.g2p.gz
```

Keep the exported multigram sidecars together. See
[library and CLI workflows](https://github.com/lenzo-ka/phonebox/blob/main/docs/WORKFLOWS.md)
for structured training results and the corresponding `train_multigram` API.

## Review and reorder pronunciations

Review an existing lexicon against a trained CART model:

```bash
# Numeric TSV, lowest support first
phonebox dict review words.dict -m model.g2p.gz -o review.tsv

# Each spelling's variants most likely first, with CMUdict-style numbering
phonebox dict review words.dict -m model.g2p.gz --format dict -o ranked.dict

# Repeat bare spellings instead of numbering variants
phonebox dict review words.dict -m model.g2p.gz --format dict \
  --no-number-senses -o ranked-bare.dict
```

Scores measure compatibility with the selected model, not pronunciation
correctness. A model trained on the reviewed words can memorize them; the
[review guide](https://github.com/lenzo-ka/phonebox/blob/main/docs/LEXICON_REVIEW.md)
explains model choice, unsupported sequences, TSV sorting, JSON provenance,
and the shared review APIs. General dictionary phone mapping and deduplication
are documented in [dictionary processing](https://github.com/lenzo-ka/phonebox/blob/main/docs/DICTIONARY_PROCESSING.md).

## Accuracy and model choice

CART uses spelling-context features and decision trees; multigram learns joint
n:m spelling/phone units and decodes sequences. Their accuracy and export sizes
depend on the lexicon, preprocessing, and training settings. The
[measured G2P comparisons](https://github.com/lenzo-ka/phonebox/blob/main/docs/G2P_BENCHMARKS.md)
report shared-data results for Phonebox and independent implementations,
including CMUdict with and without stress. Each row records held-out error
rates, training settings and source provenance. Timings across different shared
machines are descriptive, not a speed ranking. See the
[CMUdict workflow guide](https://github.com/lenzo-ka/phonebox/blob/main/docs/CMUDICT_COMPARISON.md)
for the distinction between these measurements and older locale-aware snapshots.

## Documentation and help

Start with the [documentation guide](https://github.com/lenzo-ka/phonebox/blob/main/docs/README.md)
for tutorials, API/CLI workflows, evaluation, and historical experiments.

```bash
phonebox --help
phonebox train --help
phonebox dict review --help
```

The CLI includes training, prediction, bundling, dictionary processing,
validation, evaluation, and prepared alignment/vector workflows. Command help
is the authoritative option list.

## Development

```bash
git clone https://github.com/lenzo-ka/phonebox.git
cd phonebox
pip install -e '.[dev]'
```

See the [release checks](https://github.com/lenzo-ka/phonebox/blob/main/docs/RELEASING.md#release-preparation)
for validation. Contributions should include tests for behavior changes and
keep Python APIs, CLI help, and documentation consistent.

The [methods guide](https://github.com/lenzo-ka/phonebox/blob/main/docs/G2P_METHODS.md)
explains how the compared models learn and predict. The [developer benchmark protocol](https://github.com/lenzo-ka/phonebox/blob/main/docs/REPRODUCIBLE_BENCHMARKS.md)
compares both Phonebox models with independently run Sequitur and Phonetisaurus
on pinned, shared data. It records preparation, split hashes, settings, and
source provenance so subsequent implementations can be measured again.
Reported errors are for held-out words with dictionary lookup disabled. A
dictionary-backed pronouncer can use stored pronunciations for covered words;
its running-text accuracy depends on token coverage and appropriate variant
selection, and is not measured by the held-out score alone.

## References

Background on letter–phone alignment, decision-tree G2P, and lexicon compression:

- Alan W. Black, Kevin Lenzo, and Vincent Pagel (1998).
  [“Issues in Building General Letter to Sound Rules.”](https://www.isca-archive.org/ssw_1998/black98_ssw.html)
  *Third ESCA Workshop on Speech Synthesis*, pp. 77–80.
  Describes lexicon-derived alignments, CART pronunciation rules, and evaluation
  on unseen words.
- Vincent Pagel, Kevin Lenzo, and Alan W. Black (1998).
  [“Letter to Sound Rules for Accented Lexicon Compression.”](https://www.isca-archive.org/icslp_1998/pagel98_icslp.html)
  *5th International Conference on Spoken Language Processing (ICSLP 1998)*,
  paper 0561. [doi:10.21437/ICSLP.1998-39](https://doi.org/10.21437/ICSLP.1998-39).
  Examines decision trees for joint phone/stress prediction and lexicon compression.

## Credits and license

Phonebox is by [Kevin Lenzo](https://github.com/lenzo-ka), building on
[CMU G2P research](https://www.cs.cmu.edu/afs/cs.cmu.edu/user/lenzo/html/areas/t2p/).

Source code uses the [BSD 2-Clause License](https://github.com/lenzo-ka/phonebox/blob/main/LICENSE).
Generated ICU/CLDR locale data uses the
[Unicode License v3](https://github.com/lenzo-ka/phonebox/blob/main/LICENSE-UNICODE).
[Third-party notices](https://github.com/lenzo-ka/phonebox/blob/main/THIRD_PARTY_NOTICES.md)
record the source, modifications, and pinned data versions.
