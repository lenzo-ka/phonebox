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
See the [0.2.0 changes](https://github.com/lenzo-ka/phonebox/blob/main/CHANGELOG.md)
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

## Train an English model and bundle it

This recipe downloads CMUdict, trains a decision tree, and writes a standalone
predictor. Training takes time; this is not a download of a pretrained model.
The PocketSphinx preset removes stress; the `tts` preset preserves it.

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
[reproducible CMUdict comparison](https://github.com/lenzo-ka/phonebox/blob/main/docs/CMUDICT_COMPARISON.md)
reports both models with and without stress, held-out error rates, and exact
snapshot provenance. It is a measurement of the recorded revision, not a claim
that every future release has the same results.

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

## Credits and license

Phonebox is by [Kevin Lenzo](https://github.com/lenzo-ka), building on
[CMU G2P research](https://www.cs.cmu.edu/afs/cs.cmu.edu/user/lenzo/html/areas/t2p/).

Source code uses the [BSD 2-Clause License](https://github.com/lenzo-ka/phonebox/blob/main/LICENSE).
Generated ICU/CLDR locale data uses the
[Unicode License v3](https://github.com/lenzo-ka/phonebox/blob/main/LICENSE-UNICODE).
[Third-party notices](https://github.com/lenzo-ka/phonebox/blob/main/THIRD_PARTY_NOTICES.md)
record the source, modifications, and pinned data versions.
