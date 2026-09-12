# G2P Data Directory

Downloaded CMUdict files live in `data/` by default. The repository does not
ship this training dictionary.

## Quick Start

```bash
# Fetch CMUdict
phonebox dict fetch cmudict

# Build complete PocketSphinx-style model
phonebox recipe cmudict pocketsphinx -o g2p.py
```

## Directory Structure

```
data/
├── manifest.json              # Metadata about downloaded dictionary
└── cmudict/                   # CMU Pronouncing Dictionary
    ├── cmudict.dict          # Full dictionary with stress markers
    ├── cmudict_nostress.dict # Processed without stress (for PocketSphinx)
    ├── cmudict.phones        # Phone inventory
    ├── cmudict.symbols       # Symbol list
    ├── LICENSE               # License information
    └── README                # CMUdict README
```

## Data Source

### CMUdict
- **Repository**: https://github.com/cmusphinx/cmudict
- **License**: [CMUdict license](https://github.com/cmusphinx/cmudict/blob/master/LICENSE). Carnegie Mellon
  retains copyright; redistribution must retain or reproduce the notice,
  conditions, and disclaimer as specified in that license.
- **Format**: Arpabet phonemes with stress markers (0, 1, 2)
- **Language**: English (US)
- **Size**: ~135K entries

Fetch follows the upstream `master` branch; it does not claim a fixed snapshot.
A successful fetch requires both the dictionary and its `LICENSE` notice. Keep
that downloaded notice with redistributed dictionary data. A manifest created
for manually supplied dictionary files links the upstream license but only
advertises `license_file` / “see LICENSE” when the local notice file exists;
it cannot establish the provenance of manually supplied files.

## Processing Dictionaries

Dictionary processing (stress removal, deduplication, normalization) is exposed
through the `Dictionary` Python class:

```python
from phonebox import Dictionary

d = Dictionary("data/cmudict/cmudict.dict", locale="en_US")
d.process(remove_stress=True, output="data/cmudict/cmudict_nostress.dict")
```

## Training G2P Models

### Train from CMUdict

```bash
# With stress markers
phonebox train --locale en_US --phoneset cmu \
  --lexicon data/cmudict/cmudict.dict -o models/en_US_stress.g2p.gz

# Without stress markers (PocketSphinx-style)
phonebox train --locale en_US --phoneset cmu \
  --lexicon data/cmudict/cmudict_nostress.dict \
  -o models/en_US_nostress.g2p.gz \
  --remove-stress
```

### French liaison annotations

To train a CART model to emit liaison markers or other annotated pronunciation
tails, the training lexicon must include those symbols explicitly on the
**pronunciation side**. Do not assume an ordinary French pronunciation lexicon
contains these annotations: inspect its format and inventory before training.
The symbols must also survive the selected phone preprocessing.

The French policy supplies input-side padding (`liaison_pad="#"`) for alignment;
it does not create pronunciation-side liaison annotations. The input sentinel
and the target symbols have different roles: the target marker need not be a
literal `#`. Do not append `#` to dictionary spellings yourself to enable this
behavior. Selecting `--locale fr` or `--locale fr_FR` alone does not annotate a
lexicon or teach the model when liaison should occur.

A lexicon containing only ordinary word pronunciations supplies no explicit
liaison-marker targets. If liaison output is required, prepare an annotated
lexicon with a documented marker convention and compatible phone processing.
Word-level G2P output is not a sentence-level decision about whether to realize
liaison before the next word. These data requirements apply equally to the
Python training API and CLI; bundling does not add missing annotations.

## Complete Pipeline

```bash
# One-step: fetch, train, and bundle
phonebox recipe cmudict pocketsphinx -o g2p.py

# Or step by step:
phonebox dict fetch cmudict
phonebox train --locale en_US --phoneset cmu \
  --lexicon data/cmudict/cmudict.dict -o model.g2p.gz --remove-stress
```

## Manifest

The `manifest.json` file tracks all downloaded dictionaries:

```json
{
  "dictionaries": [
    {
      "file": "cmudict/cmudict.dict",
      "has_stress": true,
      "language": "en_US",
      "phoneset": "arpabet",
      "source": "cmudict"
    }
  ],
  "sources": [
    {
      "license": "CMUdict license (see LICENSE)",
      "license_file": "cmudict/LICENSE",
      "license_url": "https://github.com/cmusphinx/cmudict/blob/master/LICENSE",
      "name": "CMUdict",
      "path": "cmudict/",
      "repository": "https://github.com/cmusphinx/cmudict"
    }
  ]
}
```

## Notes

- The `data/` directory is gitignored to avoid committing large dictionary files
- CMUdict is fetched dynamically from GitHub
- Language code: `en_US` for US English
- Phoneset: Arpabet with stress markers (0=none, 1=primary, 2=secondary)
