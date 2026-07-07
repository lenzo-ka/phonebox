# G2P Data Directory

This directory contains the CMU Pronouncing Dictionary for training English G2P models.

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
- **License**: Public Domain
- **Format**: Arpabet phonemes with stress markers (0, 1, 2)
- **Language**: English (US)
- **Size**: ~135K entries

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
phonebox model build en_US data/cmudict/cmudict.dict -o models/en_US_stress.g2p.gz

# Without stress markers (PocketSphinx-style)
phonebox model build en_US data/cmudict/cmudict_nostress.dict \
  -o models/en_US_nostress.g2p.gz \
  --remove-stress
```

## Complete Pipeline

```bash
# One-step: fetch, train, and bundle
phonebox recipe cmudict pocketsphinx -o g2p.py

# Or step by step:
phonebox dict fetch cmudict
phonebox model build en_US data/cmudict/cmudict.dict -o model.g2p.gz --remove-stress
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
      "license": "Public Domain",
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
