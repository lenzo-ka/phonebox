# Dictionary build records

Use distinct output filenames and retain a build record when training from
different dictionary revisions. Phonebox stores an input content identifier
as `dict_hash`; it does not automatically assign Git revision IDs or custom
version suffixes. See [dictionary content identifiers](DICT_VERSION_SCHEMA.md)
for the exact hashing behavior.

## Train and inspect a self-contained model

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from phonebox import G2P, train_g2p

with TemporaryDirectory() as directory:
    root = Path(directory)
    dictionary = root / "dictionary.txt"
    dictionary.write_text("cat K AE T\ndog D AO G\n", encoding="utf-8")
    result = train_g2p(
        dictionary,
        locale="en_US",
        phoneset="cmu",
        prune=False,  # A tiny demonstration has no validation holdout.
        output=root / "model.g2p.gz",
    )
    loaded = G2P(model=result.output_path)
    print(result.model.dict_hash)
    assert loaded("cat") == result.model.pronounce("cat")
```

For your own dictionary, the equivalent installed command is:

```console
phonebox train --locale en_US --phoneset cmu --lexicon dictionary.txt -o model.g2p.gz
```

Stress is preserved by default. Add `--remove-stress` when CMU stress digits
should be removed. The primary workflow prunes by default; the tiny Python
example explicitly opts out.

## Record source identity separately

For a public source such as [CMUdict](https://github.com/cmusphinx/cmudict),
record the repository URL, full commit ID, and digest of the exact input file.
After local edits, record a new file digest and a descriptive build label in
your own metadata. Renaming an input or output file does not add a suffix to
the embedded `dict_hash`.

For each model, also retain the Phonebox revision, dependency versions,
phoneset, optional stress setting, preprocessing options, and training options.
The [CMUdict benchmark snapshot](CMUDICT_COMPARISON.md) demonstrates this
separate provenance record.
